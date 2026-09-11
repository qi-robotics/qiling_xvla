"""Latency-aware asynchronous action-chunk scheduling.

This module deliberately has no ROS dependency so the queue timing policy can
be tested offline.  It owns only *future* policy actions.  The ROS bridge owns
the 50 Hz MIT reference loop and consumes due entries from this queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimedAction:
    """One physical-scale policy action scheduled on the 30 Hz timebase."""

    due_monotonic: float
    values: np.ndarray
    request_id: int
    model_step: int


@dataclass(frozen=True)
class InferenceRequest:
    """Identity and timing of the observation sent to the inference worker."""

    request_id: int
    observation_monotonic: float
    queued_actions: int


@dataclass(frozen=True)
class ChunkAdmission:
    """Diagnostic information returned after adding a model chunk to the queue."""

    request_id: int
    latency_sec: float
    latency_steps: int
    first_model_step: int
    scheduled_steps: int
    queue_before: int
    queue_after: int
    protected_steps: int
    replaced_steps: int
    blended_steps: int
    first_due_monotonic: float


class LatencyAwareActionScheduler:
    """Maintain a short protected prefix and refresh the replaceable queue tail.

    The protected prefix bridges worker latency and is never changed by a newly
    arriving chunk. Farther, stale predictions are replaced by the new chunk.
    A short overlap blend avoids a joint-position jump at the replacement
    boundary. This is receding-horizon queue refresh, not model-level RTC.
    """

    def __init__(
        self,
        *,
        action_period_sec: float,
        execution_horizon_steps: int,
        prefetch_watermark_steps: int,
        max_inference_latency_steps: int,
        protected_prefix_steps: int = 3,
        blend_steps: int = 4,
        action_dimension: int = 8,
    ) -> None:
        self.action_period_sec = float(action_period_sec)
        self.execution_horizon_steps = int(execution_horizon_steps)
        self.prefetch_watermark_steps = int(prefetch_watermark_steps)
        self.max_inference_latency_steps = int(max_inference_latency_steps)
        self.protected_prefix_steps = int(protected_prefix_steps)
        self.blend_steps = int(blend_steps)
        self.action_dimension = int(action_dimension)
        if self.action_period_sec <= 0.0:
            raise ValueError("action_period_sec must be positive")
        if self.execution_horizon_steps <= 0:
            raise ValueError("execution_horizon_steps must be positive")
        if not 0 <= self.prefetch_watermark_steps < self.execution_horizon_steps:
            raise ValueError("prefetch watermark must satisfy 0 <= watermark < execution horizon")
        if self.max_inference_latency_steps <= 0:
            raise ValueError("max_inference_latency_steps must be positive")
        if not 0 <= self.protected_prefix_steps < self.execution_horizon_steps:
            raise ValueError("protected prefix must satisfy 0 <= prefix < execution horizon")
        if self.blend_steps < 0:
            raise ValueError("blend_steps must be non-negative")
        if self.action_dimension <= 0:
            raise ValueError("action_dimension must be positive")

        self._queue: deque[TimedAction] = deque()
        self._pending: InferenceRequest | None = None
        self._next_request_id = 1

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def inference_in_flight(self) -> bool:
        return self._pending is not None

    @property
    def pending_request(self) -> InferenceRequest | None:
        return self._pending

    def should_request_inference(self) -> bool:
        """Whether the bridge should provide a new observation to the worker."""
        return self._pending is None and len(self._queue) <= self.prefetch_watermark_steps

    def begin_inference_request(self, observation_monotonic: float) -> InferenceRequest:
        if not self.should_request_inference():
            raise RuntimeError("inference request is not currently needed")
        request = InferenceRequest(
            request_id=self._next_request_id,
            observation_monotonic=float(observation_monotonic),
            queued_actions=len(self._queue),
        )
        self._next_request_id += 1
        self._pending = request
        return request

    def cancel_pending_request(self, request_id: int | None = None) -> bool:
        """Cancel an incomplete worker request, for example after disconnect."""
        if self._pending is None:
            return False
        if request_id is not None and self._pending.request_id != int(request_id):
            return False
        self._pending = None
        return True

    def clear(self) -> None:
        self._queue.clear()
        self._pending = None

    def pop_due(self, now_monotonic: float) -> list[TimedAction]:
        """Return every policy action due by ``now_monotonic`` in order."""
        due: list[TimedAction] = []
        now = float(now_monotonic)
        while self._queue and self._queue[0].due_monotonic <= now:
            due.append(self._queue.popleft())
        return due

    def peek_next(self) -> TimedAction | None:
        """Return the earliest future action without consuming it.

        The 50 Hz reference generator uses this together with the most recent
        consumed 30 Hz action to interpolate only between timestamps that the
        scheduler has already admitted and ordered.
        """
        if not self._queue:
            return None
        action = self._queue[0]
        return TimedAction(
            due_monotonic=action.due_monotonic,
            values=action.values.copy(),
            request_id=action.request_id,
            model_step=action.model_step,
        )

    def queued_actions(self) -> list[TimedAction]:
        """Return a defensive snapshot of the future queue for diagnostics/tests."""
        return [
            TimedAction(
                due_monotonic=action.due_monotonic,
                values=action.values.copy(),
                request_id=action.request_id,
                model_step=action.model_step,
            )
            for action in self._queue
        ]

    def admit_chunk(
        self,
        *,
        request_id: int,
        actions: np.ndarray,
        arrival_monotonic: float,
    ) -> ChunkAdmission:
        """Replace the non-protected tail with a timestamp-aligned new slice.

        ``actions[i]`` denotes the policy's prediction for approximately
        ``observation_time + i * action_period``. Only the nearest configured
        prefix is retained. The selected model step matches the first replaceable
        timestamp, and its first joint targets are blended against the discarded
        tail (or the final protected target) before the queue is refilled.
        """
        if self._pending is None:
            raise ValueError("no inference request is pending")
        if self._pending.request_id != int(request_id):
            raise ValueError(
                f"stale action chunk request_id={request_id}; "
                f"expected {self._pending.request_id}")
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != self.action_dimension:
            raise ValueError(f"expected non-empty [N,{self.action_dimension}] action chunk, got {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("action chunk contains non-finite values")

        arrival = float(arrival_monotonic)
        observation = self._pending.observation_monotonic
        if arrival < observation:
            raise ValueError("action chunk arrival timestamp precedes its observation")
        latency_sec = arrival - observation
        latency_steps = self._nearest_step(latency_sec)
        if latency_steps > self.max_inference_latency_steps:
            raise ValueError(
                f"inference latency {latency_sec:.3f}s ({latency_steps} policy steps) exceeds "
                f"limit {self.max_inference_latency_steps}")

        existing = list(self._queue)
        queue_before = len(existing)
        protected_count = min(self.protected_prefix_steps, queue_before)
        protected = existing[:protected_count]
        replaced_tail = existing[protected_count:]
        first_due = (
            arrival
            if not protected
            else max(arrival, protected[-1].due_monotonic + self.action_period_sec)
        )
        first_model_step = max(0, self._nearest_step(first_due - observation))
        needed = self.execution_horizon_steps - protected_count
        if first_model_step + needed > chunk.shape[0]:
            raise ValueError(
                f"chunk has {chunk.shape[0]} steps but needs [{first_model_step}, "
                f"{first_model_step + needed}) to refill the horizon")

        replacement: list[TimedAction] = []
        blended_steps = 0
        for offset in range(needed):
            model_step = first_model_step + offset
            values = chunk[model_step].copy()
            if offset < self.blend_steps:
                baseline: np.ndarray | None = None
                if offset < len(replaced_tail):
                    baseline = replaced_tail[offset].values
                elif replaced_tail:
                    baseline = replaced_tail[-1].values
                elif protected:
                    baseline = protected[-1].values
                if baseline is not None:
                    alpha = float(offset + 1) / float(self.blend_steps + 1)
                    # Blend only arm joints. O6 remains a discrete policy output
                    # and is handled by its own hysteresis downstream.
                    values[:7] = (1.0 - alpha) * baseline[:7] + alpha * values[:7]
                    blended_steps += 1
            replacement.append(TimedAction(
                due_monotonic=first_due + offset * self.action_period_sec,
                values=values,
                request_id=int(request_id),
                model_step=model_step,
            ))
        self._queue = deque([*protected, *replacement])
        self._pending = None
        return ChunkAdmission(
            request_id=int(request_id), latency_sec=latency_sec, latency_steps=latency_steps,
            first_model_step=first_model_step, scheduled_steps=needed, queue_before=queue_before,
            queue_after=len(self._queue), protected_steps=protected_count,
            replaced_steps=len(replaced_tail), blended_steps=blended_steps,
            first_due_monotonic=first_due)

    def _nearest_step(self, duration_sec: float) -> int:
        # Deliberately avoid Python's banker's rounding at x.5.
        return int(np.floor(float(duration_sec) / self.action_period_sec + 0.5))
