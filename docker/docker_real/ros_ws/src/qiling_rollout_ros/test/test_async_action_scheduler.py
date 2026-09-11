import numpy as np
import pytest

from qiling_rollout_ros.async_action_scheduler import LatencyAwareActionScheduler


def _chunk(offset: float, steps: int = 20) -> np.ndarray:
    values = np.arange(steps, dtype=np.float64) + offset
    return np.repeat(values[:, None], 8, axis=1)


def test_initial_chunk_is_latency_aligned_and_fills_horizon() -> None:
    scheduler = LatencyAwareActionScheduler(
        action_period_sec=0.1,
        execution_horizon_steps=6,
        prefetch_watermark_steps=4,
        max_inference_latency_steps=20,
        protected_prefix_steps=2,
        blend_steps=2,
    )

    request = scheduler.begin_inference_request(observation_monotonic=0.0)
    admission = scheduler.admit_chunk(
        request_id=request.request_id, actions=_chunk(0.0), arrival_monotonic=0.2)

    assert admission.first_model_step == 2
    assert admission.scheduled_steps == 6
    assert admission.protected_steps == 0
    assert admission.replaced_steps == 0
    assert admission.blended_steps == 0
    np.testing.assert_allclose(
        [scheduled.values[0] for scheduled in scheduler.queued_actions()],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )


def test_new_chunk_protects_near_term_replaces_tail_and_blends_overlap() -> None:
    scheduler = LatencyAwareActionScheduler(
        action_period_sec=0.1,
        execution_horizon_steps=6,
        prefetch_watermark_steps=4,
        max_inference_latency_steps=20,
        protected_prefix_steps=2,
        blend_steps=2,
    )

    first = scheduler.begin_inference_request(observation_monotonic=0.0)
    scheduler.admit_chunk(
        request_id=first.request_id, actions=_chunk(0.0), arrival_monotonic=0.2)
    scheduler.pop_due(0.301)

    second = scheduler.begin_inference_request(observation_monotonic=0.31)
    scheduler.pop_due(0.41)
    admission = scheduler.admit_chunk(
        request_id=second.request_id, actions=_chunk(100.0), arrival_monotonic=0.41)
    queued = scheduler.queued_actions()

    assert admission.first_model_step == 4
    assert admission.protected_steps == 2
    assert admission.replaced_steps == 1
    assert admission.blended_steps == 2
    assert len(queued) == 6
    assert [scheduled.request_id for scheduled in queued[:2]] == [first.request_id] * 2
    assert [scheduled.request_id for scheduled in queued[2:]] == [second.request_id] * 4

    # The first two replacement arm targets are blended with the discarded old
    # tail (7.0).  The O6 dimension remains discrete and is never interpolated.
    np.testing.assert_allclose(queued[2].values[:7], np.full(7, (2.0 * 7.0 + 104.0) / 3.0))
    np.testing.assert_allclose(queued[3].values[:7], np.full(7, (7.0 + 2.0 * 105.0) / 3.0))
    assert queued[2].values[7] == 104.0
    assert queued[3].values[7] == 105.0


def test_invalid_replacement_does_not_mutate_existing_queue() -> None:
    scheduler = LatencyAwareActionScheduler(
        action_period_sec=0.1,
        execution_horizon_steps=6,
        prefetch_watermark_steps=4,
        max_inference_latency_steps=20,
        protected_prefix_steps=2,
        blend_steps=2,
    )

    first = scheduler.begin_inference_request(observation_monotonic=0.0)
    scheduler.admit_chunk(
        request_id=first.request_id, actions=_chunk(0.0), arrival_monotonic=0.2)
    scheduler.pop_due(0.301)
    before = scheduler.queued_actions()
    second = scheduler.begin_inference_request(observation_monotonic=0.31)

    with pytest.raises(ValueError):
        scheduler.admit_chunk(
            request_id=second.request_id,
            actions=_chunk(100.0, steps=1),
            arrival_monotonic=0.32,
        )

    after = scheduler.queued_actions()
    assert len(after) == len(before)
    for lhs, rhs in zip(after, before):
        assert lhs.request_id == rhs.request_id
        assert lhs.model_step == rhs.model_step
        np.testing.assert_array_equal(lhs.values, rhs.values)
