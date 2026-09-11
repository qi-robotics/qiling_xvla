"""Continuous 50 Hz joint-reference generation for asynchronous policy actions.

This module is deliberately ROS-free.  Policy actions are 30 Hz position
samples, while the robot command loop is 50 Hz.  The bridge first linearly
interpolates consecutive *already scheduled* samples, then this class applies
per-joint position, velocity and acceleration limits before a MIT position
reference is emitted.
"""

from __future__ import annotations

import numpy as np


class AccelerationLimitedJointReference:
    """Track a position target without discontinuous reference position/velocity.

    A missing target is an intentional safety hold: position is frozen and the
    internal velocity is cleared.  This is used when no future policy sample
    remains or an action stream becomes stale; it prioritises a stable MIT
    position target over continuing an unverified trajectory.
    """

    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
    ) -> None:
        self.lower = self._finite_vector(lower, "lower")
        self.upper = self._finite_vector(upper, "upper")
        self.max_velocity = self._finite_vector(max_velocity, "max_velocity")
        self.max_acceleration = self._finite_vector(max_acceleration, "max_acceleration")
        if not (self.lower.shape == self.upper.shape == self.max_velocity.shape == self.max_acceleration.shape):
            raise ValueError("all joint-limit vectors must have the same shape")
        if np.any(self.lower >= self.upper):
            raise ValueError("lower limits must be strictly below upper limits")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0):
            raise ValueError("maximum velocity and acceleration must be positive")
        self.position: np.ndarray | None = None
        self.velocity = np.zeros_like(self.lower)

    @staticmethod
    def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite one-dimensional vector")
        return vector.copy()

    def reset(self, position: np.ndarray) -> np.ndarray:
        """Synchronise the generator to a known safe reference position."""
        value = self._finite_vector(position, "position")
        if value.shape != self.lower.shape:
            raise ValueError(f"position must have shape {self.lower.shape}, got {value.shape}")
        self.position = np.clip(value, self.lower, self.upper)
        self.velocity.fill(0.0)
        return self.position.copy()

    def hold(self) -> np.ndarray:
        """Freeze the reference immediately when no verified policy target exists."""
        if self.position is None:
            raise RuntimeError("reference generator must be reset before hold")
        self.velocity.fill(0.0)
        return self.position.copy()

    def step(self, target: np.ndarray | None, dt_sec: float) -> np.ndarray:
        """Advance one control tick toward ``target`` under velocity/acceleration limits."""
        if self.position is None:
            raise RuntimeError("reference generator must be reset before step")
        if target is None:
            return self.hold()
        dt = float(dt_sec)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_sec must be finite and positive")
        desired_position = self._finite_vector(target, "target")
        if desired_position.shape != self.lower.shape:
            raise ValueError(f"target must have shape {self.lower.shape}, got {desired_position.shape}")
        desired_position = np.clip(desired_position, self.lower, self.upper)

        error = desired_position - self.position
        # The stopping-speed bound starts decelerating early enough to avoid
        # reference overshoot for an unchanged target.  It is then approached
        # only at the configured maximum acceleration.
        stopping_speed = np.sqrt(2.0 * self.max_acceleration * np.abs(error))
        desired_velocity = np.sign(error) * np.minimum(self.max_velocity, stopping_speed)
        velocity_delta = np.clip(
            desired_velocity - self.velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        next_velocity = np.clip(
            self.velocity + velocity_delta,
            -self.max_velocity,
            self.max_velocity,
        )
        next_position = np.clip(self.position + next_velocity * dt, self.lower, self.upper)

        # At a hard joint bound, suppress only the outward velocity component.
        # This preserves the hard positional safety limit even if a policy
        # output points outside the URDF range.
        outward_lower = (next_position <= self.lower) & (next_velocity < 0.0)
        outward_upper = (next_position >= self.upper) & (next_velocity > 0.0)
        next_velocity[outward_lower | outward_upper] = 0.0
        self.position = next_position
        self.velocity = next_velocity
        return self.position.copy()


def interpolate_scheduled_targets(
    previous_due_monotonic: float,
    previous_target: np.ndarray,
    next_due_monotonic: float,
    next_target: np.ndarray,
    now_monotonic: float,
) -> np.ndarray:
    """Linearly interpolate two consecutive scheduled policy position samples."""
    previous = np.asarray(previous_target, dtype=np.float64)
    following = np.asarray(next_target, dtype=np.float64)
    if previous.shape != following.shape or previous.ndim != 1:
        raise ValueError("scheduled targets must be equal-size one-dimensional vectors")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(following)):
        raise ValueError("scheduled targets must be finite")
    interval = float(next_due_monotonic) - float(previous_due_monotonic)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("next scheduled target must be later than previous target")
    alpha = float(np.clip((float(now_monotonic) - float(previous_due_monotonic)) / interval, 0.0, 1.0))
    return previous + alpha * (following - previous)
