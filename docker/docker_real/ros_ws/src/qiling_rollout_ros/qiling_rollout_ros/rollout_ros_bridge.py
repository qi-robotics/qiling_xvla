#!/usr/bin/env python3
"""Host-side ROS 2 bridge for a right-arm XVLA rollout worker.

The node intentionally keeps model inference out of the ROS 2 Python 3.10
process.  A Python 3.12 worker obtains the latest complete observation over a
localhost authenticated IPC connection and returns an action chunk.  This
node owns the safety boundary before a candidate action reaches the existing
MIT/DDS command path.
"""

from __future__ import annotations

from datetime import datetime
import json
from multiprocessing.connection import Listener
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from mit_msgs.msg import MITJointCommand, MITJointCommands, MITLowState
from qi.msg import HandsCmd
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import Trigger
import yaml

from qiling_rollout_ros.async_action_scheduler import LatencyAwareActionScheduler
from qiling_rollout_ros.reference_limiter import (
    AccelerationLimitedJointReference,
    interpolate_scheduled_targets,
)


ARM_DOF = 7
FINGER_DOF = 6


class RolloutRosBridge(Node):
    """Synchronise latest robot observation, worker IPC, and safe MIT output."""

    def __init__(self) -> None:
        super().__init__("qiling_rollout_ros_bridge")
        self.declare_parameter("config_file", "")
        self.declare_parameter("execution_mode", "")
        config_file = str(self.get_parameter("config_file").value)
        if not config_file:
            raise RuntimeError("config_file parameter is required")
        self.config_path = Path(config_file).expanduser().resolve()
        with self.config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        self.config: dict[str, Any] = loaded.get(self.get_name(), {}).get("ros__parameters", {})
        if not self.config:
            raise RuntimeError(f"missing {self.get_name()}.ros__parameters in {self.config_path}")

        execution = self.config["execution"]
        override_mode = str(self.get_parameter("execution_mode").value).strip().lower()
        self.execution_mode = override_mode or str(execution.get("mode", "shadow")).lower()
        if self.execution_mode not in {"shadow", "armed"}:
            raise RuntimeError("execution mode must be shadow or armed")
        self.command_period = 1.0 / max(1.0, float(execution["command_rate_hz"]))
        self.policy_action_period = 1.0 / max(1.0, float(execution["policy_action_rate_hz"]))
        self.policy_execution_horizon_steps = int(execution["policy_execution_horizon_steps"])
        self.policy_prefetch_watermark_steps = int(execution["policy_prefetch_watermark_steps"])
        self.max_inference_latency_steps = int(execution["max_inference_latency_steps"])
        self.policy_protected_prefix_steps = int(execution.get("policy_protected_prefix_steps", 3))
        self.policy_blend_steps = int(execution.get("policy_blend_steps", 4))
        self.state_timeout = max(0.02, float(execution["state_timeout_sec"]))
        self.image_timeout = max(0.02, float(execution["image_timeout_sec"]))
        self.action_timeout = max(0.02, float(execution["action_timeout_sec"]))
        self.home_transition_duration = max(0.1, float(execution["home_transition_duration_sec"]))
        self.home_move_duration = max(0.1, float(execution["home_move_duration_sec"]))
        self.home_settle_duration = max(0.0, float(execution["home_settle_duration_sec"]))
        self.home_settle_timeout = max(0.1, float(execution["home_settle_timeout_sec"]))
        self.home_transition_tolerance = max(0.0, float(execution["home_transition_tolerance_rad"]))
        self.home_tolerance = max(0.0, float(execution["home_tolerance_rad"]))
        self.rollout_start_delay = max(0.0, float(execution.get("rollout_start_delay_sec", 0.0)))
        self.finish_service_name = str(execution.get("finish_service", "/rollout/finish"))
        self.abort_service_name = str(execution.get("abort_service", "/rollout/abort"))
        self.max_rollout_duration = max(0.0, float(execution.get("max_rollout_duration_sec", 0.0)))

        topics = self.config["topics"]
        self.image_message_type = str(topics.get("image_message_type", "raw")).strip().lower()
        if self.image_message_type not in {"raw", "compressed"}:
            raise RuntimeError("topics.image_message_type must be raw or compressed")
        robot = self.config["robot"]
        self.body_motor_count = int(robot["body_motor_count"])
        self.leg_offset = int(robot["leg_offset"])
        self.left_offset = int(robot["left_arm_offset"])
        self.right_offset = int(robot["right_arm_offset"])
        self.left_home = self._vector(robot["left_home_rad"], ARM_DOF, "left_home_rad")
        self.right_home = self._vector(robot["right_home_rad"], ARM_DOF, "right_home_rad")
        self.left_transition = self._vector(
            robot["left_home_transition_rad"], ARM_DOF, "left_home_transition_rad")
        self.right_transition = self._vector(
            robot["right_home_transition_rad"], ARM_DOF, "right_home_transition_rad")
        lower = self._vector(robot["right_joint_lower_rad"], ARM_DOF, "right_joint_lower_rad")
        upper = self._vector(robot["right_joint_upper_rad"], ARM_DOF, "right_joint_upper_rad")
        margin = max(0.0, float(robot["joint_limit_margin_rad"]))
        self.right_lower = lower + margin
        self.right_upper = upper - margin
        if np.any(self.right_lower >= self.right_upper):
            raise RuntimeError("joint limit margin leaves no valid right-arm range")
        self.right_max_velocity = self._vector(
            robot["right_max_velocity_rad_s"], ARM_DOF, "right_max_velocity_rad_s")
        self.right_max_acceleration = self._vector(
            robot["right_max_acceleration_rad_s2"], ARM_DOF, "right_max_acceleration_rad_s2")
        if np.any(self.right_max_velocity <= 0.0) or np.any(self.right_max_acceleration <= 0.0):
            raise RuntimeError("right-arm maximum velocity and acceleration must be positive")
        self.command_kp = float(robot["command_kp"])
        self.command_kd = float(robot["command_kd"])

        gravity = self.config.get("gravity", {})
        self._gravity_enabled = bool(gravity.get("enabled", False))
        self._gravity_scale = max(0.0, float(gravity.get("torque_scale", 1.0)))
        self._gravity_limit_scale = max(0.0, float(gravity.get("torque_limit_scale", 1.0)))
        self._pin = None
        self._gravity_model = None
        self._gravity_data = None
        self._gravity_effort_limits: np.ndarray | None = None
        if self._gravity_enabled:
            try:
                import pinocchio as pin
                urdf_path = Path(str(gravity["urdf_path"])).expanduser()
                self._gravity_model = pin.buildModelFromUrdf(str(urdf_path))
                if self._gravity_model.nq != 14 or self._gravity_model.nv != 14:
                    raise RuntimeError(
                        f"gravity URDF must be the 14-DoF dual-arm model, got nq={self._gravity_model.nq}")
                self._gravity_data = self._gravity_model.createData()
                self._gravity_effort_limits = np.asarray(self._gravity_model.effortLimit, dtype=np.float64)
                self._pin = pin
            except Exception as error:
                raise RuntimeError(f"failed to initialise rollout gravity compensation: {error}") from error

        self.o6 = self.config["o6"]
        self.o6_enabled = bool(self.o6.get("enabled", True))
        self.o6_close_threshold = float(self.o6["close_threshold"])
        self.o6_open_threshold = float(self.o6["open_threshold"])
        if not 0.0 <= self.o6_open_threshold <= self.o6_close_threshold <= 1.0:
            raise RuntimeError("O6 thresholds must satisfy 0 <= open <= close <= 1")
        self.o6_open = self._integer_vector(self.o6["open_position"], FINGER_DOF, "open_position")
        self.o6_close = self._integer_vector(self.o6["close_position"], FINGER_DOF, "close_position")
        self.o6_speed = int(self.o6["speed"])

        self.ipc = self.config["ipc"]
        self.ipc_host = str(self.ipc["host"])
        self.ipc_port = int(self.ipc["port"])
        self.ipc_authkey = str(self.ipc["authkey"]).encode("utf-8")
        if self.ipc_host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("IPC must remain localhost-only")

        log_config = self.config["logging"]
        log_dir = Path(log_config["log_directory"]).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"rollout_{stamp}.jsonl"
        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.diagnostics_period = max(0.5, float(log_config.get("diagnostics_period_sec", 2.0)))

        self._lock = threading.RLock()
        self._cv_bridge = CvBridge()
        self._latest_images: dict[str, tuple[np.ndarray, float]] = {}
        self._latest_state: tuple[np.ndarray, np.ndarray, float] | None = None
        self._right_reference: np.ndarray | None = None
        self._right_reference_limiter = AccelerationLimitedJointReference(
            lower=self.right_lower,
            upper=self.right_upper,
            max_velocity=self.right_max_velocity,
            max_acceleration=self.right_max_acceleration,
        )
        self._last_reference_update_monotonic: float | None = None
        self._last_policy_action = None
        self._last_action_time: float | None = None
        self._action_scheduler = LatencyAwareActionScheduler(
            action_period_sec=self.policy_action_period,
            execution_horizon_steps=self.policy_execution_horizon_steps,
            prefetch_watermark_steps=self.policy_prefetch_watermark_steps,
            max_inference_latency_steps=self.max_inference_latency_steps,
            protected_prefix_steps=self.policy_protected_prefix_steps,
            blend_steps=self.policy_blend_steps,
            action_dimension=8,
        )
        self._right_o6_closed = False
        self._last_published_o6: bool | None = None
        self._worker_connected = False
        self._worker_last_action_latency_ms: float | None = None
        self._worker_errors = 0
        self._last_block_reason = "waiting_for_state"
        self._last_block_log = 0.0
        self._running = True
        self._phase = "SHADOW_ROLLOUT" if self.execution_mode == "shadow" else "WAITING_FOR_STATE"
        self._phase_started = time.monotonic()
        self._settled_since: float | None = None
        self._phase_start_left: np.ndarray | None = None
        self._phase_start_right: np.ndarray | None = None
        self._abort_hold_left: np.ndarray | None = None
        self._abort_hold_right: np.ndarray | None = None
        self._rollout_io_active = False
        self._ipc_listener: Listener | None = None
        self._ipc_thread: threading.Thread | None = None

        self._state_sub = self.create_subscription(
            MITLowState, str(topics["lower_state"]), self._state_callback, qos_profile_sensor_data)
        self._image_subs: list[Any] = []
        self._command_pub = self.create_publisher(MITJointCommands, str(topics["lower_command"]), 10)
        self._hands_pub = self.create_publisher(HandsCmd, str(topics["hands_command"]), 10)
        self._finish_service = self.create_service(Trigger, self.finish_service_name, self._finish_service_callback)
        self._abort_service = self.create_service(Trigger, self.abort_service_name, self._abort_service_callback)

        # In armed mode observation IPC and image subscriptions intentionally do
        # not exist until both arms have reached home.  State feedback remains
        # subscribed because it is required for the safe homing trajectory.
        if self.execution_mode == "shadow":
            self._activate_rollout_io()
        self._command_timer = self.create_timer(self.command_period, self._command_timer_callback)
        self._diagnostic_timer = self.create_timer(self.diagnostics_period, self._diagnostic_timer_callback)

        self._event("bridge_started", mode=self.execution_mode, config=str(self.config_path))
        self.get_logger().warn(
            f"Rollout bridge is {self.execution_mode.upper()}. "
            + ("No robot command will be published." if self.execution_mode == "shadow"
               else "Startup homing is enabled; ensure all teleop publishers are stopped."))

    def _set_phase(self, phase: str, reason: str) -> None:
        old_phase = self._phase
        self._phase = phase
        self._phase_started = time.monotonic()
        self._settled_since = None
        self._event("rollout_phase", old=old_phase, new=phase, reason=reason)
        self.get_logger().info(f"Rollout phase {old_phase} -> {phase}: {reason}")

    @staticmethod
    def _quintic(start: np.ndarray, end: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
        u = float(np.clip(elapsed / max(duration, 1.0e-6), 0.0, 1.0))
        blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        return start + blend * (end - start)

    def _activate_rollout_io(self) -> None:
        """Begin image ingestion and worker IPC only after armed home completes."""
        if self._rollout_io_active:
            return
        topics = self.config["topics"]
        image_type = CompressedImage if self.image_message_type == "compressed" else Image
        image_qos = (
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            if self.image_message_type == "compressed"
            else qos_profile_sensor_data
        )
        self._image_subs = [
            self.create_subscription(image_type, str(topics[key]), self._image_callback(name), image_qos)
            for name, key in (("head", "head_image"), ("left", "left_image"), ("right", "right_image"))
        ]
        self._ipc_listener = Listener((self.ipc_host, self.ipc_port), authkey=self.ipc_authkey)
        self._ipc_thread = threading.Thread(target=self._ipc_server_loop, name="xvla_ipc", daemon=True)
        self._ipc_thread.start()
        self._rollout_io_active = True
        self._event("rollout_io_started")

    def _deactivate_rollout_io(self, *, stop_state_subscription: bool) -> None:
        """Stop model I/O; ABORT keeps state feedback for the safe hold loop."""
        if not self._rollout_io_active:
            return
        for subscription in self._image_subs:
            self.destroy_subscription(subscription)
        self._image_subs.clear()
        if stop_state_subscription and self._state_sub is not None:
            self.destroy_subscription(self._state_sub)
            self._state_sub = None
        with self._lock:
            self._latest_images.clear()
            self._action_scheduler.clear()
        self._rollout_io_active = False
        self._event("rollout_io_stopped")

    @staticmethod
    def _vector(values: Any, size: int, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"{name} must contain {size} finite values")
        return vector

    @staticmethod
    def _integer_vector(values: Any, size: int, name: str) -> list[int]:
        vector = [int(v) for v in values]
        if len(vector) != size or any(v < 0 or v > 255 for v in vector):
            raise RuntimeError(f"{name} must contain {size} integers in [0, 255]")
        return vector

    def _event(self, event: str, **fields: Any) -> None:
        payload = {"time_unix_ns": time.time_ns(), "event": event, **fields}
        with self._lock:
            self._log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _image_callback(self, name: str):
        def callback(message: Image | CompressedImage) -> None:
            try:
                if isinstance(message, CompressedImage):
                    image = self._cv_bridge.compressed_imgmsg_to_cv2(message, desired_encoding="rgb8")
                else:
                    image = self._cv_bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
                image = np.ascontiguousarray(image)
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"expected HWC RGB image, got {image.shape}")
            except (CvBridgeError, ValueError, cv2.error) as error:
                self.get_logger().error(f"{name} image conversion failed: {error}")
                return
            with self._lock:
                self._latest_images[name] = (image, time.monotonic())
        return callback

    def _state_callback(self, message: MITLowState) -> None:
        positions = np.asarray(message.joint_states.position, dtype=np.float64)
        velocities = np.asarray(message.joint_states.velocity, dtype=np.float64)
        if positions.size != self.body_motor_count or velocities.size != self.body_motor_count:
            self.get_logger().error(
                f"Ignoring /human_lower_state: expected {self.body_motor_count} motors, "
                f"got position={positions.size} velocity={velocities.size}")
            return
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            self.get_logger().error("Ignoring /human_lower_state containing non-finite values")
            return
        with self._lock:
            self._latest_state = (positions, velocities, time.monotonic())
            if self._right_reference is None:
                self._reset_right_reference(positions[self.right_offset:self.right_offset + ARM_DOF])

    def _snapshot_observation(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._phase == "FINISHED":
                return {"type": "observation", "status": "finished", "reason": "rollout_complete"}
            if self._phase in {"ABORT_HOLD", "ABORT_FAULT"}:
                return {"type": "observation", "status": "finished", "reason": "rollout_aborted"}
            if not self._rollout_io_active or self._phase not in {"SHADOW_ROLLOUT", "ROLLOUT"}:
                return {"type": "observation", "status": "not_ready", "reason": f"phase_{self._phase.lower()}"}
            if not self._action_scheduler.should_request_inference():
                return {
                    "type": "observation",
                    "status": "not_ready",
                    "reason": "action_buffered" if not self._action_scheduler.inference_in_flight else "inference_in_flight",
                    "queued_actions": self._action_scheduler.queue_size,
                }
            if self._latest_state is None:
                return {"type": "observation", "status": "not_ready", "reason": "missing_state"}
            positions, _, state_time = self._latest_state
            if now - state_time > self.state_timeout:
                return {"type": "observation", "status": "not_ready", "reason": "stale_state"}
            images: dict[str, np.ndarray] = {}
            for name in ("head", "left", "right"):
                item = self._latest_images.get(name)
                if item is None:
                    return {"type": "observation", "status": "not_ready", "reason": f"missing_{name}_image"}
                image, received = item
                if now - received > self.image_timeout:
                    return {"type": "observation", "status": "not_ready", "reason": f"stale_{name}_image"}
                images[name] = image.copy()
            right_q = positions[self.right_offset:self.right_offset + ARM_DOF].copy()
            inference_request = self._action_scheduler.begin_inference_request(now)
        return {
            "type": "observation",
            "status": "ready",
            "request_id": inference_request.request_id,
            "observation_monotonic": inference_request.observation_monotonic,
            "queued_actions_at_request": inference_request.queued_actions,
            "images": images,
            # This is exactly the training observation.state: right-arm q(7), in radians.
            "right_q": right_q,
        }

    def _accept_action_chunk(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._phase not in {"SHADOW_ROLLOUT", "ROLLOUT"}:
            return {"type": "ack", "ok": False, "reason": f"rollout_not_active:{self._phase}"}
        try:
            request_id = int(message["request_id"])
            actions = np.asarray(message["actions"], dtype=np.float64)
            model_latency = float(message.get("inference_latency_ms", -1.0))
            worker_completed = float(message["inference_completed_monotonic"])
        except (KeyError, TypeError, ValueError) as error:
            self._worker_errors += 1
            return {"type": "ack", "ok": False, "reason": f"malformed action chunk: {error}"}
        now = time.monotonic()
        with self._lock:
            try:
                admission = self._action_scheduler.admit_chunk(
                    request_id=request_id, actions=actions, arrival_monotonic=now)
            except ValueError as error:
                pending = self._action_scheduler.pending_request
                if pending is not None and pending.request_id == request_id:
                    self._action_scheduler.cancel_pending_request(request_id)
                self._worker_errors += 1
                return {"type": "ack", "ok": False, "reason": str(error)}
            self._last_action_time = now if admission.scheduled_steps else self._last_action_time
            self._worker_last_action_latency_ms = model_latency if np.isfinite(model_latency) and model_latency >= 0.0 else None
        self._event(
            "action_chunk_received",
            request_id=request_id,
            returned_steps=int(actions.shape[0]),
            scheduled_steps=admission.scheduled_steps,
            queue_before=admission.queue_before,
            queue_after=admission.queue_after,
            protected_steps=admission.protected_steps,
            replaced_steps=admission.replaced_steps,
            blended_steps=admission.blended_steps,
            latency_ms=round(admission.latency_sec * 1000.0, 3),
            latency_steps=admission.latency_steps,
            first_model_step=admission.first_model_step,
            inference_latency_ms=model_latency,
            worker_completed_monotonic=worker_completed,
            first_scheduled_due_monotonic=admission.first_due_monotonic,
        )
        return {
            "type": "ack",
            "ok": True,
            "accepted": admission.scheduled_steps,
            "returned_steps": int(actions.shape[0]),
            "latency_steps": admission.latency_steps,
            "first_model_step": admission.first_model_step,
            "protected_steps": admission.protected_steps,
            "replaced_steps": admission.replaced_steps,
            "blended_steps": admission.blended_steps,
        }

    def _ipc_server_loop(self) -> None:
        while self._running:
            connection = None
            try:
                connection = self._ipc_listener.accept()
                with self._lock:
                    self._worker_connected = True
                self._event("worker_connected")
                while self._running:
                    request = connection.recv()
                    if not isinstance(request, dict):
                        connection.send({"type": "error", "reason": "request must be a dict"})
                        continue
                    kind = request.get("type")
                    if kind == "next_observation":
                        connection.send(self._snapshot_observation())
                    elif kind == "action_chunk":
                        connection.send(self._accept_action_chunk(request))
                    elif kind == "worker_error":
                        self._worker_errors += 1
                        reason = str(request.get("reason", "unknown worker error"))
                        request_id = request.get("request_id")
                        with self._lock:
                            self._action_scheduler.cancel_pending_request(request_id)
                        self._event("worker_error", reason=reason)
                        connection.send({"type": "ack", "ok": True})
                    else:
                        connection.send({"type": "error", "reason": f"unknown request {kind}"})
            except (EOFError, ConnectionError, OSError) as error:
                if self._running:
                    self.get_logger().warn(f"XVLA IPC disconnected: {error}")
            except Exception as error:  # Keep ROS safety timer alive if a worker misbehaves.
                self._worker_errors += 1
                if self._running:
                    self.get_logger().error(f"XVLA IPC server error: {error}")
            finally:
                with self._lock:
                    self._worker_connected = False
                    self._action_scheduler.cancel_pending_request()
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass

    def _finish_service_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.execution_mode != "armed":
            response.success = False
            response.message = "finish is only available in armed mode; shadow never moves the robot"
            return response
        if self._phase == "FINISHED":
            response.success = True
            response.message = "rollout has already finished; subscriptions and commands are stopped"
            return response
        if self._phase != "ROLLOUT":
            response.success = False
            response.message = f"cannot finish while phase is {self._phase}"
            return response
        with self._lock:
            state = self._latest_state
            self._action_scheduler.clear()
            self._last_action_time = None
            self._last_policy_action = None
        now = time.monotonic()
        if state is None or now - state[2] > self.state_timeout:
            response.success = False
            response.message = "cannot start return-home: /human_lower_state is stale"
            return response
        positions, _, _ = state
        self._phase_start_left = positions[self.left_offset:self.left_offset + ARM_DOF].copy()
        self._phase_start_right = positions[self.right_offset:self.right_offset + ARM_DOF].copy()
        with self._lock:
            self._reset_right_reference(self._phase_start_right)
        self._set_phase("RETURN_DIRECT_TO_HOME", "finish service requested")
        response.success = True
        response.message = "worker actions stopped; returning both arms directly to home"
        return response

    def _begin_abort(self, positions: np.ndarray, reason: str) -> None:
        """Latch the physical pose and stop model I/O without moving the arms."""
        with self._lock:
            self._action_scheduler.clear()
            self._last_action_time = None
            self._last_policy_action = None
        self._abort_hold_left = positions[self.left_offset:self.left_offset + ARM_DOF].copy()
        self._abort_hold_right = positions[self.right_offset:self.right_offset + ARM_DOF].copy()
        with self._lock:
            self._reset_right_reference(self._abort_hold_right)
        self._deactivate_rollout_io(stop_state_subscription=False)
        self._set_phase("ABORT_HOLD", reason)

    def _abort_service_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.execution_mode != "armed":
            response.success = False
            response.message = "abort is only available in armed mode; shadow never moves the robot"
            return response
        if self._phase in {"ABORT_HOLD", "ABORT_FAULT"}:
            response.success = True
            response.message = f"rollout is already aborted ({self._phase})"
            return response
        if self._phase == "FINISHED":
            response.success = False
            response.message = "rollout has already completed"
            return response
        with self._lock:
            state = self._latest_state
        now = time.monotonic()
        if state is None or now - state[2] > self.state_timeout:
            response.success = False
            response.message = "cannot enter ABORT_HOLD: /human_lower_state is stale; use physical e-stop if unsafe"
            return response
        self._begin_abort(state[0], "manual abort service requested")
        response.success = True
        response.message = "model I/O stopped; both arms hold their abort-time measured pose"
        return response

    def _reset_right_reference(self, position: np.ndarray) -> None:
        """Synchronise the 50 Hz limiter whenever an external safe motion takes over."""
        self._right_reference = self._right_reference_limiter.reset(position)
        self._last_reference_update_monotonic = None

    def _next_interpolated_target(self, now: float) -> tuple[np.ndarray | None, bool]:
        """Return the 30 Hz policy target interpolated at this 50 Hz timestamp.

        Only two actions that have passed scheduler admission may be used.  If
        there is no later queued action, no policy trajectory is extrapolated:
        the reference limiter receives ``None`` and safely holds its current
        position until a new verified pair of targets is available.
        """
        action_used = False
        with self._lock:
            for timed_action in self._action_scheduler.pop_due(now):
                action = timed_action.values
                self._right_o6_closed = self._o6_state_from_value(float(action[7]), self._right_o6_closed)
                self._last_policy_action = timed_action
                action_used = True
            fresh = self._last_action_time is not None and now - self._last_action_time <= self.action_timeout
            previous = self._last_policy_action
            following = self._action_scheduler.peek_next()
            if not fresh or previous is None or following is None:
                return None, action_used
            try:
                target = interpolate_scheduled_targets(
                    previous.due_monotonic,
                    previous.values[:ARM_DOF],
                    following.due_monotonic,
                    following.values[:ARM_DOF],
                    now,
                )
            except ValueError as error:
                self._worker_errors += 1
                self._event("invalid_scheduled_interpolation", reason=str(error))
                return None, action_used
        return np.clip(target, self.right_lower, self.right_upper), action_used

    def _advance_right_reference(self, now: float, target: np.ndarray | None, measured_right: np.ndarray) -> np.ndarray:
        """Convert an interpolated 30 Hz policy target into one safe 50 Hz q_ref."""
        with self._lock:
            if self._right_reference is None:
                self._reset_right_reference(measured_right)
            if self._last_reference_update_monotonic is None:
                dt = self.command_period
            else:
                # A delayed timer must not create a proportionally larger
                # position step; a very early callback uses its real shorter
                # elapsed time, preserving physical velocity limits.
                dt = min(self.command_period, max(1.0e-4, now - self._last_reference_update_monotonic))
            self._last_reference_update_monotonic = now
            self._right_reference = self._right_reference_limiter.step(target, dt)
            return self._right_reference.copy()

    def _o6_state_from_value(self, value: float, previous: bool) -> bool:
        if value >= self.o6_close_threshold:
            return True
        if value <= self.o6_open_threshold:
            return False
        return previous

    def _publish_hands(self, right_closed: bool) -> None:
        if not self.o6_enabled or self._last_published_o6 == right_closed:
            return
        command = HandsCmd()
        command.mode = 1
        command.mode_ctrl = 1
        command.timestamp = int(time.time_ns())
        left_position = self.o6_open
        right_position = self.o6_close if right_closed else self.o6_open
        for hand_index, position in ((0, left_position), (1, right_position)):
            command.hands[hand_index].positions = position
            command.hands[hand_index].durations = [self.o6_speed] * FINGER_DOF
            command.hands[hand_index].mode = 1
            command.hands[hand_index].hand_id = hand_index
        self._hands_pub.publish(command)
        self._last_published_o6 = right_closed
        self._event("hands_command_published", right_closed=right_closed)

    def _gravity_efforts(self, positions: np.ndarray) -> np.ndarray:
        """Return 14 arm torque feedforwards from measured q, safely clipped."""
        if not self._gravity_enabled:
            return np.zeros(14, dtype=np.float64)
        try:
            arm_q = positions[self.left_offset:self.right_offset + ARM_DOF]
            gravity = np.asarray(
                self._pin.computeGeneralizedGravity(self._gravity_model, self._gravity_data, arm_q),
                dtype=np.float64)
            limits = self._gravity_effort_limits * self._gravity_limit_scale
            return np.clip(self._gravity_scale * gravity, -limits, limits)
        except Exception as error:
            self._gravity_enabled = False
            self._event("gravity_disabled_after_error", reason=str(error))
            self.get_logger().error(f"Gravity compensation disabled after error: {error}")
            return np.zeros(14, dtype=np.float64)

    def _publish_body_command(
        self,
        positions: np.ndarray,
        left_reference: np.ndarray,
        right_reference: np.ndarray,
        *,
        apply_gravity: bool,
    ) -> None:
        """Publish one complete 26-motor MIT message; legs remain measured."""
        command = MITJointCommands()
        command.stamp = self.get_clock().now().to_msg()
        command.commands = [MITJointCommand() for _ in range(self.body_motor_count)]
        for index, joint in enumerate(command.commands):
            joint.kp = 0.0
            joint.kd = 0.0
            joint.pos = float(positions[index])
            joint.vel = 0.0
            joint.eff = 0.0
        for index in range(ARM_DOF):
            left_joint = command.commands[self.left_offset + index]
            left_joint.kp = self.command_kp
            left_joint.kd = self.command_kd
            left_joint.pos = float(left_reference[index])
            right_joint = command.commands[self.right_offset + index]
            right_joint.kp = self.command_kp
            right_joint.kd = self.command_kd
            right_joint.pos = float(right_reference[index])
        if apply_gravity:
            gravity = self._gravity_efforts(positions)
            for index in range(ARM_DOF):
                command.commands[self.left_offset + index].eff = float(gravity[index])
                command.commands[self.right_offset + index].eff = float(gravity[ARM_DOF + index])
        self._command_pub.publish(command)

    @staticmethod
    def _max_arm_error(
        measured_left: np.ndarray,
        measured_right: np.ndarray,
        target_left: np.ndarray,
        target_right: np.ndarray,
    ) -> float:
        return float(max(
            np.max(np.abs(measured_left - target_left)),
            np.max(np.abs(measured_right - target_right)),
        ))

    def _enter_startup_motion(self, measured_left: np.ndarray, measured_right: np.ndarray) -> None:
        self._phase_start_left = measured_left.copy()
        self._phase_start_right = measured_right.copy()
        with self._lock:
            self._reset_right_reference(self._phase_start_right)
            self._last_policy_action = None
        self._right_o6_closed = False
        self._set_phase("MOVE_TO_TRANSITION", "fresh robot state received")
        # Home uses the neutral O6 state; the physical adapter maps this to its
        # configured open position and does not leave a previous grasp latched.
        self._publish_hands(False)

    def _settle_or_fault(
        self,
        now: float,
        measured_left: np.ndarray,
        measured_right: np.ndarray,
        target_left: np.ndarray,
        target_right: np.ndarray,
        tolerance: float,
        next_phase: str,
        reason: str,
    ) -> None:
        if self._max_arm_error(measured_left, measured_right, target_left, target_right) <= tolerance:
            if self._settled_since is None:
                self._settled_since = now
            elif now - self._settled_since >= self.home_settle_duration:
                self._set_phase(next_phase, reason)
                if next_phase == "ROLLOUT":
                    with self._lock:
                        self._reset_right_reference(self.right_home)
                        self._last_policy_action = None
                    self._activate_rollout_io()
        else:
            self._settled_since = None
        if now - self._phase_started >= self.home_settle_timeout:
            self._set_phase("FAULT", f"{self._phase} target not reached")

    def _home_references(
        self,
        now: float,
        measured_left: np.ndarray,
        measured_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Advance armed startup/return state and yield the commanded arm references."""
        phase = self._phase
        if phase == "WAITING_FOR_STATE":
            self._enter_startup_motion(measured_left, measured_right)
            return measured_left, measured_right
        if phase == "MOVE_TO_TRANSITION":
            assert self._phase_start_left is not None and self._phase_start_right is not None
            elapsed = now - self._phase_started
            left = self._quintic(self._phase_start_left, self.left_transition, elapsed, self.home_transition_duration)
            right = self._quintic(self._phase_start_right, self.right_transition, elapsed, self.home_transition_duration)
            if elapsed >= self.home_transition_duration:
                self._set_phase("SETTLE_AT_TRANSITION", "startup transition trajectory complete")
                return self.left_transition, self.right_transition
            return left, right
        if phase == "SETTLE_AT_TRANSITION":
            self._settle_or_fault(
                now, measured_left, measured_right, self.left_transition, self.right_transition,
                self.home_transition_tolerance, "MOVE_TO_HOME", "startup transition settled")
            return self.left_transition, self.right_transition
        if phase == "MOVE_TO_HOME":
            elapsed = now - self._phase_started
            left = self._quintic(self.left_transition, self.left_home, elapsed, self.home_move_duration)
            right = self._quintic(self.right_transition, self.right_home, elapsed, self.home_move_duration)
            if elapsed >= self.home_move_duration:
                self._set_phase("SETTLE_AT_HOME", "startup home trajectory complete")
                return self.left_home, self.right_home
            return left, right
        if phase == "SETTLE_AT_HOME":
            self._settle_or_fault(
                now, measured_left, measured_right, self.left_home, self.right_home,
                self.home_tolerance, "WAIT_BEFORE_ROLLOUT", "startup home settled; beginning rollout delay")
            return self.left_home, self.right_home
        if phase == "WAIT_BEFORE_ROLLOUT":
            if now - self._phase_started >= self.rollout_start_delay:
                self._set_phase("ROLLOUT", "home delay elapsed; starting observation and inference")
                with self._lock:
                    self._reset_right_reference(self.right_home)
                    self._last_policy_action = None
                self._activate_rollout_io()
            return self.left_home, self.right_home
        if phase == "RETURN_DIRECT_TO_HOME":
            assert self._phase_start_left is not None and self._phase_start_right is not None
            elapsed = now - self._phase_started
            left = self._quintic(self._phase_start_left, self.left_home, elapsed, self.home_move_duration)
            right = self._quintic(self._phase_start_right, self.right_home, elapsed, self.home_move_duration)
            if elapsed >= self.home_move_duration:
                self._set_phase("SETTLE_RETURN_HOME", "direct return-home trajectory complete")
                return self.left_home, self.right_home
            return left, right
        if phase == "SETTLE_RETURN_HOME":
            before = self._phase
            self._settle_or_fault(
                now, measured_left, measured_right, self.left_home, self.right_home,
                self.home_tolerance, "FINISHED", "return home settled; stopping rollout I/O")
            if before != "FINISHED" and self._phase == "FINISHED":
                self._deactivate_rollout_io(stop_state_subscription=True)
                self._publish_hands(False)
            return self.left_home, self.right_home
        return None

    def _command_timer_callback(self) -> None:
        now = time.monotonic()
        if self._phase in {"FINISHED", "FAULT"}:
            return
        with self._lock:
            state = self._latest_state
        if state is None or now - state[2] > self.state_timeout:
            if self._phase == "ABORT_HOLD":
                self._set_phase(
                    "ABORT_FAULT",
                    "lost /human_lower_state during ABORT_HOLD; operator must assess hardware safety",
                )
            self._block("stale_or_missing_state")
            return
        positions, _, _ = state
        measured_left = positions[self.left_offset:self.left_offset + ARM_DOF].copy()
        measured_right = positions[self.right_offset:self.right_offset + ARM_DOF].copy()
        self._last_block_reason = "ready"

        if self._phase == "ABORT_HOLD":
            assert self._abort_hold_left is not None and self._abort_hold_right is not None
            self._publish_body_command(
                positions, self._abort_hold_left, self._abort_hold_right, apply_gravity=True)
            return
        if self._phase == "ABORT_FAULT":
            return

        if self.execution_mode == "armed" and self._phase not in {"ROLLOUT", "FINISHED", "FAULT"}:
            references = self._home_references(now, measured_left, measured_right)
            if references is None or self._phase == "FAULT":
                self._block("home_fault")
                return
            left_reference, right_reference = references
            self._publish_body_command(positions, left_reference, right_reference, apply_gravity=False)
            return

        if (
            self.execution_mode == "armed"
            and self._phase == "ROLLOUT"
            and self.max_rollout_duration > 0.0
            and now - self._phase_started >= self.max_rollout_duration
        ):
            self._begin_abort(positions, f"rollout timeout after {self.max_rollout_duration:.1f} s")
            assert self._abort_hold_left is not None and self._abort_hold_right is not None
            self._publish_body_command(
                positions, self._abort_hold_left, self._abort_hold_right, apply_gravity=True)
            return
        interpolated_target, action_used = self._next_interpolated_target(now)
        right_reference = self._advance_right_reference(now, interpolated_target, measured_right)
        with self._lock:
            right_o6_closed = self._right_o6_closed

        if self.execution_mode == "shadow":
            if action_used:
                self._event(
                    "shadow_candidate",
                    right_target=right_reference.tolist(),
                    interpolated_policy_target=None if interpolated_target is None else interpolated_target.tolist(),
                    right_o6_closed=right_o6_closed,
                )
            return

        self._publish_body_command(positions, self.left_home, right_reference, apply_gravity=True)
        self._publish_hands(right_o6_closed)

    def _block(self, reason: str) -> None:
        self._last_block_reason = reason
        now = time.monotonic()
        if now - self._last_block_log >= 1.0:
            self._last_block_log = now
            self._event("command_blocked", reason=reason)
            self.get_logger().warn(f"Rollout command blocked: {reason}")

    def _diagnostic_timer_callback(self) -> None:
        now = time.monotonic()
        with self._lock:
            image_age = {
                name: round(now - data[1], 3)
                for name, data in self._latest_images.items()
            }
            state_age = None if self._latest_state is None else round(now - self._latest_state[2], 3)
            queue_size = self._action_scheduler.queue_size
            inference_in_flight = self._action_scheduler.inference_in_flight
            worker_connected = self._worker_connected
            latency = self._worker_last_action_latency_ms
        self.get_logger().info(
            f"mode={self.execution_mode} phase={self._phase} worker={worker_connected} state_age={state_age} "
            f"image_age={image_age} queued_actions={queue_size} inference_in_flight={inference_in_flight} "
            f"inference_ms={latency} "
            f"blocked={self._last_block_reason}")

    def destroy_node(self) -> bool:
        self._running = False
        if self._ipc_listener is not None:
            try:
                self._ipc_listener.close()
            except OSError:
                pass
        self._event("bridge_stopped")
        self._log_file.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RolloutRosBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
