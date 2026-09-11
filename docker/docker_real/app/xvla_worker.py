#!/usr/bin/env python3
"""Python 3.12 XVLA inference worker for the host-side rollout bridge.

No ROS package is imported here. The bridge decides when the action buffer
needs replenishing; the worker then runs the saved XVLA preprocessors and
returns the model's complete physical-scale action chunk with its timestamps.
"""

from __future__ import annotations

import argparse
from multiprocessing.connection import Client
from pathlib import Path
import os
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        # NumPy has no bfloat16 dtype; XVLA is configured for bfloat16 on CUDA.
        # Convert only at the IPC boundary, after the saved postprocessor has
        # restored the physical action scale.
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def image_hwc_to_chw_uint8(image: np.ndarray, name: str) -> torch.Tensor:
    """Convert one bridge RGB frame to the checkpoint processor input format.

    The saved LeRobot ``to_batch_processor`` intentionally only batches Torch
    tensors.  Supplying an HWC NumPy frame therefore leaves it as ``[H,W,C]``
    and ImageNet normalization interprets the width as the channel dimension.
    Keep the frame uint8 here; the saved XVLA processor performs its own
    uint8-to-float conversion, normalization, CUDA transfer, and batching.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be an HWC RGB frame with 3 channels, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must use integer RGB values in [0,255], got {array.dtype}")
        if array.min() < 0 or array.max() > 255:
            raise ValueError(f"{name} values must be in [0,255]")
        array = array.astype(np.uint8, copy=False)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


class XVLAWorker:
    def __init__(self, config_path: Path) -> None:
        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        environment = self.config["environment"]
        os.environ.setdefault("HF_HOME", str(environment["hf_home"]))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(environment["cuda_visible_devices"]))
        model_config = self.config["model"]
        self.checkpoint = Path(model_config["checkpoint"]).expanduser().resolve()
        if not (self.checkpoint / "config.json").is_file():
            raise RuntimeError(f"checkpoint is not a LeRobot pretrained_model directory: {self.checkpoint}")
        self.task = str(model_config["task"]).strip()
        if not self.task:
            raise RuntimeError("model.task must be non-empty")
        self.warmup_enabled = bool(model_config.get("warmup_enabled", True))
        self.warmup_iterations = max(0, int(model_config.get("warmup_iterations", 1)))
        ipc = self.config["ipc"]
        self.address = (str(ipc["host"]), int(ipc["port"]))
        if self.address[0] not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("rollout IPC must remain localhost-only")
        self.authkey = str(ipc["authkey"]).encode("utf-8")
        self.retry_delay = max(0.1, float(ipc["retry_delay_sec"]))
        self.not_ready_delay = max(0.005, float(ipc["not_ready_delay_sec"]))

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        print(f"Loading XVLA checkpoint: {self.checkpoint}", flush=True)
        self.policy = XVLAPolicy.from_pretrained(
            pretrained_name_or_path=self.checkpoint,
            strict=False,
            local_files_only=True,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=self.checkpoint)
        self.policy.reset()
        print(
            f"XVLA worker ready: device={self.policy.config.device}, "
            f"chunk={self.policy.config.chunk_size}, returning full chunks to bridge",
            flush=True)
        if self.warmup_enabled and self.warmup_iterations:
            self._warmup()

    def _warmup(self) -> None:
        """Compile/initialize the complete deployed inference path before IPC.

        The first CUDA invocation is materially slower than steady-state XVLA
        inference. Warm up after model loading but before connecting to the
        bridge, so the first robot observation does not consume that latency.
        """
        synthetic_observation = {
            "images": {
                name: np.zeros((480, 640, 3), dtype=np.uint8)
                for name in ("head", "left", "right")
            },
            "right_q": np.zeros(7, dtype=np.float32),
        }
        for iteration in range(self.warmup_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _, latency_ms = self.infer(synthetic_observation)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(
                f"XVLA warm-up {iteration + 1}/{self.warmup_iterations}: "
                f"{latency_ms:.1f} ms",
                flush=True)
        self.policy.reset()

    def infer(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        images = observation["images"]
        right_q = np.asarray(observation["right_q"], dtype=np.float32)
        if right_q.shape != (7,) or not np.all(np.isfinite(right_q)):
            raise ValueError(f"right_q must contain 7 finite joint positions, got {right_q.shape}")
        raw_observation = {
            # Keep source names: saved checkpoint preprocessor performs the training rename map.
            "observation.images.head": image_hwc_to_chw_uint8(images["head"], "head image"),
            "observation.images.left": image_hwc_to_chw_uint8(images["left"], "left image"),
            "observation.images.right": image_hwc_to_chw_uint8(images["right"], "right image"),
            # Training observation.state is right-arm q(7), not velocity or action. The saved
            # XVLA model pads this vector internally to its 20-D proprio representation.
            "observation.state": torch.from_numpy(np.ascontiguousarray(right_q)),
            "task": self.task,
        }
        start = time.perf_counter()
        batch = self.preprocessor(raw_observation)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        steps = int(chunk.shape[1])
        actions: list[np.ndarray] = []
        for index in range(steps):
            action = self.postprocessor(chunk[:, index])
            array = as_numpy(action)
            array = np.asarray(array, dtype=np.float32).reshape(-1)
            if array.size != 8 or not np.all(np.isfinite(array)):
                raise RuntimeError(f"invalid postprocessed XVLA action at step {index}: shape={array.shape}")
            actions.append(array)
        return np.stack(actions, axis=0), (time.perf_counter() - start) * 1000.0

    def run(self) -> None:
        while True:
            connection = None
            try:
                connection = Client(self.address, authkey=self.authkey)
                print(f"Connected to rollout ROS bridge at {self.address}", flush=True)
                while True:
                    connection.send({"type": "next_observation"})
                    reply = connection.recv()
                    if reply.get("type") != "observation":
                        raise RuntimeError(f"unexpected IPC reply: {reply}")
                    if reply.get("status") == "finished":
                        print("Rollout bridge reported completion; XVLA worker exits.", flush=True)
                        return
                    if reply.get("status") != "ready":
                        time.sleep(self.not_ready_delay)
                        continue
                    try:
                        request_id = int(reply["request_id"])
                        observation_monotonic = float(reply["observation_monotonic"])
                        inference_started_monotonic = time.monotonic()
                        actions, latency_ms = self.infer(reply)
                        inference_completed_monotonic = time.monotonic()
                        connection.send({
                            "type": "action_chunk",
                            "request_id": request_id,
                            "observation_monotonic": observation_monotonic,
                            "actions": actions,
                            "inference_latency_ms": latency_ms,
                            "inference_started_monotonic": inference_started_monotonic,
                            "inference_completed_monotonic": inference_completed_monotonic,
                        })
                        ack = connection.recv()
                        if not ack.get("ok", False):
                            print(f"Bridge rejected action: {ack}", flush=True)
                        else:
                            print(
                                f"XVLA chunk returned={ack.get('returned_steps')} "
                                f"scheduled={ack.get('accepted')} "
                                f"model_start={ack.get('first_model_step')} "
                                f"latency_steps={ack.get('latency_steps')} "
                                f"inference={latency_ms:.1f} ms", flush=True)
                    except Exception as error:
                        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
                        print(f"XVLA inference error: {detail}", flush=True)
                        connection.send({
                            "type": "worker_error",
                            "request_id": reply.get("request_id"),
                            "reason": detail,
                        })
                        connection.recv()
                        time.sleep(self.retry_delay)
            except KeyboardInterrupt:
                return
            except (ConnectionError, EOFError, OSError, RuntimeError) as error:
                print(f"XVLA worker IPC error: {error}; reconnecting in {self.retry_delay:.1f}s", flush=True)
                time.sleep(self.retry_delay)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="/home/ub/program/qiling_television/rollout/config/xvla_rollout.yaml",
        help="Path to xvla_rollout.yaml",
    )
    args = parser.parse_args()
    XVLAWorker(Path(args.config).expanduser().resolve()).run()


if __name__ == "__main__":
    main()
