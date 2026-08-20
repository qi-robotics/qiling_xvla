#!/usr/bin/env python3
"""Serve a local LeRobot XVLA checkpoint to the Isaac rollout process via stdio IPC.

Protocol: length-prefixed pickle messages over stdin/stdout (same as pi0.5 server).

Request types
-------------
  {"type": "ping"}
      → {"ok": True, "checkpoint": str}

  {"type": "schema"}
      → {"ok": True, "checkpoint": ..., "loader": ..., "input_features": ...,
         "output_features": ..., "chunk_size": int, "n_action_steps": int,
         "n_obs_steps": int, "normalization_mapping": ...,
         "input_state_dim": int}

  {"type": "reset"}
      Resets the XVLA action queue (call before each new episode).
      → {"ok": True}

  {"type": "predict",
   "state_bytes": bytes,          # float32 raw bytes, shape (state_dim,)
   "image_bytes": {name: bytes},  # uint8 raw bytes HWC 480×640×3
   "instruction": str,
   "robot_type": str}
      → {"ok": True, "action_bytes": bytes, "action_shape": [1, 10],
         "inference_ms": float}

  {"type": "shutdown"}
      → {"ok": True}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("chest", "left_wrist", "right_wrist")
HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

# XVLA was trained with these renamed keys (see policy_preprocessor.json)
CAMERA_RENAME_MAP = {
    "chest": "image",
    "left_wrist": "image2",
    "right_wrist": "image3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/last/pretrained_model"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def recv_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("client disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(stream) -> Any:
    size = HEADER.unpack(recv_exact(stream, HEADER.size))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size: {size}")
    return pickle.loads(recv_exact(stream, size))  # noqa: S301


def send_message(stream, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"response too large: {len(payload)}")
    stream.write(HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


class XVLAInference:
    """Wraps XVLAPolicy for Isaac rollout inference."""

    def __init__(self, checkpoint: Path, device: str, seed: int) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # registers "xvla"

        for filename in ("config.json", "model.safetensors",
                         "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        started = time.perf_counter()
        self.device = torch.device(device)

        # Load config and override runtime-only flags.
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        policy_config.device = str(self.device)

        print("[XVLA server] loading XVLAPolicy.from_pretrained", file=sys.stderr, flush=True)
        self.policy = XVLAPolicy.from_pretrained(
            checkpoint,
            config=policy_config,
            local_files_only=True,
        )
        self.policy.eval()
        self.policy.reset()

        # Load processor pipelines from checkpoint (includes rename_map + stats).
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {
                    "device": str(self.device),
                    "float_dtype": None,
                }
            },
            postprocessor_overrides={
                "device_processor": {"device": "cpu", "float_dtype": "float32"}
            },
        )

        self._state_dim = int(self.policy.config.input_features["observation.state"].shape[0])
        self._action_dim = int(self.policy.config.output_features["action"].shape[0])

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        elapsed = time.perf_counter() - started
        print(
            f"[XVLA server] loaded checkpoint={checkpoint} device={self.device} "
            f"state_dim={self._state_dim} action_dim={self._action_dim} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"in {elapsed:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[XVLA server] normalization={self.policy.config.normalization_mapping} "
            f"max_state_dim={self.policy.config.max_state_dim} "
            f"max_action_dim={self.policy.config.max_action_dim} "
            f"action_mode={self.policy.config.action_mode}",
            file=sys.stderr,
            flush=True,
        )

    def reset(self) -> None:
        """Reset the internal action queue (call before each episode)."""
        self.policy.reset()

    @torch.inference_mode()
    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Run one step of XVLA select_action and return a single (1, 10) action."""
        state_bytes = request["state_bytes"]
        state = np.frombuffer(state_bytes, dtype=np.float32).copy()
        if not np.isfinite(state).all():
            raise ValueError(f"XVLA state contains non-finite values: {state}")

        instruction = str(request.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("XVLA requires a non-empty language instruction")

        images_raw = request["image_bytes"]
        # Build observation dict using original camera names (preprocessor renames them).
        observation: dict[str, np.ndarray] = {
            "observation.state": state,
        }
        for name in CAMERA_NAMES:
            image = (
                np.frombuffer(images_raw[name], dtype=np.uint8)
                .reshape(480, 640, 3)
                .copy()
            )
            # Use original name here; preprocessor pipeline handles rename internally.
            observation[f"observation.images.{name}"] = image

        started = time.perf_counter()

        # Convert raw numpy obs → tensor dict the same way pi0.5 server does.
        from lerobot.policies.utils import prepare_observation_for_inference

        robot_type = str(request.get("robot_type", "S4_RIGHT_ARM_O6_FIXED_SOCKET_RJ45_XVLA"))
        obs_prepared = prepare_observation_for_inference(
            observation,
            self.device,
            task=instruction,
            robot_type=robot_type,
        )
        # Run through preprocessor (rename, tokenize, ImageNet normalise, device, normalise).
        batch = self.preprocessor(obs_prepared)

        # XVLA select_action: internally manages chunk queue.
        action_tensor = self.policy.select_action(batch)
        # action_tensor: (action_dim,) tensor after postprocessor unnorm is NOT applied
        # here; postprocessor handles unnormalization.
        action_post = self.postprocessor(action_tensor)
        if isinstance(action_post, dict):
            action_post = action_post["action"]

        # Model runs in bfloat16; numpy cannot convert that dtype directly.
        action_np = action_post.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if action_np.ndim == 1:
            action_np = action_np[np.newaxis, :]  # (1, 10)

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not np.isfinite(action_np).all():
            raise RuntimeError("XVLA returned NaN or inf")
        if action_np.shape[1] != self._action_dim:
            raise RuntimeError(f"Unexpected action shape: {action_np.shape}")

        return action_np, elapsed_ms


def serve(args: argparse.Namespace) -> int:
    checkpoint = resolve(args.checkpoint)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = XVLAInference(checkpoint, args.device, args.seed)

    if args.check_only:
        print("[XVLA server] --check-only: loaded successfully, exiting", file=sys.stderr, flush=True)
        return 0

    # Redirect stdout to binary for IPC; all text output goes to stderr.
    stdin_bin = sys.stdin.buffer
    stdout_bin = sys.stdout.buffer
    sys.stdout = sys.stderr

    print("[XVLA server] ready, waiting for requests", file=sys.stderr, flush=True)

    while True:
        try:
            request = recv_message(stdin_bin)
        except EOFError:
            print("[XVLA server] client closed connection", file=sys.stderr, flush=True)
            return 0
        except Exception as exc:
            print(f"[XVLA server] recv error: {exc}", file=sys.stderr, flush=True)
            return 1

        try:
            req_type = str(request.get("type", ""))
            if req_type == "ping":
                send_message(stdout_bin, {"ok": True, "checkpoint": str(checkpoint)})

            elif req_type == "schema":
                cfg = model.policy.config
                send_message(stdout_bin, {
                    "ok": True,
                    "checkpoint": str(checkpoint),
                    "loader": "XVLAPolicy.from_pretrained + make_pre_post_processors",
                    "input_features": {
                        k: {"type": str(v.type), "shape": list(v.shape)}
                        for k, v in cfg.input_features.items()
                    },
                    "output_features": {
                        k: {"type": str(v.type), "shape": list(v.shape)}
                        for k, v in cfg.output_features.items()
                    },
                    "chunk_size": int(cfg.chunk_size),
                    "n_action_steps": int(cfg.n_action_steps),
                    "n_obs_steps": int(cfg.n_obs_steps),
                    "normalization_mapping": {
                        str(key): str(value)
                        for key, value in dict(cfg.normalization_mapping).items()
                    },
                    "input_state_dim": model._state_dim,
                    "action_mode": str(cfg.action_mode),
                    "max_state_dim": int(cfg.max_state_dim),
                    "max_action_dim": int(cfg.max_action_dim),
                })

            elif req_type == "reset":
                model.reset()
                send_message(stdout_bin, {"ok": True})

            elif req_type == "predict":
                action, inference_ms = model.predict(request)
                send_message(stdout_bin, {
                    "ok": True,
                    "action_bytes": action.tobytes(order="C"),
                    "action_shape": list(action.shape),
                    "inference_ms": inference_ms,
                })

            elif req_type == "shutdown":
                send_message(stdout_bin, {"ok": True})
                print("[XVLA server] shutdown requested", file=sys.stderr, flush=True)
                return 0

            else:
                send_message(stdout_bin, {"ok": False, "error": f"unknown request type: {req_type!r}"})

        except Exception as exc:
            import traceback
            msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            print(f"[XVLA server] error handling {req_type!r}: {msg}", file=sys.stderr, flush=True)
            try:
                send_message(stdout_bin, {"ok": False, "error": msg})
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
