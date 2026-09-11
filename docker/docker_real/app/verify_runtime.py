#!/usr/bin/env python3
"""Fail-fast verification for the qi-reasoning container."""

from __future__ import annotations

import sys

import lerobot
import torch
import transformers


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, got {sys.version.split()[0]}")
    if not torch.__version__.startswith("2.7.0"):
        raise RuntimeError(f"expected torch 2.7.0, got {torch.__version__}")
    if transformers.__version__ != "5.3.0":
        raise RuntimeError(f"expected transformers 5.3.0, got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; install a compatible NVIDIA driver and NVIDIA Container Toolkit"
        )
    print(f"Python: {sys.version.split()[0]}")
    print(f"LeRobot: {getattr(lerobot, '__version__', '0.5.0')}")
    print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
