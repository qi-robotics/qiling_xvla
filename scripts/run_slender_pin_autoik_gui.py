#!/usr/bin/env python3
"""Run one complete slender-pin Auto-IK assembly episode in Isaac Sim GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# This launcher is called by the headless recorder with the repository root as
# cwd. Make the sibling scripts import explicit instead of relying on cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_handle_pin_grasp_gui import main


if __name__ == "__main__":
    print(
        f"[slender-launcher] main={main.__module__} "
        f"file={Path(main.__code__.co_filename).resolve()}",
        flush=True,
    )
    if "--task-config" not in sys.argv:
        sys.argv.extend(
            ["--task-config", "configs/task_slender_pin_insertion_right_arm.yaml"]
        )
    if "--complete-insertion" not in sys.argv:
        sys.argv.append("--complete-insertion")
    raise SystemExit(main())
