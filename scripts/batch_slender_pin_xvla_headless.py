#!/usr/bin/env python3
"""Headless XVLA seed sweep for slender-pin insertion.

Runs the existing closed-loop script sequentially (no GUI, no video, no
compensation). Success = rollout metadata ``success`` / ``ever_inserted``
(wedged-in counts as success). Does not modify ``run_slender_pin_xvla_rollout.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / "scripts" / "run_slender_pin_xvla_rollout.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/last/pretrained_model",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets/slender_pin_lerobot_v3_pi05_v1",
    )
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--execution-horizon", type=int, default=32)
    parser.add_argument("--max-control-steps", type=int, default=800)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seed-end", type=int, default=300)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "outputs/slender_pin_xvla_rollout/batch_seed200_300_summary.json",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Temp per-seed dirs (metadata/npz). Deleted after each seed unless --keep-scratch.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep per-seed metadata/npz (still no video).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter that can import Isaac Sim (the qiling_isaac env).",
    )
    return parser.parse_args()


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    txt = path.with_suffix(".txt")
    lines = [
        f"checkpoint: {payload['checkpoint']}",
        f"dataset: {payload['dataset']}",
        f"execution_horizon: {payload['execution_horizon']}",
        f"seeds: {payload['seed_start']}..{payload['seed_end']} (inclusive)",
        f"done: {payload['n_done']}/{payload['n_total']}",
        f"success: {payload['n_success']}",
        f"fail: {payload['n_fail']}",
        f"error: {payload['n_error']}",
        f"interrupted: {payload['n_interrupted']}",
        f"success_rate_done: {payload['success_rate_done']}",
        f"success_rate_total: {payload['success_rate_total']}",
        f"success_seeds: {payload['success_seeds']}",
        f"fail_seeds: {payload['fail_seeds']}",
        f"error_seeds: {payload['error_seeds']}",
        f"elapsed_s: {payload['elapsed_s']}",
        "",
    ]
    txt.write_text("\n".join(lines), encoding="utf-8")


def episode_result(ep_dir: Path, returncode: int) -> dict[str, Any]:
    meta_path = ep_dir / "metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {"status": "ERROR", "success": False, "parse_error": True}

    status = str(meta.get("status") or "")
    success = bool(meta.get("success", False))
    if not status:
        if returncode == 0:
            status, success = "PASS", True
        elif returncode == 2:
            status, success = "FAIL", False
        elif returncode in (130, -2):
            status, success = "INTERRUPTED", False
        else:
            status, success = "ERROR", False

    return {
        "status": status,
        "success": success,
        "returncode": returncode,
        "ever_inserted": bool(meta.get("ever_inserted", success)),
        "finish_reason": meta.get("finish_reason"),
        "executed_control_steps": meta.get("executed_control_steps"),
        "release_damping_applied": meta.get("release_damping_applied"),
        "final_pin_to_socket_xy_m": meta.get("final_pin_to_socket_xy_m"),
        "final_pin_to_socket_z_m": meta.get("final_pin_to_socket_z_m"),
        "final_grasp": meta.get("final_grasp"),
    }


def build_summary(
    *,
    args: argparse.Namespace,
    results: dict[int, dict[str, Any]],
    started_at: float,
    finished: bool,
) -> dict[str, Any]:
    seeds = list(range(args.seed_start, args.seed_end + 1))
    success_seeds = sorted(s for s, r in results.items() if r.get("success"))
    fail_seeds = sorted(s for s, r in results.items() if r.get("status") == "FAIL")
    error_seeds = sorted(s for s, r in results.items() if r.get("status") == "ERROR")
    interrupted_seeds = sorted(s for s, r in results.items() if r.get("status") == "INTERRUPTED")
    n_done = len(results)
    n_success = len(success_seeds)
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "policy_device": args.policy_device,
        "execution_horizon": args.execution_horizon,
        "max_control_steps": args.max_control_steps,
        "headless": True,
        "record_video": False,
        "compensation": False,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "n_total": len(seeds),
        "n_done": n_done,
        "n_success": n_success,
        "n_fail": len(fail_seeds),
        "n_error": len(error_seeds),
        "n_interrupted": len(interrupted_seeds),
        "success_rate_done": (n_success / n_done) if n_done else None,
        "success_rate_total": n_success / len(seeds) if seeds else None,
        "finished": finished,
        "success_seeds": success_seeds,
        "fail_seeds": fail_seeds,
        "error_seeds": error_seeds,
        "interrupted_seeds": interrupted_seeds,
        "elapsed_s": round(time.time() - started_at, 1),
        "per_seed": {str(k): v for k, v in sorted(results.items())},
    }


def main() -> int:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise SystemExit("--seed-end must be >= --seed-start")
    if not ROLLOUT.is_file():
        raise SystemExit(f"missing rollout script: {ROLLOUT}")

    scratch_root = args.scratch_dir or Path(
        tempfile.mkdtemp(prefix="xvla_batch_scratch_", dir=str(ROOT / "outputs"))
    )
    scratch_root.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    started_at = time.time()
    interrupted = False

    print(
        f"[batch] seeds {args.seed_start}..{args.seed_end} "
        f"horizon={args.execution_horizon} headless no-video no-compensation",
        flush=True,
    )
    print(f"[batch] summary -> {args.summary_path.resolve()}", flush=True)

    try:
        for seed in range(args.seed_start, args.seed_end + 1):
            ep_dir = scratch_root / f"seed{seed:04d}"
            if ep_dir.exists():
                shutil.rmtree(ep_dir)
            cmd = [
                str(args.python),
                str(ROLLOUT),
                "--headless",
                "--checkpoint",
                str(args.checkpoint),
                "--dataset",
                str(args.dataset),
                "--policy-device",
                str(args.policy_device),
                "--execution-horizon",
                str(args.execution_horizon),
                "--max-control-steps",
                str(args.max_control_steps),
                "--seed",
                str(seed),
                "--out-dir",
                str(ep_dir),
            ]
            print(f"[batch] seed {seed} start", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
            rec = episode_result(ep_dir, proc.returncode)
            rec["elapsed_s"] = round(time.time() - t0, 1)
            results[seed] = rec
            print(
                f"[batch] seed {seed} {rec['status']} success={rec['success']} "
                f"returncode={proc.returncode} {rec['elapsed_s']}s "
                f"running {sum(1 for r in results.values() if r.get('success'))}/{len(results)}",
                flush=True,
            )
            write_summary(
                args.summary_path,
                build_summary(args=args, results=results, started_at=started_at, finished=False),
            )
            if not args.keep_scratch and ep_dir.exists():
                shutil.rmtree(ep_dir, ignore_errors=True)
            if rec["status"] == "INTERRUPTED" or proc.returncode in (130, -2):
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[batch] KeyboardInterrupt — writing partial summary", flush=True)

    summary = build_summary(
        args=args,
        results=results,
        started_at=started_at,
        finished=not interrupted and len(results) == (args.seed_end - args.seed_start + 1),
    )
    write_summary(args.summary_path, summary)

    n_total = summary["n_total"]
    n_success = summary["n_success"]
    print(
        f"[batch] done finished={summary['finished']} "
        f"success={n_success}/{n_total} "
        f"rate_total={summary['success_rate_total']} "
        f"rate_done={summary['success_rate_done']}",
        flush=True,
    )
    print(f"[batch] success_seeds={summary['success_seeds']}", flush=True)
    print(f"[batch] summary={args.summary_path.resolve()}", flush=True)
    print(f"[batch] txt={args.summary_path.with_suffix('.txt').resolve()}", flush=True)

    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)

    if interrupted:
        return 130
    return 0 if summary["finished"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
