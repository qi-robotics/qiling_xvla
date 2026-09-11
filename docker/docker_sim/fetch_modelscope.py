#!/usr/bin/env python3
"""Download only the XVLA prefixes from the course ModelScope dataset.

The repo keno123/qi-studio_embodied_edu also contains bottleInBowl / smolVLA.
Those are not used by this product. This script lists xvla/ and pulls:

  - slender_pin_lerobot_v3_xvla_datasets  (~2.1 GB)
  - checkpoints/200000/pretrained_model   (~1.8 GB)

Optimizer training_state (~3.5 GB) is skipped unless --with-optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ID = "keno123/qi-studio_embodied_edu"
REVISION = "master"
ENDPOINT = os.environ.get("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn").rstrip("/")

REMOTE_DATASET = "xvla/slender_pin_lerobot_v3_xvla_datasets"
REMOTE_CKPT = (
    "xvla/xvla_slender_pin_full_from60k_plus200k_v1/"
    "checkpoints/200000/pretrained_model"
)
REMOTE_OPTIMIZER = (
    "xvla/xvla_slender_pin_full_from60k_plus200k_v1/"
    "checkpoints/200000/training_state"
)
REMOTE_CKPT_ROOT = "xvla/xvla_slender_pin_full_from60k_plus200k_v1"

LOCAL_DATASET = "datasets/slender_pin_lerobot_v3_xvla_v1"
LOCAL_OUTPUT = "outputs/xvla_slender_pin_full_from60k_plus200k_v1"

USER_AGENT = "qiling-xvla-fetch/1.0"
CHUNK = 8 * 1024 * 1024
RETRIES = 5


def api_get(url: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            print(f"[retry {attempt}/{RETRIES}] tree api: {exc}", flush=True)
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"ModelScope tree failed: {last_err}")


def list_blobs(root: str) -> list[dict]:
    blobs: list[dict] = []
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "Revision": REVISION,
                "Root": root,
                "Recursive": "true",
                "PageNumber": page,
                "PageSize": 100,
            }
        )
        url = f"{ENDPOINT}/api/v1/datasets/{REPO_ID}/repo/tree?{query}"
        payload = api_get(url)
        if payload.get("Code") not in (200, "200", None) and payload.get("Data") is None:
            raise RuntimeError(f"ModelScope tree failed: {payload}")
        data = payload.get("Data") or {}
        files = data.get("Files") or []
        for item in files:
            if item.get("Type") == "blob":
                blobs.append(item)
        total = int(data.get("TotalCount") or len(files))
        if page * 100 >= total or not files:
            break
        page += 1
    return blobs


def local_path(dest_root: Path, remote_path: str) -> Path:
    if remote_path == REMOTE_DATASET or remote_path.startswith(REMOTE_DATASET + "/"):
        rel = remote_path[len(REMOTE_DATASET) :].lstrip("/")
        return dest_root / LOCAL_DATASET / rel
    if remote_path.startswith(REMOTE_CKPT_ROOT + "/"):
        rel = remote_path[len(REMOTE_CKPT_ROOT) :].lstrip("/")
        return dest_root / LOCAL_OUTPUT / rel
    raise ValueError(f"unexpected remote path: {remote_path}")


def want_blob(path: str, with_optimizer: bool) -> bool:
    if path == REMOTE_DATASET or path.startswith(REMOTE_DATASET + "/"):
        return True
    if path == REMOTE_CKPT or path.startswith(REMOTE_CKPT + "/"):
        return True
    if with_optimizer and (
        path == REMOTE_OPTIMIZER or path.startswith(REMOTE_OPTIMIZER + "/")
    ):
        return True
    return False


def resolve_url(remote_path: str) -> str:
    quoted = urllib.parse.quote(remote_path, safe="/")
    return f"{ENDPOINT}/datasets/{REPO_ID}/resolve/{REVISION}/{quoted}"


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def download_file(url: str, dest: Path, expected: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected > 0 and dest.stat().st_size == expected:
        print(f"[skip] {dest} ({fmt_bytes(expected)})", flush=True)
        return

    tmp = dest.with_name(dest.name + ".part")
    start = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if start > 0 and expected > 0 and start < expected:
        headers["Range"] = f"bytes={start}-"
        print(f"[resume] {dest.name} from {fmt_bytes(start)}", flush=True)
    elif tmp.is_file():
        tmp.unlink()
        start = 0

    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                mode = "ab" if headers.get("Range") else "wb"
                written = start
                t0 = time.time()
                last_print = t0
                with open(tmp, mode) as fh:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if expected >= 50 * 1024 * 1024 and now - last_print >= 5:
                            pct = (100.0 * written / expected) if expected else 0.0
                            print(
                                f"  ... {dest.name} {fmt_bytes(written)}"
                                f"{f' / {fmt_bytes(expected)} ({pct:.0f}%)' if expected else ''}",
                                flush=True,
                            )
                            last_print = now
            size = tmp.stat().st_size
            if expected > 0 and size != expected:
                raise IOError(
                    f"size mismatch for {dest}: got {size}, expected {expected}"
                )
            tmp.replace(dest)
            print(f"[ok] {dest} ({fmt_bytes(size)})", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            print(f"[retry {attempt}/{RETRIES}] {dest.name}: {exc}", flush=True)
            headers.pop("Range", None)
            start = tmp.stat().st_size if tmp.is_file() else 0
            if start > 0:
                headers["Range"] = f"bytes={start}-"
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"failed to download {url}: {last_err}")


def ensure_last_symlink(dest_root: Path) -> None:
    ckpt_dir = dest_root / LOCAL_OUTPUT / "checkpoints"
    step = ckpt_dir / "200000"
    last = ckpt_dir / "last"
    if not (step / "pretrained_model").is_dir():
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if last.is_symlink() or last.exists():
        if last.is_dir() and not last.is_symlink():
            return
        last.unlink()
    last.symlink_to("200000")
    print(f"[link] {last} -> 200000", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=os.environ.get("QILING_ROOT", str(Path.home() / "X-VLA")),
        help="host work root (default: $QILING_ROOT or ~/X-VLA)",
    )
    parser.add_argument(
        "--with-optimizer",
        action="store_true",
        help="also download training_state (only needed to resume that 200k job)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list files, do not download")
    args = parser.parse_args()

    dest_root = Path(args.dest).expanduser().resolve()
    print(f"[fetch] repo {REPO_ID}  dest {dest_root}", flush=True)
    print("[fetch] only xvla/ prefixes; skip bottleInBowl and smolVLA", flush=True)

    blobs = [b for b in list_blobs("xvla") if want_blob(b.get("Path") or "", args.with_optimizer)]
    blobs.sort(key=lambda item: item.get("Path") or "")
    total = sum(int(item.get("Size") or 0) for item in blobs)
    print(f"[fetch] {len(blobs)} files, {fmt_bytes(total)}", flush=True)
    for item in blobs:
        path = item["Path"]
        size = int(item.get("Size") or 0)
        mapped = local_path(dest_root, path)
        print(f"  {fmt_bytes(size):>10}  {path} -> {mapped}", flush=True)

    if args.dry_run:
        return 0

    dest_root.mkdir(parents=True, exist_ok=True)
    for item in blobs:
        path = item["Path"]
        size = int(item.get("Size") or 0)
        download_file(resolve_url(path), local_path(dest_root, path), size)

    ensure_last_symlink(dest_root)
    print("[fetch] done", flush=True)
    print(f"  dataset    {dest_root / LOCAL_DATASET}", flush=True)
    print(
        f"  checkpoint {dest_root / LOCAL_OUTPUT / 'checkpoints/last/pretrained_model'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
