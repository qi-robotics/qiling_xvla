#!/usr/bin/env python3
"""YAML-driven launcher for LeRobot 0.5 XVLA full fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml


def option(key: str, value: object) -> str:
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    return f"--{key}={rendered}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    environment = config["environment"]
    dataset = config["dataset"]
    policy = config["xvla"]
    training = config["training"]
    logging = config["logging"]

    dataset_root = Path(dataset["root"])
    pretrained_path = Path(policy["pretrained_policy_path"])
    output_dir = Path(training["output_dir"])
    if not (dataset_root / "meta/info.json").is_file():
        raise RuntimeError(f"LeRobot dataset unavailable: {dataset_root}")
    if not (pretrained_path / "config.json").is_file():
        raise RuntimeError(f"XVLA base unavailable: {pretrained_path}")
    if output_dir.exists() and not args.dry_run:
        raise RuntimeError(f"training output already exists: {output_dir}")

    command = [
        str(Path(environment["python"]).with_name("lerobot-train")),
        option("dataset.repo_id", dataset["repo_id"]),
        option("dataset.root", dataset["root"]),
        option("dataset.video_backend", dataset["video_backend"]),
        option("rename_map", json.dumps(dataset["rename_map"])),
        option("output_dir", training["output_dir"]),
        option("job_name", training["job_name"]),
        option("seed", training["seed"]),
        option("batch_size", training["batch_size"]),
        option("num_workers", training["num_workers"]),
        option("steps", training["steps"]),
        option("log_freq", training["log_freq"]),
        option("save_freq", training["save_freq"]),
        option("eval_freq", training["eval_freq"]),
        option("save_checkpoint", training["save_checkpoint"]),
        option("wandb.enable", logging["wandb_enabled"]),
        option("wandb.project", logging["wandb_project"]),
        option("policy.path", policy["pretrained_policy_path"]),
        option("policy.device", policy["device"]),
        option("policy.dtype", policy["dtype"]),
        option("policy.action_mode", policy["action_mode"]),
        option("policy.max_action_dim", policy["max_action_dim"]),
        option("policy.chunk_size", policy["chunk_size"]),
        option("policy.n_action_steps", policy["n_action_steps"]),
        option("policy.num_image_views", policy["num_image_views"]),
        option("policy.empty_cameras", policy["empty_cameras"]),
        option("policy.freeze_vision_encoder", policy["freeze_vision_encoder"]),
        option("policy.freeze_language_encoder", policy["freeze_language_encoder"]),
        option("policy.train_policy_transformer", policy["train_policy_transformer"]),
        option("policy.train_soft_prompts", policy["train_soft_prompts"]),
        option("policy.optimizer_lr", training["learning_rate"]),
        option("policy.scheduler_warmup_steps", training["warmup_steps"]),
        option("policy.scheduler_decay_steps", training["decay_steps"]),
        option("policy.scheduler_decay_lr", training["decay_learning_rate"]),
        option("policy.push_to_hub", False),
        *args.extra,
    ]
    process_environment = os.environ.copy()
    process_environment.update(
        CUDA_VISIBLE_DEVICES=str(environment["cuda_visible_devices"]),
        HF_HOME=str(environment["hf_home"]),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, env=process_environment, check=True)


if __name__ == "__main__":
    main()
