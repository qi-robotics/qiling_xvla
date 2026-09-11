#!/usr/bin/env python3
"""Render container-specific runtime files from the single public YAML."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re
from typing import Any

import yaml


MODEL_ROOT = PurePosixPath("/opt/qiling/models")
DATASET_ROOT = PurePosixPath("/opt/qiling/datasets")
OUTPUT_ROOT = PurePosixPath("/opt/qiling/outputs")
RUNTIME_ROOT = PurePosixPath("/opt/qiling/runtime")


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def clean_relative(value: Any, name: str) -> PurePosixPath:
    text = str(value).strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a non-empty relative path without '..': {text!r}")
    return path


def asset_path(assets: dict[str, Any], name: str, root: PurePosixPath) -> str:
    asset = require_mapping(assets.get(name), f"assets.{name}")
    local_subdir = clean_relative(asset.get("local_subdir"), f"assets.{name}.local_subdir")
    remote_path = clean_relative(asset.get("remote_path"), f"assets.{name}.remote_path")
    return str(root / local_subdir / remote_path)


def resolve_override(value: Any, fallback: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    if not PurePosixPath(text).is_absolute():
        raise ValueError(f"{name} must be an absolute in-container path")
    return text


def write_yaml(path: Path, content: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")


def cyclonedds_xml(interface: str) -> str:
    if interface == "auto":
        network = '<NetworkInterface autodetermine="true" priority="default" multicast="default"/>'
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
            raise ValueError("deployment.dds_interface contains unsupported characters")
        network = f'<NetworkInterface name="{interface}" priority="default" multicast="default"/>'
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<CycloneDDS xmlns=\"https://cdds.io/config\">
  <Domain id=\"any\">
    <General>
      <Interfaces>{network}</Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
"""


def safe_env_value(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        raise ValueError(f"{name} cannot be represented safely in release.env")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = require_mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "root")
    if int(config.get("version", 0)) != 1:
        raise ValueError("only qiling release config version 1 is supported")

    deployment = require_mapping(config.get("deployment"), "deployment")
    assets = require_mapping(config.get("assets"), "assets")
    training = require_mapping(config.get("training"), "training")
    rollout = require_mapping(config.get("rollout"), "rollout")
    bridge = deepcopy(require_mapping(rollout.get("bridge"), "rollout.bridge"))

    mode = str(deployment.get("execution_mode", "shadow")).strip().lower()
    if mode not in {"shadow", "armed"}:
        raise ValueError("deployment.execution_mode must be shadow or armed")
    ros_domain_id = int(deployment.get("ros_domain_id", 0))
    if not 0 <= ros_domain_id <= 232:
        raise ValueError("deployment.ros_domain_id must be between 0 and 232")

    execution = require_mapping(bridge.get("execution"), "rollout.bridge.execution")
    horizon = int(execution["policy_execution_horizon_steps"])
    watermark = int(execution["policy_prefetch_watermark_steps"])
    if horizon <= 0 or not 0 <= watermark < horizon:
        raise ValueError("policy prefetch watermark must satisfy 0 <= watermark < horizon")
    execution["mode"] = mode

    gravity = require_mapping(bridge.get("gravity"), "rollout.bridge.gravity")
    gravity["urdf_path"] = "/opt/qiling/assets/s4_dual_arm.urdf"
    logging = require_mapping(bridge.get("logging"), "rollout.bridge.logging")
    logging["log_directory"] = str(RUNTIME_ROOT / "records")

    args.output.mkdir(parents=True, exist_ok=True)
    write_yaml(
        args.output / "rollout_host.yaml",
        {"qiling_rollout_ros_bridge": {"ros__parameters": bridge}},
    )

    rollout_model = require_mapping(rollout.get("model"), "rollout.model")
    checkpoint_asset = str(rollout_model.get("checkpoint_asset", "rollout_checkpoint"))
    checkpoint = resolve_override(
        rollout_model.get("checkpoint_path"),
        asset_path(assets, checkpoint_asset, MODEL_ROOT),
        "rollout.model.checkpoint_path",
    )
    ipc = require_mapping(bridge.get("ipc"), "rollout.bridge.ipc")
    worker_config = {
        "environment": {
            "python": "/usr/local/bin/python",
            "cuda_visible_devices": str(rollout_model.get("cuda_visible_devices", "0")),
            "hf_home": str(RUNTIME_ROOT / "hf_cache"),
        },
        "model": {
            "checkpoint": checkpoint,
            "task": str(rollout_model.get("task", "")).strip(),
            "warmup_enabled": bool(rollout_model.get("warmup_enabled", True)),
            "warmup_iterations": int(rollout_model.get("warmup_iterations", 1)),
        },
        "ipc": {
            "host": str(ipc["host"]),
            "port": int(ipc["port"]),
            "authkey": str(ipc["authkey"]),
            "retry_delay_sec": float(ipc.get("retry_delay_sec", 1.0)),
            "not_ready_delay_sec": float(ipc.get("not_ready_delay_sec", 0.02)),
        },
    }
    if not worker_config["model"]["task"]:
        raise ValueError("rollout.model.task must be non-empty")
    write_yaml(args.output / "xvla_rollout.yaml", worker_config)

    train_dataset = require_mapping(training.get("dataset"), "training.dataset")
    dataset_subdir = clean_relative(train_dataset.get("local_subdir"), "training.dataset.local_subdir")
    dataset_fallback = str(DATASET_ROOT / dataset_subdir)
    dataset_asset_name = str(train_dataset.get("asset", "")).strip()
    if dataset_asset_name:
        dataset_asset = require_mapping(assets.get(dataset_asset_name), f"assets.{dataset_asset_name}")
        if str(dataset_asset.get("remote_path", "")).strip():
            dataset_fallback = asset_path(assets, dataset_asset_name, DATASET_ROOT)
    dataset_path = resolve_override(
        train_dataset.get("root_path"),
        dataset_fallback,
        "training.dataset.root_path",
    )
    pretrained_name = str(training.get("pretrained_asset", "xvla_base"))
    pretrained_path = resolve_override(
        training.get("pretrained_policy_path"),
        asset_path(assets, pretrained_name, MODEL_ROOT),
        "training.pretrained_policy_path",
    )
    xvla = deepcopy(require_mapping(training.get("xvla"), "training.xvla"))
    xvla["pretrained_policy_path"] = pretrained_path
    run = deepcopy(require_mapping(training.get("run"), "training.run"))
    output_subdir = clean_relative(run.pop("output_subdir"), "training.run.output_subdir")
    run["output_dir"] = str(OUTPUT_ROOT / output_subdir)
    train_config = {
        "environment": {
            "python": "/usr/local/bin/python",
            "cuda_visible_devices": str(rollout_model.get("cuda_visible_devices", "0")),
            "hf_home": str(RUNTIME_ROOT / "hf_cache"),
        },
        "dataset": {
            "repo_id": str(train_dataset["repo_id"]),
            "root": dataset_path,
            "video_backend": str(train_dataset.get("video_backend", "pyav")),
            "rename_map": deepcopy(train_dataset.get("rename_map", {})),
        },
        "xvla": xvla,
        "training": run,
        "logging": deepcopy(require_mapping(training.get("logging"), "training.logging")),
    }
    write_yaml(args.output / "xvla_training.yaml", train_config)

    dds_interface = str(deployment.get("dds_interface", "auto")).strip() or "auto"
    (args.output / "cyclonedds.xml").write_text(cyclonedds_xml(dds_interface), encoding="utf-8")
    rmw_implementation = safe_env_value(
        deployment.get("rmw_implementation", "rmw_cyclonedds_cpp"),
        "deployment.rmw_implementation",
    )
    finish_service = safe_env_value(
        execution.get("finish_service", "/rollout/finish"),
        "rollout.bridge.execution.finish_service",
    )
    abort_service = safe_env_value(
        execution.get("abort_service", "/rollout/abort"),
        "rollout.bridge.execution.abort_service",
    )
    env_lines = [
        f"ROS_DOMAIN_ID={ros_domain_id}",
        f"ROS_LOCALHOST_ONLY={int(deployment.get('ros_localhost_only', 0))}",
        f"RMW_IMPLEMENTATION={rmw_implementation}",
        "CYCLONEDDS_URI=file:///opt/qiling/runtime/cyclonedds.xml",
        f"QILING_EXECUTION_MODE={mode}",
        f"QILING_STATE_WAIT_SEC={max(1, int(deployment.get('startup_state_wait_sec', 30)))}",
        f"QILING_FINISH_SERVICE={finish_service}",
        f"QILING_ABORT_SERVICE={abort_service}",
    ]
    (args.output / "release.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(f"Rendered release configuration in {args.output}")
    print(f"execution_mode={mode}, rollout_checkpoint={checkpoint}")
    print(f"training_pretrained={pretrained_path}, dataset={dataset_path}")


if __name__ == "__main__":
    main()
