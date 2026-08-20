#!/usr/bin/env python3
"""Convert slender-pin expert recordings to LeRobot v3 for XVLA training."""

from convert_fixed_socket_rj45_to_lerobot_v3_common import (
    ConversionSpec,
    EEF_ACTION_NAMES,
    EEF_STATE_NAMES,
    run_cli,
)


SPEC = ConversionSpec(
    mode="xvla_eef_action",
    default_output_dir="datasets/slender_pin_lerobot_v3_xvla_v1",
    default_repo_id="qiling/slender_pin_xvla_v1",
    default_report="reports/convert_slender_pin_lerobot_v3_xvla_v1.json",
    default_recorded_root="datasets/recorded_slender_pin_v1",
    robot_type="S4_RIGHT_ARM_O6_SLENDER_PIN_XVLA",
    state_key="observation_state",
    action_key="action",
    state_width=17,
    action_width=10,
    state_names=EEF_STATE_NAMES,
    action_names=EEF_ACTION_NAMES,
)


if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
