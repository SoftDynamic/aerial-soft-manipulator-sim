#!/usr/bin/env python3
"""Run the optional one-drone PX4 SITL/MuJoCo composition."""

import argparse
import json
from pathlib import Path

from multi_drone_mujoco.backends.px4_environment import Px4EnvironmentMode
from multi_drone_mujoco.backends.px4_runner import (
    Px4MujocoRunner,
    Px4RunnerConfig,
)


def _vector3(value: str):
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected x,y,z") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a rigid CF2X MuJoCo plant against PX4 SITL"
    )
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--bind-port", required=True, type=int)
    parser.add_argument("--launch-owned-px4", action="store_true")
    parser.add_argument("--autopilot-dir", type=Path)
    parser.add_argument("--simulator-host")
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument("--px4-instance", type=int, default=0)
    parser.add_argument("--initial-position", type=_vector3, default=(0, 0, 1))
    parser.add_argument("--target-position", type=_vector3, default=(0, 0, 1))
    parser.add_argument("--sim-freq", type=int, default=2400)
    parser.add_argument("--control-freq", type=int, default=200)
    parser.add_argument("--plant-step-freq", type=int)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    parser.add_argument("--actuator-timeout", type=float, default=1.0)
    parser.add_argument("--bootstrap-max-frames", type=int, default=1000)
    parser.add_argument("--bootstrap-wall-interval", type=float, default=0.001)
    parser.add_argument("--bootstrap-final-timeout", type=float, default=1.0)
    parser.add_argument("--motor-time-constant", type=float, default=0.0)
    parser.add_argument("--gcs-bind-host")
    parser.add_argument("--gcs-bind-port", type=int)
    parser.add_argument("--auto-flight", action="store_true")
    parser.add_argument("--preflight-duration", type=float, default=10.0)
    parser.add_argument("--offboard-warmup", type=float, default=1.0)
    parser.add_argument("--armed-ramp-timeout", type=float, default=8.0)
    parser.add_argument("--release-hover-rpm-fraction", type=float, default=1.1)
    parser.add_argument("--landing-duration", type=float, default=8.0)
    parser.add_argument("--landing-descent", type=float, default=3.0)
    parser.add_argument("--landing-height", type=float, default=0.06)
    parser.add_argument("--spool-target-height-offset", type=float, default=1.0)
    parser.add_argument("--release-transition", type=float, default=1.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--interaction-mode",
        choices=(
            "disabled", "replay", "mouse_force", "keyboard_force",
            "tendon_modal", "joint_torque",
        ),
        default="disabled",
    )
    parser.add_argument("--interaction-script", type=Path)
    parser.add_argument("--interaction-target-body")
    parser.add_argument("--interaction-target-site")
    parser.add_argument(
        "--interaction-target-policy",
        choices=("arm_segments", "single"),
        default="arm_segments",
    )
    parser.add_argument("--force-step-mn", type=float, default=0.5)
    parser.add_argument("--max-force-mn", type=float, default=1.0)
    parser.add_argument("--force-spring-n-per-m", type=float, default=0.25)
    parser.add_argument("--force-damping-n-s-per-m", type=float, default=0.01)
    parser.add_argument("--force-release", choices=("immediate",), default="immediate")
    parser.add_argument("--allow-unverified-force", action="store_true")
    parser.add_argument("--joint-torque-step-unm", type=float, default=10.0)
    parser.add_argument("--joint-torque-limit-unm", type=float, default=50.0)
    parser.add_argument("--allow-unverified-joint-torque", action="store_true")
    parser.add_argument("--tendon-acceleration-step", type=float, default=0.005)
    parser.add_argument("--tendon-acceleration-limit", type=float, default=0.04)
    parser.add_argument(
        "--no-interaction-overlay",
        dest="interaction_overlay",
        action="store_false",
        default=True,
    )
    parser.add_argument("--log-dir", type=Path, default=Path("results/px4"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plant", choices=("rigid", "passive_arm", "active_arm"), default="rigid"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode(args.mode),
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        launch_owned_px4=args.launch_owned_px4,
        autopilot_dir=args.autopilot_dir,
        simulator_host=args.simulator_host,
        parameter_file=args.parameter_file,
        px4_instance=args.px4_instance,
        initial_position_enu_m=args.initial_position,
        target_position_enu_m=args.target_position,
        sim_freq_hz=args.sim_freq,
        control_freq_hz=args.control_freq,
        plant_step_freq_hz=args.plant_step_freq,
        duration_s=args.duration,
        handshake_timeout_s=args.handshake_timeout,
        actuator_timeout_s=args.actuator_timeout,
        bootstrap_max_frames=args.bootstrap_max_frames,
        bootstrap_wall_interval_s=args.bootstrap_wall_interval,
        bootstrap_final_timeout_s=args.bootstrap_final_timeout,
        motor_time_constant_s=args.motor_time_constant,
        gui=args.viewer,
        log_dir=args.log_dir,
        random_seed=args.seed,
        gcs_bind_host=args.gcs_bind_host,
        gcs_bind_port=args.gcs_bind_port,
        auto_flight=args.auto_flight,
        preflight_duration_s=args.preflight_duration,
        offboard_warmup_s=args.offboard_warmup,
        armed_ramp_timeout_s=args.armed_ramp_timeout,
        release_hover_rpm_fraction=args.release_hover_rpm_fraction,
        landing_duration_s=args.landing_duration,
        landing_descent_s=args.landing_descent,
        landing_height_enu_m=args.landing_height,
        spool_target_height_offset_m=args.spool_target_height_offset,
        release_transition_s=args.release_transition,
        plant=args.plant,
        interaction_mode=args.interaction_mode,
        interaction_script=args.interaction_script,
        interaction_target_body=args.interaction_target_body,
        interaction_target_site=args.interaction_target_site,
        interaction_target_policy=args.interaction_target_policy,
        interaction_force_step_n=args.force_step_mn / 1000.0,
        interaction_max_force_n=args.max_force_mn / 1000.0,
        interaction_spring_n_per_m=args.force_spring_n_per_m,
        interaction_damping_n_s_per_m=args.force_damping_n_s_per_m,
        interaction_force_release=args.force_release,
        tendon_acceleration_step_m_s2=args.tendon_acceleration_step,
        tendon_acceleration_limit_m_s2=args.tendon_acceleration_limit,
        interaction_overlay=args.interaction_overlay,
        allow_unverified_force=args.allow_unverified_force,
        joint_torque_step_nm=args.joint_torque_step_unm * 1e-6,
        joint_torque_limit_nm=args.joint_torque_limit_unm * 1e-6,
        allow_unverified_joint_torque=args.allow_unverified_joint_torque,
    )
    runner = Px4MujocoRunner.create_default(config)
    metrics = runner.run()
    print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))
    return 0 if metrics.outcome == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
