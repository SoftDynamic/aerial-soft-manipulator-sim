#!/usr/bin/env python3
"""One startup path for native manual controls and the existing PX4 flow."""

import argparse
import queue
import sys
import time
from pathlib import Path

# Direct execution must use this worktree, not another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    scaled_stablesoft_baseline_spec, ScaledStableSoftVisualSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import IdealTendonTensionSpec
from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    initialize_scaled_stablesoft_feedback, update_scaled_stablesoft_feedback,
)
from multi_drone_mujoco.backends.px4_runner import Px4RunnerConfig, Px4MujocoRunner


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
        description="CF2X + StableSoft native actuator simulation"
    )
    parser.add_argument("--mode", choices=("local", "remote"), default="local")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", default=4560, type=int)
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
    parser.add_argument("--duration", type=float, default=300.0)
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
        "--plant", choices=("rigid", "passive_arm", "active_arm"), default="active_arm"
    )
    parser.add_argument("--uav-source", choices=("manual", "px4"), default="manual")
    parser.add_argument("--arm-source", choices=("manual", "controller"), default="manual")
    parser.add_argument("--arm-acceleration", nargs=2, type=float, default=(0.0, 0.0), metavar=("AX", "AY"))
    parser.add_argument("--visual-mode", choices=("mesh", "cylinder"), default="mesh")
    parser.add_argument("--export-xml", type=Path)
    parser.add_argument("--headless", dest="viewer", action="store_false")
    parser.set_defaults(viewer=True, interaction_mode=None)
    return parser


def create_manual_env(args):
    if args.arm_source == "controller" and args.plant != "active_arm":
        raise ValueError("arm controller requires --plant active_arm")
    frequency = args.plant_step_freq or 1200
    if args.arm_source == "controller" and frequency % 240:
        raise ValueError("plant step frequency must be divisible by 240 Hz")
    if args.duration <= 0 or not np.isfinite(args.duration):
        raise ValueError("duration must be finite and positive")
    arm = scaled_stablesoft_baseline_spec() if args.plant != "rigid" else None
    return BaseAviary(
        initial_xyzs=np.asarray([args.initial_position]),
        sim_freq=args.sim_freq, ctrl_freq=frequency, gui=False,
        integrator="implicitfast" if arm else "RK4",
        scaled_stablesoft_arm=arm,
        scaled_stablesoft_visual=ScaledStableSoftVisualSpec(mode=args.visual_mode) if arm else None,
        scaled_stablesoft_tendon_drive=IdealTendonTensionSpec(0.01) if args.plant == "active_arm" else None,
        uav_source="manual", arm_source=args.arm_source,
    )


def run_manual(env, args):
    """Read native input, update feedback at 240 Hz, delegate physics to BaseAviary."""
    events = queue.SimpleQueue()
    viewer = None
    controller = None
    env.reset()
    if args.arm_source == "controller":
        controller, tendon_ids = initialize_scaled_stablesoft_feedback(env)
    stride = env.CTRL_FREQ // 240 if controller else None
    paused = args.viewer
    single_step = False
    index = 0
    steps = int(round(args.duration * env.CTRL_FREQ))
    deadline = time.monotonic()
    try:
        if args.viewer:
            viewer = mujoco.viewer.launch_passive(
                env.model, env.data, key_callback=lambda key: events.put(key),
                show_left_ui=True, show_right_ui=True,
            )
            with viewer.lock():
                viewer.cam.lookat[:] = env.pos[0]
                viewer.cam.distance = 0.6
            print("Native Control: rotor thrust [N], tendon tension [N]. Space pause/resume; . single step; Backspace reset; Q close.")
        while index < steps and (viewer is None or viewer.is_running()):
            if viewer:
                viewer.sync()
                import glfw
                while not events.empty():
                    key = events.get_nowait()
                    if key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key == glfw.KEY_PERIOD and paused:
                        single_step = True
                    elif key == glfw.KEY_Q:
                        return
                    elif key == glfw.KEY_BACKSPACE:
                        with viewer.lock():
                            env.reset()
                            if controller:
                                controller, tendon_ids = initialize_scaled_stablesoft_feedback(env)
                        index = 0
                if paused and not single_step:
                    env.data.xfrc_applied[:] = 0.0
                    time.sleep(1 / 60)
                    continue
            if viewer:
                with viewer.lock():
                    if controller and index % stride == 0:
                        update_scaled_stablesoft_feedback(env, controller, tendon_ids, args.arm_acceleration)
                    env.step()
            else:
                if controller and index % stride == 0:
                    update_scaled_stablesoft_feedback(env, controller, tendon_ids, args.arm_acceleration)
                env.step()
            index += 1
            single_step = False
            if viewer:
                viewer.sync()
                deadline = max(deadline + env.CTRL_TIMESTEP, time.monotonic())
                time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        env.clear_automatic_commands()
        if viewer:
            viewer.close()
            # Preserve the existing native teardown grace before model destruction.
            time.sleep(2.0)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.uav_source == "manual":
        if args.launch_owned_px4 or args.auto_flight:
            raise ValueError("PX4 launch/flight options require --uav-source px4")
        env = create_manual_env(args)
        try:
            if args.export_xml:
                env.save_generated_xml(args.export_xml)
            run_manual(env, args)
        finally:
            env.close()
        return 0
    config = Px4RunnerConfig(
        mode=args.mode,
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
        arm_source=args.arm_source, arm_acceleration_m_s2=tuple(args.arm_acceleration),
        visual_mode=args.visual_mode,
        interaction_mode=args.interaction_mode or (
            "mouse_force" if args.viewer and args.plant != "rigid" else "disabled"
        ),
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
    try:
        if args.export_xml:
            runner.aviary.save_generated_xml(args.export_xml)
        metrics = runner.run()
    finally:
        runner.close()
    print(f"PX4 outcome={metrics.outcome}; steps={metrics.physics_steps}; log={metrics.log_path}")
    return 0 if metrics.outcome == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
