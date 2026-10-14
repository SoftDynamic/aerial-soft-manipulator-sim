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
    parser = argparse.ArgumentParser(description="CF2X + StableSoft native actuator simulation")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/sim.yaml")
    parser.add_argument("--uav-source", choices=("manual", "px4"), default=None)
    parser.add_argument("--arm-source", choices=("manual", "controller"), default=None)
    parser.add_argument("--initial-position", type=_vector3, default=None)
    parser.add_argument("--target-position", type=_vector3, default=None)
    parser.add_argument("--arm-acceleration", nargs=2, type=float, default=None, metavar=("AX", "AY"))
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--export-xml", type=Path, default=None)
    parser.add_argument("--visual-mode", choices=("mesh", "cylinder"), default=None)
    return parser


def resolve_config(cli):
    """Resolve once: verified defaults < explicit YAML < explicit CLI."""
    import yaml
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    from multi_drone_mujoco.models.parameters import ResolvedUavParameters
    from multi_drone_mujoco.models.components.tendon_drive import BASELINE_MAX_TENSION_N

    uav_names = {"mass_kg": "mass", "ixx_kg_m2": "ixx", "iyy_kg_m2": "iyy", "izz_kg_m2": "izz",
                 "arm_length_m": "arm_length", "kf_n_per_rpm2": "kf", "km_nm_per_rpm2": "km",
                 "thrust2weight_ratio": "thrust2weight_ratio", "prop_radius_m": "prop_radius",
                 "collision_height_m": "collision_h", "collision_radius_m": "collision_r",
                 "collision_z_offset_m": "collision_z_offset"}
    arm_names = {"segment_count": "num_segments", "total_length_m": "total_length", "total_mass_kg": "total_mass",
                 "body_radius_m": "body_radius", "joint_stiffness_nm_per_rad": "joint_stiffness",
                 "joint_damping_nm_s_per_rad": "joint_damping", "target_damping_ratio": "target_damping_ratio",
                 "tendon_frictionloss_n": "tendon_frictionloss"}
    baseline_arm = scaled_stablesoft_baseline_spec()
    physical = ResolvedUavParameters.resolve()
    values = {
        "robot": {"uav": {key: physical[value] for key, value in uav_names.items()},
                  "soft_arm": {**{key: asdict(baseline_arm)[value] for key, value in arm_names.items()},
                               "max_tension_n": BASELINE_MAX_TENSION_N, "visual_mode": "mesh"}},
        "simulation": {"uav_source": "manual", "arm_source": "manual", "initial_position_enu_m": [0, 0, 1],
                       "target_position_enu_m": [0, 0, 1], "arm_acceleration_m_s2": [0, 0], "gui": True,
                       "duration_s": None, "headless_duration_s": 5.0, "sim_freq_hz": 2400,
                       "control_freq_hz": 200, "plant_step_freq_hz": 1200, "export_xml": None,
                       "log_dir": str(Path.cwd() / "results/sim")},
        "px4": {"mode": "local", "bind_host": "127.0.0.1", "bind_port": 4560,
                "launch_owned_px4": True, "auto_flight": True, "autopilot_dir": "/home/ustc/PX4-Autopilot",
                "simulator_host": "127.0.0.1", "parameter_file": str(Path(__file__).resolve().parents[1] / "configs/px4/cf2x_scaled_stablesoft_passive.params"),
                "gcs_bind_host": "127.0.0.1", "gcs_bind_port": 14550},
        "diagnostics": {"trial": None, "sample_hz": 20, "max_samples": 256,
                        "baseline_s": 2.0, "stable_window_s": 1.0, "position_threshold_m": 0.10,
                        "speed_threshold_m_s": 0.10, "pulse_tendon_index": 0, "pulse_tension_n": 0.001,
                        "pulse_duration_s": 0.30, "recovery_s": 5.0}}
    origins = {}
    def mark(data, prefix="", origin="default"):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict): mark(value, name, origin)
            else: origins[name] = origin
    mark(values)
    path = cli.config.resolve()
    explicit = yaml.safe_load(path.read_text()) or {}
    if not isinstance(explicit, dict): raise ValueError("simulation YAML must be a mapping")
    # Only backend dataclass settings are accepted beyond the displayed PX4 defaults.
    from dataclasses import fields
    backend_names = {item.name for item in fields(Px4RunnerConfig)} - {
        "uav_parameters", "arm_spec", "max_tension_n", "effective_config", "diagnostics",
        "plant", "arm_source", "arm_acceleration_m_s2", "visual_mode", "duration_s", "gui", "log_dir",
        "sim_freq_hz", "control_freq_hz", "plant_step_freq_hz", "initial_position_enu_m", "target_position_enu_m"}
    def merge(dst, src, prefix=""):
        if not isinstance(src, dict): raise ValueError(f"{prefix} must be a mapping")
        for key, value in src.items():
            name = f"{prefix}.{key}" if prefix else key
            if key not in dst and not (prefix == "px4" and key in backend_names):
                raise ValueError(f"unknown configuration value: {name}")
            if isinstance(dst.get(key), dict): merge(dst[key], value, name)
            else: dst[key] = value; origins[name] = "yaml"
    merge(values, explicit)
    for group, key in (("simulation", "log_dir"), ("simulation", "export_xml"),
                       ("px4", "autopilot_dir"), ("px4", "parameter_file"), ("px4", "interaction_script")):
        value = values[group].get(key)
        if value is not None:
            item = Path(value).expanduser()
            values[group][key] = str((path.parent / item).resolve())
    cli_map = {"uav_source": ("simulation", "uav_source"), "arm_source": ("simulation", "arm_source"),
               "initial_position": ("simulation", "initial_position_enu_m"), "target_position": ("simulation", "target_position_enu_m"),
               "arm_acceleration": ("simulation", "arm_acceleration_m_s2"), "duration": ("simulation", "duration_s"),
               "export_xml": ("simulation", "export_xml"), "visual_mode": ("robot", "soft_arm", "visual_mode")}
    for name, keys in cli_map.items():
        value = getattr(cli, name)
        if value is not None:
            dst = values
            for key in keys[:-1]: dst = dst[key]
            dst[keys[-1]] = str(value.resolve()) if isinstance(value, Path) else value
            origins[".".join(keys)] = "cli"
    sim = values["simulation"]
    if cli.headless is not None: sim["gui"] = False; origins["simulation.gui"] = "cli"
    if sim["duration_s"] is None and not sim["gui"]:
        sim["duration_s"] = sim["headless_duration_s"]
        origins["simulation.duration_s"] = origins["simulation.headless_duration_s"]
    if sim["duration_s"] is not None and (not np.isfinite(sim["duration_s"]) or sim["duration_s"] <= 0):
        raise ValueError("duration must be finite and positive; GUI may use null")
    if sim["uav_source"] not in ("manual", "px4") or sim["arm_source"] not in ("manual", "controller"):
        raise ValueError("invalid control source")
    for name, size in (("initial_position_enu_m", 3), ("target_position_enu_m", 3), ("arm_acceleration_m_s2", 2)):
        if np.asarray(sim[name]).shape != (size,) or not np.all(np.isfinite(sim[name])):
            raise ValueError(f"{name} must contain {size} finite values")
    physical = ResolvedUavParameters.resolve(overrides={uav_names[k]: v for k, v in values["robot"]["uav"].items()})
    arm = values["robot"]["soft_arm"]
    spec = replace(baseline_arm, **{arm_names[k]: v for k, v in arm.items() if k in arm_names})
    drive = IdealTendonTensionSpec(arm["max_tension_n"])
    ScaledStableSoftVisualSpec(mode=arm["visual_mode"])
    if spec.num_segments != 9: raise ValueError("daily StableSoft model supports nine segments only")
    for name in ("sim_freq_hz", "control_freq_hz", "plant_step_freq_hz"):
        if type(sim[name]) is not int or sim[name] <= 0: raise ValueError(f"{name} must be a positive integer")
    if sim["sim_freq_hz"] % sim["plant_step_freq_hz"] or sim["plant_step_freq_hz"] % 240:
        raise ValueError("plant frequency must divide simulation frequency and be divisible by 240 Hz")
    if arm["max_tension_n"] != BASELINE_MAX_TENSION_N or spec != baseline_arm:
        print("Configured arm differs from verified controller baseline; accuracy/stability has not been established.")
    px4 = values["px4"]
    if px4["auto_flight"] and (px4["mode"] != "local" or not px4["launch_owned_px4"]):
        raise ValueError("automatic arming is restricted to explicitly configured local owned SITL")
    return SimpleNamespace(**{k: v for k, v in sim.items() if k != "export_xml"}, initial_position=sim["initial_position_enu_m"], target_position=sim["target_position_enu_m"],
                           arm_acceleration=sim["arm_acceleration_m_s2"], viewer=sim["gui"], duration=sim["duration_s"],
                           visual_mode=arm["visual_mode"], uav_parameters=physical, arm_spec=spec, tendon_drive=drive,
                           effective_config={"values": values, "origins": origins, "source_file": str(path)},
                           px4=px4, diagnostics=values["diagnostics"], export_xml=Path(sim["export_xml"]) if sim["export_xml"] else None)


def create_manual_env(args):
    if not hasattr(args, "effective_config"):
        args = resolve_config(args)
    return BaseAviary(
        initial_xyzs=np.asarray([args.initial_position]),
        sim_freq=args.sim_freq_hz, ctrl_freq=args.plant_step_freq_hz, gui=False,
        integrator="implicitfast", uav_parameters=args.uav_parameters,
        scaled_stablesoft_arm=args.arm_spec,
        scaled_stablesoft_visual=ScaledStableSoftVisualSpec(mode=args.visual_mode),
        scaled_stablesoft_tendon_drive=args.tendon_drive,
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
    steps = None if args.duration is None else int(round(args.duration * env.CTRL_FREQ))
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
        while (steps is None or index < steps) and (viewer is None or viewer.is_running()):
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
    import json
    from datetime import datetime, timezone
    args = resolve_config(build_parser().parse_args(argv))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = log_dir / ("effective-config-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    snapshot_path.write_text(json.dumps(args.effective_config, indent=2) + "\n")
    print(f"Effective configuration: {snapshot_path}")
    if args.uav_source == "manual":
        if args.diagnostics["trial"] is not None:
            raise ValueError("tendon diagnostic requires actual owned PX4 SITL")
        env = create_manual_env(args)
        try:
            if args.export_xml: env.save_generated_xml(args.export_xml)
            run_manual(env, args)
        finally: env.close()
        return 0
    config = Px4RunnerConfig(
        **args.px4, initial_position_enu_m=tuple(args.initial_position),
        target_position_enu_m=tuple(args.target_position), sim_freq_hz=args.sim_freq_hz,
        control_freq_hz=args.control_freq_hz, plant_step_freq_hz=args.plant_step_freq_hz,
        duration_s=args.duration, gui=args.viewer, log_dir=log_dir, plant="active_arm",
        arm_source=args.arm_source, arm_acceleration_m_s2=tuple(args.arm_acceleration),
        visual_mode=args.visual_mode, uav_parameters=args.uav_parameters, arm_spec=args.arm_spec,
        max_tension_n=args.tendon_drive.max_tension, diagnostics=args.diagnostics,
        effective_config=args.effective_config,
    )
    runner = Px4MujocoRunner.create_default(config)
    try:
        if args.export_xml: runner.aviary.save_generated_xml(args.export_xml)
        metrics = runner.run()
    finally: runner.close()
    print(f"PX4 outcome={metrics.outcome}; steps={metrics.physics_steps}; log={metrics.log_path}")
    return 0 if metrics.outcome == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
