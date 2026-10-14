"""Focused native actuator, source ownership and physics-boundary regressions."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from scripts import run_sim
from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.transport import FakeTransport
from multi_drone_mujoco.control.bounded_tendon_allocator import AllocationNumericalError
from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    initialize_scaled_stablesoft_feedback, update_scaled_stablesoft_feedback,
)
from multi_drone_mujoco.envs.base_aviary import BaseAviary
import multi_drone_mujoco.envs.base_aviary as env_module
from multi_drone_mujoco.utils.enums import DroneModel


def _env(uav="manual", arm="manual"):
    args = run_sim.build_parser().parse_args(["--headless", "--arm-source", arm])
    env = run_sim.create_manual_env(args)
    env.uav_source = uav
    return env


@pytest.mark.parametrize("rpm", [
    [12700., 13900., 15100., 16300.], [15000., 0., 0., 0.],
    [0., 16000., 0., 0.], [13000., 11000., 12000., 17000.],
])
def test_actual_actuator_wrench_matches_commit1_nonzero_rpm(rpm):
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output([
        "git", "show", "9fd5bbe47f7e9ecc2d1adc6055e1ec110d4d7957:multi_drone_mujoco/envs/base_aviary.py"
    ], cwd=root, text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "BaseAviary")
    old = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_physics")
    namespace = {"np": np, "mujoco": mujoco, "DroneModel": DroneModel}
    exec(compile(ast.Module(body=[old], type_ignores=[]), "commit1_propulsion", "exec"), namespace)
    env = _env("px4")
    try:
        quaternion = np.empty(4)
        mujoco.mju_euler2Quat(quaternion, np.array([0.2, -0.3, 0.4]), "xyz")
        env.data.qpos[3:7] = quaternion
        mujoco.mj_forward(env.model, env.data)
        body = env.model.body("drone0").id
        namespace["_physics"](env, np.asarray(rpm), 0)
        old_wrench = env.data.xfrc_applied[body].copy()
        expected = np.zeros(env.model.nv)
        mujoco.mj_applyFT(env.model, env.data, old_wrench[:3], old_wrench[3:], env.data.xipos[body], body, expected)
        env.data.xfrc_applied[:] = 0
        env.set_rotor_rpm_command(rpm)
        env.apply_automatic_commands()
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.qfrc_actuator, expected, rtol=1e-12, atol=1e-14)
        assert np.count_nonzero(env.data.xfrc_applied) == 0
        assert np.linalg.norm(env.data.qfrc_actuator[:6]) > 0
    finally:
        env.close()


@pytest.mark.parametrize("uav,arm", [
    ("manual", "manual"), ("manual", "controller"), ("px4", "manual"), ("px4", "controller"),
])
def test_named_groups_hold_only_automatic_cache_before_every_physics_substep(monkeypatch, uav, arm):
    original_build = env_module.build_aviary_mjcf

    def interleaved_build(*args, **kwargs):
        build = original_build(*args, **kwargs)
        tree = ET.fromstring(build.xml)
        actuator = tree.find("actuator")
        elements = list(actuator)
        actuator[:] = [elements[i] for i in (4, 0, 5, 1, 6, 2, 7, 3)]
        return replace(build, xml=ET.tostring(tree, encoding="unicode"))

    monkeypatch.setattr(env_module, "build_aviary_mjcf", interleaved_build)
    env = _env(uav, arm)
    rotor = env.ROTOR_CONTROL_ADDRESSES[0]
    tendon = np.asarray([b.control_address for b in env.SCALED_STABLESOFT_TENDON_BINDINGS])
    assert env.model.nu == 8
    assert rotor.tolist() == [1, 3, 5, 7]
    assert tendon.tolist() == [0, 2, 4, 6]
    manual_rotor = np.array([0.01, 0.02, 0.03, 0.04])
    manual_tendon = np.array([0.001, 0.002, 0.003, 0.004])
    if uav == "px4":
        env.set_rotor_rpm_command([12700, 13900, 15100, 16300])
    else:
        env.data.ctrl[rotor] = manual_rotor
    if arm == "controller":
        env.set_scaled_stablesoft_tension_command([0.004, 0.003, 0.002, 0.001])
    else:
        env.data.ctrl[tendon] = manual_tendon
    original_step = mujoco.mj_step
    observed = []

    def dirty_gui_values(model, data):
        # Compare immediately at the real MuJoCo boundary. A simulated later UI
        # sync changes automatic fields, including between controller updates.
        expected_r = env._rotor_thrust_command_n[0] if uav == "px4" else manual_rotor
        expected_t = env._automatic_tendon_command_n if arm == "controller" else manual_tendon
        np.testing.assert_allclose(data.ctrl[rotor], expected_r, rtol=0, atol=0)
        np.testing.assert_allclose(data.ctrl[tendon], expected_t, rtol=0, atol=0)
        observed.append(data.ctrl.copy())
        original_step(model, data)
        if uav == "px4":
            data.ctrl[rotor] = 0.09
        if arm == "controller":
            data.ctrl[tendon] = 0.009

    monkeypatch.setattr(mujoco, "mj_step", dirty_gui_values)
    try:
        for _ in range(3):
            env.step()
        assert len(observed) == 3 * env.SIM_STEPS_PER_CTRL
        if uav == "manual":
            assert env._rotor_thrust_command_n is None
        if arm == "manual":
            assert env._automatic_tendon_command_n is None
        env.clear_automatic_commands()
        np.testing.assert_allclose(env.data.ctrl[rotor], 0 if uav == "px4" else manual_rotor)
        np.testing.assert_allclose(env.data.ctrl[tendon], 0 if arm == "controller" else manual_tendon)
        if uav == "px4" and arm == "controller":
            env.set_scaled_stablesoft_tension_command([0.001] * 4)
            env.set_rotor_rpm_command([15000] * 4, armed=False)
            env.apply_automatic_commands()
            np.testing.assert_array_equal(env.data.ctrl[rotor], 0)
            np.testing.assert_array_equal(env.data.ctrl[tendon], 0)
    finally:
        env.close()


@pytest.mark.parametrize("uav,arm", [("manual", "manual"), ("manual", "controller"), ("px4", "manual"), ("px4", "controller")])
def test_four_cli_paths_compute_at_intended_frequency_and_keep_manual_values(monkeypatch, tmp_path, uav, arm):
    counts = {"feedback": 0, "physics": 0}
    original_update = run_sim.update_scaled_stablesoft_feedback

    def feedback(*args):
        counts["feedback"] += 1
        return original_update(*args)

    monkeypatch.setattr(run_sim, "update_scaled_stablesoft_feedback", feedback)
    import multi_drone_mujoco.control.scaled_stablesoft_bending_control as control_module
    monkeypatch.setattr(control_module, "update_scaled_stablesoft_feedback", feedback)
    original_manual = run_sim.create_manual_env
    original_px4 = run_sim.Px4MujocoRunner.create_default
    captured = []

    def instrument(env):
        captured.append(env)
        step = env.step
        rotor = env.ROTOR_CONTROL_ADDRESSES[0]
        tendon = [b.control_address for b in env.SCALED_STABLESOFT_TENDON_BINDINGS]

        def slider_sync_then_step(action=None):
            if uav == "manual":
                env.data.ctrl[rotor] = [0.01, 0.02, 0.03, 0.04]
            if arm == "manual":
                env.data.ctrl[tendon] = [0.001, 0.002, 0.003, 0.004]
            result = step(action)
            counts["physics"] += 1
            if uav == "manual":
                np.testing.assert_array_equal(env.data.ctrl[rotor], [0.01, 0.02, 0.03, 0.04])
            if arm == "manual":
                np.testing.assert_array_equal(env.data.ctrl[tendon], [0.001, 0.002, 0.003, 0.004])
            return result

        env.step = slider_sync_then_step
        return env

    monkeypatch.setattr(run_sim, "create_manual_env", lambda args: instrument(original_manual(args)))

    def fake_px4(config):
        runner = original_px4(config)
        instrument(runner.aviary)
        packets = [NormalizedMotorCommand((2 / 3,) * 4, True, 1_000_000 + i * 5000, i + 1) for i in range(6)]
        transport = FakeTransport(packets)
        runner.backend.transport = transport
        runner.backend.coordinator.transport = transport
        return runner

    monkeypatch.setattr(run_sim.Px4MujocoRunner, "create_default", fake_px4)
    import yaml
    cfg = tmp_path / "sim.yaml"
    cfg.write_text(yaml.safe_dump({"simulation": {"log_dir": str(tmp_path)}, "px4": {
        "mode": "remote", "launch_owned_px4": False, "auto_flight": False,
        "gcs_bind_host": None, "gcs_bind_port": None}}))
    assert run_sim.main([
        "--config", str(cfg), "--uav-source", uav, "--arm-source", arm, "--headless", "--duration", "0.025",
        "--export-xml", str(tmp_path / "model.xml"),
    ]) == 0
    assert counts == {"feedback": 6 if arm == "controller" else 0, "physics": 30}
    env = captured[0]
    assert env.data.time == pytest.approx(0.025, abs=1e-14)
    assert mujoco.MjModel.from_xml_path(str(tmp_path / "model.xml")).nu == 8


@pytest.mark.parametrize("body_name", ["drone0", "drone0_scaled_stablesoft_arm_segment_08"])
@pytest.mark.parametrize("with_wind", [False, True])
def test_native_wrench_persists_once_per_substep_and_release_clears(monkeypatch, body_name, with_wind):
    env = _env()
    rotor = env.ROTOR_CONTROL_ADDRESSES[0]
    env.data.ctrl[rotor] = 0.03
    body = env.model.body(body_name).id
    wrench = np.array([0.001, -0.002, 0.0005, 1e-6, -2e-6, 3e-6])
    env.data.xfrc_applied[body] = wrench
    env._updateAndStoreKinematicInformation()
    np.testing.assert_array_equal(env.data.xfrc_applied[body], wrench)
    root = env.model.body("drone0").id
    wind = np.array([0.0003, -0.0002, 0.0001]) if with_wind else np.zeros(3)
    if with_wind:
        env._wind_field = SimpleNamespace(get_force=lambda **kw: wind)
    expected = np.zeros_like(env.data.xfrc_applied)
    expected[body] = wrench
    expected[root, :3] += wind
    real_step = mujoco.mj_step
    snapshots = []

    def observe(model, data):
        snapshots.append(data.xfrc_applied.copy())
        real_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", observe)
    try:
        env.step()
        assert len(snapshots) == env.SIM_STEPS_PER_CTRL
        for values in snapshots:
            np.testing.assert_array_equal(values, expected)
        assert not np.any(env.data.xfrc_applied)
        snapshots.clear()
        env.step()
        expected[:] = 0
        expected[root, :3] = wind
        for values in snapshots:
            np.testing.assert_array_equal(values, expected)
    finally:
        env.close()


def test_feedback_reset_and_numerical_failure_zero_cache(monkeypatch):
    env = _env("manual", "controller")
    try:
        controller, ids = initialize_scaled_stablesoft_feedback(env)
        update_scaled_stablesoft_feedback(env, controller, ids, (0.005, -0.005))
        env.reset()
        assert env._automatic_tendon_command_n is None
        controller, ids = initialize_scaled_stablesoft_feedback(env)
        assert controller.state.control_tick == 0

        def fail(*args):
            raise AllocationNumericalError("injected numerical failure")

        monkeypatch.setattr(controller.allocator, "allocate", fail)
        with pytest.raises(RuntimeError, match="controller fault"):
            update_scaled_stablesoft_feedback(env, controller, ids, (0.005, -0.005))
        assert controller.fault_latched
        assert env._automatic_tendon_command_n == (0.0,) * 4
    finally:
        env.close()


@pytest.mark.parametrize("arm", ["manual", "controller"])
@pytest.mark.parametrize("failure", ["disarm", "timeout", "disconnect"])
def test_px4_disarm_and_failure_zero_before_write_without_touching_manual_arm(tmp_path, arm, failure):
    from multi_drone_mujoco.backends.px4_runner import Px4RunnerConfig, Px4MujocoRunner
    config = Px4RunnerConfig(
        mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="active_arm", arm_source=arm, duration_s=0.01,
        bootstrap_wall_interval_s=0, log_dir=tmp_path,
        arm_acceleration_m_s2=(0.005, -0.005),
    )
    runner = Px4MujocoRunner.create_default(config)
    packets = [
        NormalizedMotorCommand((0.,) * 4, False, 1_000_000, 1),
        NormalizedMotorCommand((2 / 3,) * 4, True, 1_005_000, 2),
    ]
    if failure == "disarm":
        # Nonzero payload must still be ignored when the armed bit is false.
        packets.append(NormalizedMotorCommand((2 / 3,) * 4, False, 1_010_000, 3))
    transport = FakeTransport(packets)
    runner.backend.transport = transport
    runner.backend.coordinator.transport = transport
    env = runner.aviary
    rotor = env.ROTOR_CONTROL_ADDRESSES[0]
    tendon = [b.control_address for b in env.SCALED_STABLESOFT_TENDON_BINDINGS]
    manual = [0.001, 0.002, 0.003, 0.004]
    real_step = env.step
    states = []

    def observe(action=None):
        if arm == "manual":
            env.data.ctrl[tendon] = manual
        real_step(action)
        states.append(env.data.ctrl.copy())
        # Simulate slider writes between all controller/physics updates.
        env.data.ctrl[rotor] = 0.1
        if arm == "controller":
            env.data.ctrl[tendon] = 0.009
        if len(states) == 6 and failure == "disconnect":
            transport.disconnect()

    env.step = observe
    metrics = runner.run()
    assert metrics.outcome == {"disarm": "completed", "timeout": "timed_out", "disconnect": "disconnected"}[failure]
    assert len(states) == (12 if failure == "disarm" else 6)
    assert np.all(states[0][rotor] > 0)
    if failure == "disarm":
        for state in states[6:]:
            np.testing.assert_array_equal(state[rotor], 0)
            np.testing.assert_array_equal(state[tendon], 0 if arm == "controller" else manual)
    assert not env._rotors_armed
    np.testing.assert_array_equal(env.data.ctrl[rotor], 0)
    np.testing.assert_array_equal(env.data.ctrl[tendon], 0 if arm == "controller" else manual)
    assert (runner._tendon_controller is None) == (arm == "manual")


@pytest.mark.parametrize("target,force,torque", [
    ("drone0", (0.02, 0., 0.), (1e-6, 2e-6, 3e-6)),
    ("drone0", (0., 0., 0.), (1e-6, 2e-6, 3e-6)),
    ("drone0_scaled_stablesoft_arm_segment_08", (0., 0., 0.), (1e-6, 2e-6, 3e-6)),
])
def test_px4_native_mouse_wrench_keeps_selected_body_and_torque_then_releases(monkeypatch, tmp_path, target, force, torque):
    from multi_drone_mujoco.backends.px4_runner import Px4RunnerConfig, Px4MujocoRunner
    from multi_drone_mujoco.backends.px4_interactive_viewer import Px4InteractionViewer
    env = _env("px4")
    body = env.model.body(target).id
    wrench = np.array([*force, *torque])

    class Viewer:
        perturb = SimpleNamespace(select=body)
        def is_running(self): return True
        def sync(self): env.data.xfrc_applied[body] = wrench
        def close(self): pass

    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **kw: Viewer())
    config = Px4RunnerConfig(
        mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="active_arm", gui=True, interaction_mode="mouse_force",
        interaction_overlay=False, log_dir=tmp_path,
    )
    runner = Px4MujocoRunner.create_default(config)
    runner.aviary.close()
    runner.aviary = env
    runner._setup_interaction()
    adapter = Px4InteractionViewer(env, runner._force_applier, config)
    try:
        command = adapter.poll(0.)
        assert command.target_body_id == body
        assert command.source == "native_mouse_wrench" and command.active
        assert runner._prepare_interaction_command(command) is command
        expected = wrench * (config.interaction_max_force_n / np.linalg.norm(force)) if any(force) else wrench.copy()
        np.testing.assert_allclose(env.data.xfrc_applied[body], expected)
        np.testing.assert_allclose(command.generated_torque_nm, expected[3:])
        env.set_rotor_rpm_command([15000] * 4)
        env.step()
        assert not np.any(env.data.xfrc_applied)
        wrench[:] = 0
        released = adapter.poll(0.001)
        assert not released.active and released.released
        env.step()
        assert not np.any(env.data.xfrc_applied)
    finally:
        adapter.close()
        runner.close()


def test_config_precedence_paths_and_effective_log(tmp_path):
    import json
    import yaml
    config = tmp_path / "nested" / "sim.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"simulation": {"duration_s": 0.02,
        "initial_position_enu_m": [1, 2, 3], "log_dir": "logs", "export_xml": "model.xml"},
        "robot": {"soft_arm": {"max_tension_n": 0.02}}}))
    args = run_sim.resolve_config(run_sim.build_parser().parse_args(["--config", str(config), "--headless"]))
    assert args.initial_position == [1, 2, 3]
    assert args.duration == 0.02
    assert args.log_dir == str(config.parent / "logs")
    assert args.export_xml == config.parent / "model.xml"
    origins = args.effective_config["origins"]
    assert origins["simulation.duration_s"] == "yaml"
    assert origins["robot.uav.mass_kg"] == "default"
    assert origins["simulation.gui"] == "cli"
    assert run_sim.main(["--config", str(config), "--headless", "--duration", "0.01", "--initial-position", "0,0,1"]) == 0
    saved = json.loads(next((config.parent / "logs").glob("effective-*.json")).read_text())
    assert saved["values"]["simulation"]["duration_s"] == 0.01
    assert saved["origins"]["simulation.duration_s"] == "cli"


def test_resolved_physical_snapshot_reaches_compiled_model_and_controller(tmp_path):
    import yaml
    from multi_drone_mujoco.models.parameters import DRONE_PARAMS
    original = dict(DRONE_PARAMS[DroneModel.CF2X])
    config = tmp_path / "sim.yaml"
    config.write_text(yaml.safe_dump({"robot": {"uav": {"mass_kg": 0.028, "ixx_kg_m2": 1.5e-5,
        "arm_length_m": 0.04, "thrust2weight_ratio": 2.3, "collision_radius_m": 0.055},
        "soft_arm": {"total_length_m": 0.09, "total_mass_kg": 0.004, "body_radius_m": 0.004,
                     "max_tension_n": 0.02, "joint_stiffness_nm_per_rad": 0.0035}}}))
    args = run_sim.resolve_config(run_sim.build_parser().parse_args(["--config", str(config), "--headless", "--arm-source", "controller"]))
    env = run_sim.create_manual_env(args)
    try:
        assert env.UAV_PARAMETERS is args.uav_parameters
        assert env.model.body("drone0").mass[0] == 0.028
        assert env.model.body("drone0").inertia[0] == 1.5e-5
        assert env.M == pytest.approx(0.032)
        assert sum(env.model.body_mass) == pytest.approx(env.M)
        assert env.MAX_THRUST == pytest.approx(2.3 * 0.028 * 9.81)
        np.testing.assert_allclose(env.model.actuator_ctrlrange[env.ROTOR_CONTROL_ADDRESSES[0], 1], env.MAX_THRUST / 4)
        assert env.SCALED_STABLESOFT_ARM_SPEC.total_length == 0.09
        assert env.SCALED_STABLESOFT_ARM_SPEC.joint_stiffness == 0.0035
        controller, ids = initialize_scaled_stablesoft_feedback(env)
        assert controller.config.allocator.max_tension_n == 0.02
        for binding in env.SCALED_STABLESOFT_TENDON_BINDINGS:
            assert env.model.actuator_ctrlrange[binding.actuator_id, 1] == 0.02
        update_scaled_stablesoft_feedback(env, controller, ids, (0.005, 0.0))
        assert max(env._automatic_tendon_command_n) <= 0.02
        env.step()
    finally: env.close()
    assert DRONE_PARAMS[DroneModel.CF2X] == original


def test_default_xml_mechanics_and_actuators_match_previous_commit(tmp_path):
    import sys
    root = Path(__file__).resolve().parents[2]
    # Evaluate the old builder against the unchanged component builders.
    source = subprocess.check_output(["git", "show", "7faf780:multi_drone_mujoco/models/mjcf_builder.py"], cwd=root, text=True)
    scope = {"__file__": str(root / "multi_drone_mujoco/models/mjcf_builder.py"), "__name__": "_baseline_builder"}
    import types
    module = types.ModuleType("_baseline_builder")
    module.__dict__.update(scope)
    sys.modules[module.__name__] = module
    exec(compile(source, "baseline", "exec"), module.__dict__)
    args = run_sim.resolve_config(run_sim.build_parser().parse_args(["--headless"]))
    env = run_sim.create_manual_env(args)
    try:
        build = module.build_aviary_mjcf(1, DroneModel.CF2X, np.array([[0, 0, 1]]), np.zeros((1, 3)),
            timestep=1/2400, integrator="implicitfast", scaled_stablesoft_arm=args.arm_spec,
            scaled_stablesoft_visual=env.SCALED_STABLESOFT_VISUAL_SPEC, scaled_stablesoft_tendon_drive=args.tendon_drive)
        old = mujoco.MjModel.from_xml_string(build.xml)
        for name in ("body_mass", "body_inertia", "body_pos", "body_parentid", "body_jntnum",
                     "jnt_type", "jnt_stiffness", "dof_damping", "site_pos", "actuator_gear", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(env.model, name), getattr(old, name))
        root_id = env.model.body(env.SCALED_STABLESOFT_ARM_MANIFEST.segment_body_names[0]).id
        assert env.model.body_parentid[root_id] == env.model.body("drone0").id
        assert env.model.body_jntnum[root_id] == 0
    finally: env.close()


def test_gui_unlimited_explicit_duration_and_close_past_300(monkeypatch):
    import glfw
    args = run_sim.resolve_config(run_sim.build_parser().parse_args([]))
    assert args.duration is None
    assert run_sim.resolve_config(run_sim.build_parser().parse_args(["--headless"])).duration == 5.0
    assert run_sim.resolve_config(run_sim.build_parser().parse_args(["--duration", "301"])).duration == 301
    calls = {"steps": 0}
    env = SimpleNamespace(CTRL_FREQ=1, CTRL_TIMESTEP=1, pos=np.array([[0, 0, 1]]), model=None,
        data=SimpleNamespace(xfrc_applied=np.zeros((1, 6))), reset=lambda: None,
        clear_automatic_commands=lambda: None)
    env.step = lambda: calls.__setitem__("steps", calls["steps"] + 1)
    from contextlib import nullcontext
    def launch(*a, **kw):
        kw["key_callback"](glfw.KEY_SPACE)
        return SimpleNamespace(is_running=lambda: calls["steps"] < 302, sync=lambda: None,
            lock=nullcontext, cam=SimpleNamespace(lookat=np.zeros(3), distance=0), close=lambda: None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(run_sim.time, "sleep", lambda _: None)
    run_sim.run_manual(env, args)
    assert calls["steps"] == 302
    calls["steps"] = 0
    args.duration = 2
    run_sim.run_manual(env, args)
    assert calls["steps"] == 2


def test_cli_paths_follow_cwd_and_remote_auto_arm_is_rejected(tmp_path, monkeypatch):
    import yaml
    cfg = tmp_path / "config" / "sim.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump({"simulation": {"export_xml": "model.xml"}}))
    monkeypatch.chdir(tmp_path)
    args = run_sim.resolve_config(run_sim.build_parser().parse_args(["--config", str(cfg), "--export-xml", "cli.xml"]))
    assert args.export_xml == tmp_path / "cli.xml"
    cfg.write_text(yaml.safe_dump({"px4": {"mode": "remote", "launch_owned_px4": False}}))
    with pytest.raises(ValueError, match="automatic arming"):
        run_sim.resolve_config(run_sim.build_parser().parse_args(["--config", str(cfg)]))
