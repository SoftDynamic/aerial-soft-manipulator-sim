"""Headless tests for the manual active UAV-arm viewer demo."""

from __future__ import annotations

from pathlib import Path
import queue
from types import SimpleNamespace

import glfw
import mujoco
import numpy as np
import pytest

import multi_drone_mujoco.envs.base_aviary as base_aviary_module
import multi_drone_mujoco.examples.manual_aerial_soft_arm_demo as demo
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftVisualSpec,
)


def _state(initial_mode="zero", *, tension_limit=0.01):
    return demo.ManualCommandState.create(
        initial_mode=initial_mode,
        hover_rpm=15000.0,
        max_rpm=20000.0,
        tension_limit=tension_limit,
        rpm_step=100.0,
        tension_step=0.001,
    )


class _FakeWarning:
    def __init__(self, number=0):
        self.number = number


class _FakeData:
    def __init__(self):
        self.time = 0.0
        self.xfrc_applied = np.zeros((3, 6), dtype=float)
        self.ctrl = np.zeros(4, dtype=float)
        self.ten_length = np.full(4, 0.08, dtype=float)
        self.site_xpos = np.asarray([[0.0, 0.0, 0.9]], dtype=float)
        self.warning = tuple(_FakeWarning() for _ in range(8))


class _FakeEnv:
    SIM_FREQ = 2400
    CTRL_FREQ = 240
    SIM_STEPS_PER_CTRL = 10
    CTRL_TIMESTEP = 1.0 / CTRL_FREQ
    HOVER_RPM = 15000.0
    MAX_RPM = 20000.0

    def __init__(self, *, step_error=None):
        self.data = _FakeData()
        self.model = object()
        self.pos = np.asarray([[0.0, 0.0, 1.0]], dtype=float)
        self.rpy = np.zeros((1, 3), dtype=float)
        self.SCALED_STABLESOFT_TENDON_DRIVE_SPEC = SimpleNamespace(
            max_tension=demo.VERIFIED_ACTIVE_MAX_TENSION_N
        )
        self.events = []
        self.step_actions = []
        self.tension_commands = []
        self.snapshot_values = []
        self.cached_wrench = np.zeros(6)
        self.clear_calls = 0
        self.reset_calls = 0
        self.close_calls = 0
        self.step_error = step_error

    def reset(self):
        self.events.append("reset")
        self.reset_calls += 1
        self.data.time = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0
        return np.zeros(1), {}

    def snapshot_uav_viewer_wrenches(self):
        self.events.append("snapshot")
        self.cached_wrench = self.data.xfrc_applied[1].copy()
        self.snapshot_values.append(self.cached_wrench.copy())

    def clear_uav_viewer_wrenches(self):
        self.events.append("clear")
        self.cached_wrench[:] = 0.0
        self.clear_calls += 1

    def set_scaled_stablesoft_tension_command(self, command):
        values = tuple(command)
        self.events.append(("tension", values))
        self.tension_commands.append(values)
        self.data.ctrl[:] = values
        return values

    def step(self, action):
        values = np.asarray(action, dtype=float).copy()
        self.events.append(("step", tuple(values)))
        self.step_actions.append(values)
        if self.step_error is not None:
            raise self.step_error
        self.data.time += self.CTRL_TIMESTEP
        self.data.xfrc_applied[:] = 0.0
        return np.zeros(1), 0.0, False, False, {}

    def close(self):
        self.close_calls += 1


class _FakeViewer:
    def __init__(
        self,
        iterations,
        *,
        env=None,
        sync_wrenches=(),
        overlay_error=None,
    ):
        self.remaining = iterations
        self.env = env
        self.sync_wrenches = list(sync_wrenches)
        self.overlay_error = overlay_error
        self.sync_calls = 0
        self.close_calls = 0
        self.overlays = []

    def is_running(self):
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    def sync(self):
        self.sync_calls += 1
        if self.env is not None:
            self.env.events.append("sync")
            self.env.data.xfrc_applied[:] = 0.0
            if self.sync_wrenches:
                self.env.data.xfrc_applied[1] = self.sync_wrenches.pop(0)
            # Represent a user moving the native Control-panel sliders.
            self.env.data.ctrl[:] = -1.0

    def set_texts(self, texts):
        self.overlays.append(texts)
        if self.overlay_error is not None:
            raise self.overlay_error

    def close(self):
        self.close_calls += 1

    def lock(self):
        raise AssertionError("viewer.lock must not wrap env.step")


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _entities():
    return demo._ResolvedEntities(
        drone_body_id=1,
        b8_body_id=2,
        tip_site_id=0,
        tendon_ids=(0, 1, 2, 3),
        tendon_names=("t1", "t2", "t3", "t4"),
        actuator_names=("a1", "a2", "a3", "a4"),
    )


def _run_fake_loop(env, viewer, state, events=()):
    event_queue = queue.SimpleQueue()
    for keycode in events:
        event_queue.put(keycode)
    clock = _FakeClock()
    demo._run_manual_loop(
        env,
        viewer,
        state,
        _entities(),
        "mesh",
        event_queue,
        clock=clock,
        sleep_fn=clock.sleep,
    )
    return clock


def test_manual_state_starts_paused_and_supports_zero_and_hover_modes():
    zero = _state("zero")
    hover = _state("hover")

    assert zero.paused and not zero.pending_single_step
    np.testing.assert_array_equal(zero.rotor_rpm, np.zeros(4))
    np.testing.assert_array_equal(hover.rotor_rpm, np.full(4, 15000.0))
    np.testing.assert_array_equal(hover.tendon_tension, np.zeros(4))
    assert hover.status == "PAUSED"


def test_rotor_channels_select_adjust_and_clip_independently():
    state = _state()
    state.select_relative(1)
    state.adjust_selected(1)
    state.select_relative(1)
    state.rotor_rpm[2] = state.max_rpm
    state.adjust_selected(1)

    np.testing.assert_array_equal(state.rotor_rpm, [0.0, 100.0, 20000.0, 0.0])
    state.adjust_selected(-1)
    assert state.rotor_rpm[2] == 19900.0
    state.rotor_rpm[2] = 0.0
    state.adjust_selected(-1)
    assert state.rotor_rpm[2] == 0.0


def test_tendon_channels_select_adjust_and_never_go_negative():
    state = _state(tension_limit=0.003)
    state.toggle_group()
    state.select_relative(3)
    for _ in range(10):
        state.adjust_selected(1)
    assert state.tendon_tension[3] == pytest.approx(0.003)
    assert np.count_nonzero(state.tendon_tension[:3]) == 0
    for _ in range(10):
        state.adjust_selected(-1)
    assert state.tendon_tension[3] == 0.0
    assert np.all(state.tendon_tension >= 0.0)


def test_zero_shortcuts_only_clear_their_own_command_group():
    state = _state()
    state.rotor_rpm[:] = [1.0, 2.0, 3.0, 4.0]
    state.tendon_tension[:] = [0.001, 0.002, 0.003, 0.004]

    demo._apply_key(state, glfw.KEY_F8)
    np.testing.assert_array_equal(state.rotor_rpm, np.zeros(4))
    assert np.count_nonzero(state.tendon_tension) == 4
    demo._apply_key(state, glfw.KEY_F9)
    np.testing.assert_array_equal(state.tendon_tension, np.zeros(4))


def test_reset_restores_cli_initial_state_and_clears_pending_step():
    state = _state("hover")
    state.rotor_rpm[:] = 12.0
    state.tendon_tension[:] = 0.005
    state.selected_group = demo.GROUP_TENDON
    state.selected_channel = 3
    state.paused = False
    state.pending_single_step = True

    state.reset()

    np.testing.assert_array_equal(state.rotor_rpm, np.full(4, 15000.0))
    np.testing.assert_array_equal(state.tendon_tension, np.zeros(4))
    assert state.selected_group == demo.GROUP_ROTOR
    assert state.selected_channel == 0
    assert state.paused and not state.pending_single_step


def test_key_callback_only_enqueues_and_does_not_touch_state():
    state = _state()
    event_queue = queue.SimpleQueue()
    callback = demo._make_key_callback(event_queue)

    callback(glfw.KEY_RIGHT)

    np.testing.assert_array_equal(state.rotor_rpm, np.zeros(4))
    assert event_queue.get_nowait() == glfw.KEY_RIGHT


def test_each_received_key_event_changes_exactly_one_step():
    state = _state()
    demo._apply_key(state, glfw.KEY_RIGHT)
    assert state.rotor_rpm[0] == 100.0
    demo._apply_key(state, glfw.KEY_RIGHT)
    assert state.rotor_rpm[0] == 200.0


def test_paused_loop_syncs_and_sleeps_without_stepping():
    env = _FakeEnv()
    viewer = _FakeViewer(2, env=env)
    state = _state()

    clock = _run_fake_loop(env, viewer, state)

    assert viewer.sync_calls == 2
    assert env.step_actions == []
    assert len(clock.sleeps) == 2
    assert all(seconds == pytest.approx(demo.PAUSED_REFRESH_PERIOD_S) for seconds in clock.sleeps)
    assert np.count_nonzero(env.cached_wrench) == 0


def test_single_step_advances_one_environment_period_then_remains_paused():
    env = _FakeEnv()
    viewer = _FakeViewer(2, env=env)
    state = _state()

    _run_fake_loop(env, viewer, state, events=(glfw.KEY_PERIOD,))

    assert len(env.step_actions) == 1
    assert env.data.time == pytest.approx(1.0 / 240.0)
    assert state.paused and not state.pending_single_step
    assert "one control period" in state.notice
    assert any("SINGLE-STEP" in item[0][3] for item in viewer.overlays)


def test_period_while_running_does_not_add_an_extra_step():
    env = _FakeEnv()
    viewer = _FakeViewer(1, env=env)
    state = _state()

    _run_fake_loop(
        env,
        viewer,
        state,
        events=(glfw.KEY_SPACE, glfw.KEY_PERIOD),
    )

    assert len(env.step_actions) == 1
    assert not state.pending_single_step
    assert "ignored" in state.notice


def test_current_rpm_and_tension_are_forwarded_before_every_step():
    env = _FakeEnv()
    viewer = _FakeViewer(2, env=env)
    state = _state()
    state.rotor_rpm[:] = [100.0, 200.0, 300.0, 400.0]
    state.tendon_tension[:] = [0.001, 0.002, 0.003, 0.004]

    _run_fake_loop(env, viewer, state, events=(glfw.KEY_SPACE,))

    assert len(env.step_actions) == len(env.tension_commands) == 2
    for action in env.step_actions:
        np.testing.assert_array_equal(action, state.rotor_rpm)
    assert env.tension_commands == [tuple(state.tendon_tension)] * 2
    for index, event in enumerate(env.events):
        if isinstance(event, tuple) and event[0] == "step":
            assert env.events[index - 1][0] == "tension"
    np.testing.assert_array_equal(env.data.ctrl, state.tendon_tension)


def test_sync_precedes_snapshot_and_viewer_wrench_is_refreshed_each_cycle():
    env = _FakeEnv()
    first = np.arange(1.0, 7.0)
    viewer = _FakeViewer(
        2,
        env=env,
        sync_wrenches=(first, np.zeros(6)),
    )
    state = _state()

    _run_fake_loop(env, viewer, state, events=(glfw.KEY_SPACE,))

    np.testing.assert_array_equal(env.snapshot_values[0], first)
    np.testing.assert_array_equal(env.snapshot_values[1], np.zeros(6))
    first_snapshot = env.events.index("snapshot")
    assert env.events[first_snapshot - 1] == "sync"
    assert np.count_nonzero(env.cached_wrench) == 0


def test_loop_clears_cached_wrench_when_step_raises():
    env = _FakeEnv(step_error=RuntimeError("step failed"))
    viewer = _FakeViewer(1, env=env, sync_wrenches=(np.ones(6),))
    state = _state()

    with pytest.raises(RuntimeError, match="step failed"):
        _run_fake_loop(env, viewer, state, events=(glfw.KEY_SPACE,))

    np.testing.assert_array_equal(env.cached_wrench, np.zeros(6))


def test_reset_clears_cached_wrench_and_restores_state():
    env = _FakeEnv()
    env.cached_wrench[:] = 2.0
    viewer = _FakeViewer(1, env=env, sync_wrenches=(np.ones(6),))
    state = _state("hover")
    state.rotor_rpm[:] = 1.0

    _run_fake_loop(env, viewer, state, events=(glfw.KEY_BACKSPACE,))

    assert env.reset_calls == 1
    np.testing.assert_array_equal(env.cached_wrench, np.zeros(6))
    np.testing.assert_array_equal(state.rotor_rpm, np.full(4, 15000.0))
    assert state.paused and not state.pending_single_step


def test_overlay_is_a_list_of_four_tuples_and_contains_required_state():
    env = _FakeEnv()
    state = _state("hover")
    state.rotor_rpm[:] = [1.0, 2.0, 3.0, 4.0]
    state.tendon_tension[:] = [0.001, 0.002, 0.003, 0.004]

    overlay = demo._build_overlay(env, state, _entities(), "mesh")

    assert isinstance(overlay, list)
    assert all(isinstance(item, tuple) and len(item) == 4 for item in overlay)
    flattened = "\n".join(str(field) for item in overlay for field in item[2:])
    assert "OPEN-LOOP — NO FEEDBACK CONTROLLER" in flattened
    assert "OPEN-LOOP HOVER BIAS" in flattened
    assert "R1=1.0" in flattened and "R4=4.0" in flattened
    assert "T1=0.001000" in flattened and "T4=0.004000" in flattened
    assert "UAV RPY rad" in flattened
    assert "arm-tip world xyz" in flattened
    assert "Double-click select" in flattened


def test_overlay_value_error_warns_once_and_does_not_prevent_steps(capsys):
    env = _FakeEnv()
    viewer = _FakeViewer(
        2,
        env=env,
        overlay_error=ValueError("unsupported overlay"),
    )
    state = _state()

    _run_fake_loop(env, viewer, state, events=(glfw.KEY_SPACE,))

    assert len(env.step_actions) == 2
    assert len(viewer.overlays) == 1
    assert capsys.readouterr().err.count("viewer overlay disabled") == 1


@pytest.mark.parametrize("requested,expected", [(None, 0.01), (0.004, 0.004)])
def test_runtime_tension_limit_only_tightens_compiled_drive(requested, expected):
    drive = SimpleNamespace(max_tension=0.01)
    assert demo._runtime_tension_limit(requested, drive) == expected


@pytest.mark.parametrize("requested", [0.0, -0.1, 0.0100001, float("inf")])
def test_runtime_tension_limit_rejects_invalid_or_larger_values(requested):
    with pytest.raises(ValueError):
        demo._runtime_tension_limit(requested, SimpleNamespace(max_tension=0.01))


def test_cli_has_no_actuator_max_tension_option_and_validates_steps():
    parser = demo._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--rpm-step", "100",
                "--tension-step", "0.001",
                "--max-tension", "0.02",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--rpm-step", "0", "--tension-step", "0.001"])


def test_cli_defaults_fix_verified_frequency_integrator_boundary():
    args = demo._build_parser().parse_args(
        ["--rpm-step", "100", "--tension-step", "0.001"]
    )
    assert args.sim_freq == 2400
    assert args.ctrl_freq == 240
    assert args.visual_mode == ScaledStableSoftVisualSpec().mode == "mesh"
    assert args.tension_limit is None
    with pytest.raises(SystemExit):
        demo._build_parser().parse_args(
            [
                "--rpm-step", "100",
                "--tension-step", "0.001",
                "--sim-freq", "4800",
            ]
        )


def test_help_smoke_does_not_launch_a_viewer(capsys):
    with pytest.raises(SystemExit) as exc_info:
        demo._build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--initial-rpm" in output
    assert "Space run/pause" in output
    assert "Double-click select" in output


def test_runtime_source_imports_no_tests_and_contains_no_controller_or_mixer():
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "multi_drone_mujoco.tests" not in source
    assert "PIDControl" not in source
    assert "pid_control" not in source
    assert "def _mix" not in source.lower()
    assert "mixer(" not in source.lower()
    assert "mujoco.mj_step" not in source


def test_run_viewer_closes_viewer_once_then_environment(monkeypatch):
    env = _FakeEnv()
    viewer = _FakeViewer(0, env=env)
    close_order = []

    def env_close():
        close_order.append("env")
        env.close_calls += 1

    def viewer_close():
        close_order.append("viewer")
        viewer.close_calls += 1

    env.close = env_close
    viewer.close = viewer_close
    monkeypatch.setattr(demo, "_resolve_entities", lambda unused: _entities())
    monkeypatch.setattr(demo, "_run_manual_loop", lambda *args, **kwargs: None)
    args = demo._build_parser().parse_args(
        ["--rpm-step", "100", "--tension-step", "0.001"]
    )

    demo.run_viewer(
        args,
        env_factory=lambda **kwargs: env,
        launch_viewer=lambda *args, **kwargs: viewer,
        sleep_fn=lambda seconds: None,
    )

    assert viewer.close_calls == 1
    assert env.close_calls == 1
    assert close_order == ["viewer", "env"]


def test_run_viewer_closes_once_and_preserves_python_exception(monkeypatch):
    env = _FakeEnv()
    viewer = _FakeViewer(0, env=env)
    monkeypatch.setattr(demo, "_resolve_entities", lambda unused: _entities())

    def fail(*args, **kwargs):
        raise RuntimeError("loop failed")

    monkeypatch.setattr(demo, "_run_manual_loop", fail)
    args = demo._build_parser().parse_args(
        ["--rpm-step", "100", "--tension-step", "0.001"]
    )

    with pytest.raises(RuntimeError, match="loop failed"):
        demo.run_viewer(
            args,
            env_factory=lambda **kwargs: env,
            launch_viewer=lambda *args, **kwargs: viewer,
            sleep_fn=lambda seconds: None,
        )

    assert viewer.close_calls == 1
    assert env.close_calls == 1
    np.testing.assert_array_equal(env.cached_wrench, np.zeros(6))


@pytest.mark.parametrize("visual_mode", ["mesh", "cylinder"])
def test_demo_builds_verified_active_topology_without_changing_physics(visual_mode):
    env = demo._make_env(
        sim_freq=demo.DEFAULT_SIM_FREQ,
        ctrl_freq=demo.DEFAULT_CTRL_FREQ,
        visual_mode=visual_mode,
    )
    try:
        manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        assert env.SCALED_STABLESOFT_TENDON_DRIVE_SPEC.max_tension == 0.01
        assert env.model.ntendon == 4
        assert env.model.nu == 4
        assert env.model.na == 0
        assert len(manifest.segment_body_names) == 9
        assert len(manifest.tendon_names) == 4
        assert len(manifest.tendon_actuator_names) == 4
        assert env.model.opt.timestep == pytest.approx(1.0 / 2400)
        assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert env.M == pytest.approx(0.030, abs=1e-12)
        np.testing.assert_allclose(
            env.model.body_mass,
            [0.0, 0.027] + [1.0 / 3000.0] * 9,
            rtol=0.0,
            atol=1e-15,
        )
        expected_segment_inertia = [
            2.9447873799725653e-09,
            2.9447873799725653e-09,
            1.5e-09,
        ]
        np.testing.assert_allclose(
            env.model.body_inertia,
            [[0.0, 0.0, 0.0], [1.4e-05, 1.4e-05, 2.17e-05]]
            + [expected_segment_inertia] * 9,
            rtol=0.0,
            atol=1e-18,
        )
        assert not env.SCALED_STABLESOFT_ARM_SPEC.collision_enabled
    finally:
        env.close()


def test_manifest_runtime_bindings_are_resolved_once_in_command_order():
    env = demo._make_env(sim_freq=2400, ctrl_freq=240, visual_mode="mesh")
    try:
        entities = demo._resolve_entities(env)
        manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        assert entities.tendon_names == tuple(manifest.tendon_names)
        assert entities.actuator_names == tuple(manifest.tendon_actuator_names)
        assert entities.tendon_ids == tuple(
            binding.tendon_id
            for binding in env.SCALED_STABLESOFT_TENDON_BINDINGS
        )
        assert entities.b8_body_id == mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            manifest.segment_body_names[-1],
        )
    finally:
        env.close()


def test_native_uav_wrench_is_added_once_per_each_of_ten_substeps_and_b8_is_untouched(
    monkeypatch,
):
    env = demo._make_env(sim_freq=2400, ctrl_freq=240, visual_mode="mesh")
    manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
    drone_id = env._manual_drone_body_ids[0]
    b8_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        manifest.segment_body_names[-1],
    )
    native_uav = np.arange(1.0, 7.0) * 1e-8
    native_b8 = np.arange(7.0, 13.0) * 1e-8
    rpm = np.full(4, 0.25 * env.HOVER_RPM)
    records = []
    rotor_records = []
    real_mj_step = base_aviary_module.mujoco.mj_step
    real_base_physics = base_aviary_module.BaseAviary._physics

    def recording_base_physics(self, substep_rpm, nth_drone):
        real_base_physics(self, substep_rpm, nth_drone)
        rotor_records.append(
            self.data.xfrc_applied[self._manual_drone_body_ids[nth_drone]].copy()
        )

    def recording_step(model, data):
        records.append(
            (
                data.xfrc_applied[drone_id].copy(),
                data.xfrc_applied[b8_id].copy(),
            )
        )
        return real_mj_step(model, data)

    monkeypatch.setattr(base_aviary_module.mujoco, "mj_step", recording_step)
    monkeypatch.setattr(
        base_aviary_module.BaseAviary,
        "_physics",
        recording_base_physics,
    )
    try:
        env.data.xfrc_applied[drone_id] = native_uav
        env.data.xfrc_applied[b8_id] = native_b8
        env.snapshot_uav_viewer_wrenches()
        env.set_scaled_stablesoft_tension_command((0.0,) * 4)
        env.step(rpm)
    finally:
        env.clear_uav_viewer_wrenches()
        env.close()

    assert len(records) == len(rotor_records) == 10
    for rotor_wrench, (uav_wrench, b8_wrench) in zip(rotor_records, records):
        np.testing.assert_allclose(
            uav_wrench,
            rotor_wrench + native_uav,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(b8_wrench, native_b8)


def test_released_uav_wrench_and_reset_do_not_leak_into_later_cycles(monkeypatch):
    env = demo._make_env(sim_freq=2400, ctrl_freq=240, visual_mode="mesh")
    drone_id = env._manual_drone_body_ids[0]
    records = []
    real_mj_step = base_aviary_module.mujoco.mj_step

    def recording_step(model, data):
        records.append(data.xfrc_applied[drone_id].copy())
        return real_mj_step(model, data)

    monkeypatch.setattr(base_aviary_module.mujoco, "mj_step", recording_step)
    try:
        env.data.xfrc_applied[drone_id] = np.ones(6) * 1e-8
        env.snapshot_uav_viewer_wrenches()
        env.step(np.zeros(4))
        env.clear_uav_viewer_wrenches()

        # viewer.sync after release would write a zero row; emulate that here.
        env.data.xfrc_applied[drone_id] = 0.0
        env.snapshot_uav_viewer_wrenches()
        env.step(np.zeros(4))
        env.clear_uav_viewer_wrenches()

        env.set_uav_viewer_wrench(0, np.ones(6))
        env.reset()
        env.clear_uav_viewer_wrenches()
        np.testing.assert_array_equal(
            env._manual_uav_viewer_wrenches,
            np.zeros((1, 6)),
        )
    finally:
        env.close()

    assert len(records) == 20
    for wrench in records[:10]:
        np.testing.assert_array_equal(wrench, np.ones(6) * 1e-8)
    for wrench in records[10:]:
        np.testing.assert_array_equal(wrench, np.zeros(6))
