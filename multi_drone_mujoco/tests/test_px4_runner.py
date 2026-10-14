from pathlib import Path
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.backends.actuation import (
    PwmToRotorSpeedModel,
    Px4NormalizedOutputToPwm,
)
from multi_drone_mujoco.backends.cf2x_px4 import (
    cf2x_normalized_to_pwm_config,
    cf2x_pwm_to_rpm_config,
)
from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.lockstep import LockstepConfig
from multi_drone_mujoco.backends.interaction import (
    DeterministicInteractionTimeline,
)
from multi_drone_mujoco.backends.px4_environment import Px4EnvironmentMode
from multi_drone_mujoco.backends.px4_runner import (
    Px4MujocoRunner,
    Px4RunnerConfig,
)
from multi_drone_mujoco.backends.px4_sitl import PX4SITLBackend
from multi_drone_mujoco.backends.sensors import IdealSensorAdapter, IdealSensorConfig, MujocoTruthSource
from multi_drone_mujoco.backends.transport import FakeTransport


class FakeAviary:
    def __init__(self):
        self.pos = np.array([[0.0, 0.0, 1.0]])
        self.vel = np.zeros((1, 3))
        self.quat = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.ang_v = np.zeros((1, 3))
        self.rpy = np.zeros((1, 3))
        self.actions = []
        self.closed = False
        self.SCALED_STABLESOFT_ARM_MANIFEST = None
        from multi_drone_mujoco.models.parameters import ResolvedUavParameters
        self.UAV_PARAMETERS = ResolvedUavParameters.resolve()
        self.ARM_MASS = self.PAYLOAD_MASS = 0.0
        self.HOVER_RPM = 15000.0
        self._generated_xml = '<mujoco model="fake"/>'
        self.data = SimpleNamespace(ncon=0, ctrl=np.zeros(0))
        self.SCALED_STABLESOFT_TENDON_BINDINGS = ()
        self.command = np.zeros(4)

    def reset(self, seed=None):
        self.pos[:] = (0.0, 0.0, 1.0)
        self.vel[:] = 0.0
        self.actions.clear()

    def set_rotor_rpm_command(self, rpm, *, armed=True):
        self.command = np.asarray(rpm) if armed else np.zeros(4)

    def clear_automatic_commands(self):
        self.command = np.zeros(4)

    def step(self, action=None):
        self.actions.append(tuple(self.command))

    def render(self):
        pass

    def close(self):
        self.closed = True


def make_backend(transport):
    return PX4SITLBackend(
        IdealSensorAdapter(IdealSensorConfig(
            30.0, 120.0, 100.0, 9.81, (0.3, 0.0, 0.4),
            400.0, 10.0, 50.0, 50.0,
        )),
        transport,
        Px4NormalizedOutputToPwm(cf2x_normalized_to_pwm_config()),
        PwmToRotorSpeedModel(cf2x_pwm_to_rpm_config()),
        LockstepConfig(0.1, (0.0, 0.0, 0.0, 0.0)),
    )


def test_runner_config_separates_remote_from_local_checkout(tmp_path):
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="0.0.0.0",
        bind_port=4560,
        duration_s=0.1,
        log_dir=tmp_path,
    )
    assert config.autopilot_dir is None
    assert not config.launch_owned_px4


def test_interaction_config_requires_arm_viewer_and_explicit_high_force(tmp_path):
    event_path = tmp_path / "events.json"
    event_path.write_text("[]")
    with pytest.raises(ValueError, match="passive_arm or active_arm"):
        Px4RunnerConfig(
            mode="remote", bind_host="127.0.0.1", bind_port=4560,
            interaction_mode="replay", interaction_script=event_path,
        )
    with pytest.raises(ValueError, match="requires --viewer"):
        Px4RunnerConfig(
            mode="remote", bind_host="127.0.0.1", bind_port=4560,
            plant="passive_arm", interaction_mode="mouse_force",
        )
    with pytest.raises(ValueError, match="requires allow_unverified_force"):
        Px4RunnerConfig(
            mode="remote", bind_host="127.0.0.1", bind_port=4560,
            plant="passive_arm", interaction_mode="replay",
            interaction_script=event_path, interaction_max_force_n=0.010,
        )
    config = Px4RunnerConfig(
        mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="passive_arm", interaction_mode="replay",
        interaction_script=event_path, interaction_max_force_n=0.010,
        allow_unverified_force=True,
    )
    assert config.interaction_max_force_n == 0.010


def test_owned_launch_requires_local_checkout_and_exact_instance_port(tmp_path):
    with pytest.raises(ValueError, match="local mode"):
        Px4RunnerConfig(
            mode=Px4EnvironmentMode.REMOTE,
            bind_host="127.0.0.1",
            bind_port=4560,
            launch_owned_px4=True,
            autopilot_dir=tmp_path,
            simulator_host="127.0.0.1",
        )

    with pytest.raises(ValueError, match="connects to port 4561"):
        Px4RunnerConfig(
            mode=Px4EnvironmentMode.LOCAL,
            bind_host="127.0.0.1",
            bind_port=4560,
            launch_owned_px4=True,
            autopilot_dir=tmp_path,
            simulator_host="127.0.0.1",
            px4_instance=1,
        )


def test_fractional_microsecond_period_uses_drift_free_clock(tmp_path):
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        sim_freq_hz=2400,
        control_freq_hz=240,
        log_dir=tmp_path,
    )
    runner = Px4MujocoRunner(
        config, FakeAviary(), make_backend(FakeTransport()), MujocoTruthSource()
    )
    timestamps = [runner._clock(index).timestamp_us for index in range(4)]
    assert timestamps == [1_000_000, 1_004_167, 1_008_333, 1_012_500]


def test_runner_converts_absolute_target_to_px4_local_origin(tmp_path):
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        initial_position_enu_m=(1.0, 2.0, 3.0),
        target_position_enu_m=(1.5, 1.0, 4.0),
        log_dir=tmp_path,
    )
    runner = Px4MujocoRunner(
        config, FakeAviary(), make_backend(FakeTransport()), MujocoTruthSource()
    )
    assert runner._target_local_enu() == (0.5, -1.0, 1.0)
    assert runner._spool_target_local_enu() == (0.5, -1.0, 2.0)
    assert runner._flight_target_local_enu(0.0) == (0.5, -1.0, 2.0)
    assert runner._flight_target_local_enu(1.0) == (0.5, -1.0, 1.0)


def test_active_arm_uses_exact_200_240_hz_common_scheduler(tmp_path):
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        plant="active_arm",
        duration_s=6.0,
        log_dir=tmp_path,
    )
    assert config.plant_step_freq_hz == 1200
    assert config.plant_step_freq_hz // config.control_freq_hz == 6
    assert config.plant_step_freq_hz // 240 == 5
    with pytest.raises(ValueError, match="divisible by 240 Hz"):
        Px4RunnerConfig(
            mode="remote", bind_host="127.0.0.1", bind_port=4560,
            plant="active_arm", plant_step_freq_hz=200, log_dir=tmp_path,
        )


def test_fake_runner_bootstraps_then_advances_physics_and_closes(tmp_path):
    packets = [
        NormalizedMotorCommand((0.0,) * 4, False, 1_000_000, 1),
        NormalizedMotorCommand((0.2,) * 4, True, 1_005_000, 2),
        NormalizedMotorCommand((0.2,) * 4, True, 1_010_000, 3),
    ]
    transport = FakeTransport(packets)
    aviary = FakeAviary()
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        duration_s=2 / 200,
        bootstrap_wall_interval_s=0.0,
        log_dir=tmp_path,
    )
    runner = Px4MujocoRunner(
        config, aviary, make_backend(transport), MujocoTruthSource()
    )
    metrics = runner.run()
    assert metrics.outcome == "completed"
    assert metrics.bootstrap_frames == 1
    assert metrics.physics_steps == 2
    assert len(aviary.actions) == 2
    assert aviary.closed
    assert Path(metrics.log_path).is_file()
    payload = json.loads(Path(metrics.log_path).read_text())
    assert payload["schema_version"] == 5
    assert payload["started_utc"] < payload["ended_utc"]
    assert payload["metrics"]["log_path"] == metrics.log_path
    assert transport.health.state.value == "closed"


def test_fake_runner_timeout_never_advances_physics(tmp_path):
    transport = FakeTransport([
        NormalizedMotorCommand((0.0,) * 4, False, 1_000_000, 1),
    ])
    aviary = FakeAviary()
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        duration_s=1 / 200,
        bootstrap_wall_interval_s=0.0,
        log_dir=tmp_path,
    )
    runner = Px4MujocoRunner(
        config, aviary, make_backend(transport), MujocoTruthSource()
    )
    metrics = runner.run()
    assert metrics.outcome == "timed_out"
    assert metrics.physics_steps == 0
    assert aviary.actions == []
    assert metrics.timeout_count == 1


def _fake_packets(count, *, normalized=2.0 / 3.0):
    return [
        NormalizedMotorCommand(
            ((0.0,) * 4 if index == 0 else (normalized,) * 4),
            index != 0,
            1_000_000 + index * 5_000,
            index + 1,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("gui", [False, True])
def test_replay_applies_tip_force_once_per_step_and_clears_it(tmp_path, monkeypatch, gui):
    if gui:
        from mujoco import viewer as native_viewer
        viewer = SimpleNamespace(
            perturb=SimpleNamespace(select=0), is_running=lambda: True,
            sync=lambda: None, close=lambda: None,
        )
        monkeypatch.setattr(native_viewer, "launch_passive", lambda *a, **kw: viewer)
    event_path = tmp_path / "force.json"
    event_path.write_text(json.dumps({"events": [{
        "start_s": 0.005,
        "duration_s": 0.010,
        "mode": "external_force",
        "force_n": [0.002, 0.0, 0.0],
    }]}))
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        plant="passive_arm",
        duration_s=0.020,
        bootstrap_wall_interval_s=0.0,
        gui=gui, interaction_mode="replay",
        interaction_script=event_path,
        interaction_max_force_n=0.005,
        log_dir=tmp_path,
    )
    created = Px4MujocoRunner.create_default(config)
    created.backend.transport = FakeTransport(_fake_packets(5))
    created.backend.coordinator.transport = created.backend.transport
    body_name, _ = created._resolve_interaction_target()
    body_id = mujoco.mj_name2id(
        created.aviary.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    before = []
    after = []
    real_step = created.aviary.step

    def recording_step(action=None):
        before.append(created.aviary.data.xfrc_applied[body_id].copy())
        result = real_step(action)
        after.append(created.aviary.data.xfrc_applied[body_id].copy())
        return result

    created.aviary.step = recording_step
    metrics = created.run()
    assert metrics.outcome == "completed"
    assert metrics.max_external_force_n == pytest.approx(0.002)
    assert metrics.interaction_command_count == 2
    assert metrics.interaction_release_count == 1
    assert sum(np.linalg.norm(item[:3]) > 0.0 for item in before) == 2
    assert all(np.count_nonzero(item) == 0 for item in after)
    replay = DeterministicInteractionTimeline.from_json(metrics.log_path)
    assert len(replay.events) == 1
    assert replay.events[0].force_n == pytest.approx((0.002, 0.0, 0.0))


def test_headless_tendon_replay_enters_frozen_modal_controller(tmp_path, monkeypatch):
    event_path = tmp_path / "tendon.json"
    event_path.write_text(json.dumps({"events": [{
        "start_s": 0.004,
        "duration_s": 0.016,
        "mode": "tendon_modal",
        "tendon_acceleration_m_s2": [0.005, -0.005],
    }]}))
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.REMOTE,
        bind_host="127.0.0.1",
        bind_port=4560,
        plant="active_arm",
        arm_source="controller",
        duration_s=0.025,
        bootstrap_wall_interval_s=0.0,
        interaction_mode="replay",
        interaction_script=event_path,
        log_dir=tmp_path,
    )
    created = Px4MujocoRunner.create_default(config)
    import multi_drone_mujoco.control.ideal_tendon_tension as tension_module

    def unexpected_resolution(*args):
        pytest.fail("the runner must reuse the environment's actuator bindings")

    monkeypatch.setattr(tension_module, "resolve_ideal_tendon_actuator_bindings", unexpected_resolution)
    original_parse = DeterministicInteractionTimeline.from_json
    parsed_paths = []

    def counted_parse(path):
        parsed_paths.append(path)
        return original_parse(path)

    monkeypatch.setattr(DeterministicInteractionTimeline, "from_json", staticmethod(counted_parse))
    created.backend.transport = FakeTransport(_fake_packets(6))
    created.backend.coordinator.transport = created.backend.transport
    metrics = created.run()
    assert metrics.outcome == "completed"
    assert metrics.interaction_command_count > 0
    assert metrics.interaction_release_count == 1
    assert parsed_paths == [event_path]
    assert metrics.max_tendon_tension_n < 0.010
    assert np.linalg.norm(metrics.final_tendon_reference_m) > 0.0
    assert np.linalg.norm(metrics.final_tendon_reference_m, ord=1) <= 300e-6


def test_headless_joint_diagnostic_replay_writes_only_arm_dof_and_clears(tmp_path):
    probe = Px4RunnerConfig(
        mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="passive_arm", duration_s=0.020,
        bootstrap_wall_interval_s=0.0, log_dir=tmp_path,
    )
    seed_runner = Px4MujocoRunner.create_default(probe)
    joint = seed_runner.aviary.SCALED_STABLESOFT_ARM_MANIFEST.ball_joint_names[3]
    seed_runner.close()
    event_path = tmp_path / "joint.json"
    event_path.write_text(json.dumps({"events": [{
        "start_s": 0.005, "duration_s": 0.010,
        "mode": "joint_torque", "target_joint": joint,
        "target_dof_axis": 1, "joint_torque_nm": 10e-6,
    }]}))
    config = Px4RunnerConfig(
        mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="passive_arm", duration_s=0.020,
        bootstrap_wall_interval_s=0.0, interaction_mode="replay",
        interaction_script=event_path, log_dir=tmp_path,
    )
    created = Px4MujocoRunner.create_default(config)
    created.backend.transport = FakeTransport(_fake_packets(5))
    created.backend.coordinator.transport = created.backend.transport
    joint_id = mujoco.mj_name2id(
        created.aviary.model, mujoco.mjtObj.mjOBJ_JOINT, joint
    )
    target_dof = int(created.aviary.model.jnt_dofadr[joint_id]) + 1
    before = []
    after = []
    real_step = created.aviary.step

    def recording_step(action=None):
        before.append(created.aviary.data.qfrc_applied.copy())
        result = real_step(action)
        after.append(created.aviary.data.qfrc_applied.copy())
        return result

    created.aviary.step = recording_step
    metrics = created.run()
    assert metrics.outcome == "completed"
    assert metrics.max_joint_torque_nm == pytest.approx(10e-6)
    assert sum(abs(item[target_dof]) > 0.0 for item in before) == 2
    assert all(np.count_nonzero(item[:6]) == 0 for item in before)
    assert all(
        np.count_nonzero(np.delete(item, target_dof)) == 0 for item in before
    )
    np.testing.assert_array_equal(created.aviary.data.qfrc_applied, 0.0)


def _diagnostic_runner(tmp_path, trial="pulse"):
    from collections import deque
    import yaml
    from scripts.run_sim import resolve_config, build_parser
    file = tmp_path / "sim.yaml"
    file.write_text(yaml.safe_dump({"simulation": {"uav_source": "px4", "gui": False, "duration_s": 7.3},
                                    "diagnostics": {"trial": trial}}))
    args = resolve_config(build_parser().parse_args(["--config", str(file)]))
    config = Px4RunnerConfig(**args.px4, gui=False, plant="active_arm", duration_s=7.3,
                             diagnostics=args.diagnostics, log_dir=tmp_path)
    runner = Px4MujocoRunner.create_default(config)
    runner.aviary.reset()
    runner._diagnostic_samples = deque(maxlen=256)
    runner._baseline_sum = 0.0
    runner._baseline_count = 0
    runner._baseline_stable = True
    runner._diagnostic_done = False
    runner._final_actuator_armed = True
    runner.flight_control = None
    return runner


@pytest.mark.parametrize("stable", [True, False])
def test_diagnostic_baseline_gate_real_actuator_pulse_and_full_rate_peaks(tmp_path, stable):
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    runner = _diagnostic_runner(tmp_path)
    metrics = Px4RunMetrics()
    env = runner.aviary
    try:
        tip = runner._arm_tip_relative()
        dt = 1 / 1200
        for index in range(1, 2401):
            runner._diagnostic_tension((index-1)*dt, metrics)
            metrics.physics_steps = index
            runner._measure_diagnostic(index*dt, 0.02 if stable else 0.11, 0.03, tip, tip, metrics)
            assert not np.any(env.data.ctrl[4:])
        assert metrics.baseline_status == ("passed" if stable else "未在规定窗口内满足悬停判据")
        assert metrics.interaction_recovery_time_s is None
        runner._diagnostic_tension(2.0, metrics)
        if not stable:
            assert runner._diagnostic_done
            assert metrics.pulse_start_s is None
            assert not np.any(env.data.ctrl[4:])
            return
        assert metrics.pulse_start_s == 2.0
        rotor = env.ROTOR_CONTROL_ADDRESSES[0]
        env.set_rotor_rpm_command(np.sqrt(np.array([0.01, 0.02, 0.03, 0.04]) / env.KF))
        expected_rotor = env._rotor_thrust_command_n[0].copy()
        bindings = env.SCALED_STABLESOFT_TENDON_BINDINGS
        env.step()
        actual = [env.data.actuator_force[b.output_address] for b in bindings]
        assert actual == pytest.approx([0.001, 0, 0, 0])
        np.testing.assert_array_equal(env.data.ctrl[rotor], expected_rotor)
        metrics.physics_steps = 2401  # deliberately off saved sample cadence
        runner._measure_diagnostic(2+dt, 0.15, 0.25, tip, tip, metrics)
        assert metrics.diagnostic_peak_position_error_m == 0.15
        assert metrics.diagnostic_peak_speed_m_s == 0.25
        assert metrics.actual_pulse_max_tension_n == 0.001
        assert len(runner._diagnostic_samples) == 40
        runner._diagnostic_tension(2.3-dt, metrics)
        assert env.data.ctrl[bindings[0].control_address] == 0.001
        runner._diagnostic_tension(2.3, metrics)
        assert metrics.pulse_release_s == 2.3
        assert metrics.pulse_release_s-metrics.pulse_start_s == pytest.approx(0.30)
        assert [env.data.ctrl[b.control_address] for b in bindings] == [0]*4
        env.step()
        assert [env.data.actuator_force[b.output_address] for b in bindings] == [0]*4
        metrics.physics_steps = 8760
        runner._measure_diagnostic(7.3, 0.15, 0.25, tip, tip, metrics)
        assert metrics.recovery_status == "未在规定恢复窗口内达标"
        assert runner._diagnostic_done
    finally: runner.close()


def test_recovery_requires_continuous_window_and_each_release_resets_result(tmp_path):
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    from multi_drone_mujoco.backends.interaction import InteractionCommand, InteractionMode, InteractionFrame
    runner = _diagnostic_runner(tmp_path)
    metrics = Px4RunMetrics(last_interaction_release_s=2.3)
    try:
        candidate = None
        for t in (2.4, 2.7, 3.0):
            candidate = runner._update_recovery(t, 0.05, 0.05, candidate, metrics)
        assert metrics.interaction_recovery_time_s is None
        candidate = runner._update_recovery(3.1, 0.11, 0.05, candidate, metrics)
        assert candidate is None
        candidate = runner._update_recovery(3.2, 0.05, 0.05, candidate, metrics)
        candidate = runner._update_recovery(4.2, 0.05, 0.05, candidate, metrics)
        assert metrics.interaction_recovery_time_s == pytest.approx(0.9)
        command = InteractionCommand(timestamp_us=5000000, sequence=1, mode=InteractionMode.DISABLED,
            target_body=None, target_site=None, frame=InteractionFrame.WORLD_ENU, source="test", released=True)
        runner._record_interaction_command(command, 5.0, metrics)
        assert metrics.interaction_recovery_time_s is None
        assert metrics.last_interaction_release_s == 5.0
        # Run loop resets the candidate on this release, preventing old reuse.
        candidate = runner._update_recovery(5.1, 0.05, 0.05, None, metrics)
        assert metrics.interaction_recovery_time_s is None
        runner._update_recovery(6.1, 0.05, 0.05, candidate, metrics)
        assert metrics.interaction_recovery_time_s == pytest.approx(0.1)
    finally: runner.close()


def test_ordinary_gui_does_not_accumulate_detailed_interaction_events(tmp_path):
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    from multi_drone_mujoco.backends.interaction import InteractionCommand, InteractionMode, InteractionFrame
    runner = Px4MujocoRunner.create_default(Px4RunnerConfig(mode="remote", bind_host="127.0.0.1", bind_port=4560,
        gui=True, plant="active_arm", log_dir=tmp_path))
    try:
        metrics = Px4RunMetrics()
        for i in range(10000):
            cmd = InteractionCommand(timestamp_us=i+1, sequence=i+1, mode=InteractionMode.EXTERNAL_FORCE,
                target_body="drone0", target_site=None, frame=InteractionFrame.WORLD_ENU, force_n=(0.001*(i%2), 0, 0), source="test", active=True)
            runner._record_interaction_command(cmd, i/1200, metrics)
        assert len(runner._interaction_events) == 0
    finally: runner.close()


def test_legacy_gui_close_past_300_lands_without_closed_viewer(tmp_path, monkeypatch):
    from multi_drone_mujoco.backends.px4_control import Px4FlightControlConfig, Px4MavlinkFlightControl
    aviary = FakeAviary()
    native = SimpleNamespace(is_running=lambda: len(aviary.actions) < 602,
                             sync=lambda: None, close=lambda: None)
    aviary.render = lambda: setattr(aviary, "_viewer", native)
    config = Px4RunnerConfig(mode="local", bind_host="127.0.0.1", bind_port=4560,
        gui=True, duration_s=None, sim_freq_hz=2, control_freq_hz=2, auto_flight=True,
        launch_owned_px4=True, autopilot_dir=Path("/home/ustc/PX4-Autopilot"), simulator_host="127.0.0.1",
        gcs_bind_host="127.0.0.1", gcs_bind_port=14550, log_dir=tmp_path)
    packets = [NormalizedMotorCommand((0.2,)*4, True, 1000000+i*500000, i+1) for i in range(604)]
    control = Px4MavlinkFlightControl(Px4FlightControlConfig("127.0.0.1", 14550))
    monkeypatch.setattr(control, "open", lambda: None)
    monkeypatch.setattr(control, "poll", lambda: None)
    monkeypatch.setattr(control, "send_position_setpoint_enu", lambda *a: None)
    runner = Px4MujocoRunner(config, aviary, make_backend(FakeTransport(packets)), MujocoTruthSource(), flight_control=control)
    monkeypatch.setattr(runner, "_prepare_auto_flight", lambda *a: None)
    landed = []
    def land(*a):
        assert aviary._viewer is None
        landed.append(True)
        runner._final_actuator_armed = False
        return 1
    monkeypatch.setattr(runner, "_land_and_disarm", land)
    metrics = runner.run()
    assert metrics.outcome == "completed" and metrics.simulation_duration_s == 301
    assert metrics.landed and landed == [True]
    assert not np.any(aviary.command)


def test_pause_resume_keeps_clock_cadence_and_zeroes_until_fresh_output(tmp_path, monkeypatch):
    config = Px4RunnerConfig(mode="remote", bind_host="127.0.0.1", bind_port=4560,
        plant="active_arm", gui=True, duration_s=0.01, log_dir=tmp_path,
        bootstrap_wall_interval_s=0)
    runner = Px4MujocoRunner.create_default(config)
    transport = FakeTransport([
        NormalizedMotorCommand((0,)*4, False, 1000000, 1),
        NormalizedMotorCommand((0.2,)*4, True, 1005000, 2),
        NormalizedMotorCommand((0.2,)*4, True, 1005000, 2), # stale pre-pause packet
        NormalizedMotorCommand((0.4,)*4, True, 1010000, 3)])
    runner.backend.transport = runner.backend.coordinator.transport = transport
    paused = []
    class Viewer:
        exit_requested = False
        permission = True
        def poll(self, t):
            if t > 1/1200 and not paused:
                self.permission = False
                paused.append(t)
            else: self.permission = True
            return runner._sample_interaction(t, Px4RunMetrics())
        def consume_advance_permission(self): return self.permission
        def after_step(self): pass
        def render_only(self): pass
        def close(self): pass
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    monkeypatch.setattr(runner, "_open_interactive_viewer", lambda: setattr(runner, "_interaction_viewer", Viewer()))
    monkeypatch.setattr("multi_drone_mujoco.backends.px4_runner.time.sleep", lambda _: None)
    original = runner.aviary.step
    commands = []
    def step():
        original()
        commands.append(runner.aviary.data.ctrl[runner.aviary.ROTOR_CONTROL_ADDRESSES[0]].copy())
    runner.aviary.step = step
    metrics = runner.run()
    assert metrics.outcome == "completed" and len(commands) == 12
    assert np.all(commands[0] > 0)
    for command in commands[1:6]: np.testing.assert_array_equal(command, 0)
    assert np.all(commands[6] > commands[0])
    assert [f.timestamp_us for f in transport.sent_sensor_frames] == [1000000, 1005000, 1010000]
    assert [f.sequence for f in transport.sent_sensor_frames] == [0, 1, 2]
    assert not np.any(runner.aviary.data.ctrl[runner.aviary.ROTOR_CONTROL_ADDRESSES[0]])


def test_diagnostic_cleanup_returns_owned_tendon_group_to_zero(tmp_path):
    runner = _diagnostic_runner(tmp_path)
    runner.aviary.set_scaled_stablesoft_tension_command((0.001, 0, 0, 0))
    runner.close()
    assert [runner.aviary.data.ctrl[b.control_address] for b in runner.aviary.SCALED_STABLESOFT_TENDON_BINDINGS] == [0]*4


def test_nan_measurement_cannot_start_recovery_window(tmp_path):
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    runner = _diagnostic_runner(tmp_path)
    try:
        metrics = Px4RunMetrics(last_interaction_release_s=2.3)
        assert runner._update_recovery(3.0, float("nan"), 0, None, metrics) is None
        assert metrics.interaction_recovery_time_s is None
    finally: runner.close()


def test_actual_sent_setpoint_phase_and_nonzero_world_offset(tmp_path):
    from dataclasses import replace
    from multi_drone_mujoco.backends.contracts import SimulatorClock
    runner = _diagnostic_runner(tmp_path, "baseline")
    runner.config = replace(runner.config, initial_position_enu_m=(2, 3, 4), target_position_enu_m=(5, 6, 7))
    sent = []
    runner.flight_control = SimpleNamespace(send_position_setpoint_enu=lambda t, p: sent.append((t, p)))
    try:
        for index, t in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            clock = SimulatorClock(2000000 + index*250000, index+1)
            local = runner._flight_target_local_enu(t)
            phase = "height_setpoint_transition" if t < 1 else "fixed_target_hover"
            runner._send_position_setpoint(clock, local, phase, t)
            actual = runner._last_position_setpoint
            assert sent[-1] == (clock.timestamp_us, local)
            assert actual["setpoint_local_enu_m"] == local
            assert actual["setpoint_world_enu_m"] == pytest.approx((5, 6, 8-t))
            assert actual["setpoint_sent_time_since_release_s"] == t
            assert actual["setpoint_transition_complete"] == (t >= 1)
        assert runner._flight_phase_timing["fixed_target_hover"]["timestamp_us"] == 3000000
        # This deliberately non-profile command proves the recorder observes the
        # transmitted argument instead of reconstructing a nominal trajectory.
        runner._send_position_setpoint(SimulatorClock(3005000, 6), (0.42, 0.43, 0.44), "fixed_target_hover", 1.005)
        assert runner._last_position_setpoint["setpoint_world_enu_m"] == pytest.approx((2.42, 3.43, 4.44))
        runner.flight_control = None
    finally: runner.close()


@pytest.mark.parametrize("outlier", [False, True])
def test_eight_second_baseline_final_window_high_rate_peaks_and_bounded_samples(tmp_path, outlier):
    from dataclasses import replace
    from multi_drone_mujoco.backends.px4_runner import Px4RunMetrics
    from multi_drone_mujoco.backends.contracts import SimulatorClock
    runner = _diagnostic_runner(tmp_path, "baseline")
    runner.config = replace(runner.config, duration_s=8, diagnostics={**runner.config.diagnostics, "baseline_s": 8})
    metrics = Px4RunMetrics()
    env = runner.aviary
    tip = runner._arm_tip_relative()
    runner.flight_control = SimpleNamespace(send_position_setpoint_enu=lambda *a: None)
    runner._send_position_setpoint(SimulatorClock(2005000, 201), (0, 0, 0), "fixed_target_hover", 1)
    runner.flight_control = None
    try:
        for i in range(1, 9601):
            t = i/1200
            runner._diagnostic_tension((i-1)/1200, metrics)
            env.pos[0, 2] = 1.05
            env.vel[0, 2] = 0.01
            # Startup outliers must not fail the final 7..8 s criterion.
            error, speed = (0.2, 0.3) if i == 601 else (0.05, 0.01)
            if i == 601: env.pos[0, 2] = 1.2
            if i == 8401 and outlier: error = 0.11
            metrics.physics_steps = i
            runner._measure_diagnostic(t, error, speed, tip, tip, metrics)
        assert metrics.baseline_status == ("未在规定窗口内满足悬停判据" if outlier else "passed")
        stats = metrics.hover_baseline
        assert stats["height_max"] == 1.2 and stats["height_max_time_since_release_s"] == pytest.approx(601/1200)
        assert metrics.baseline_max_speed_m_s == 0.3
        assert stats["final_window"]["count"] == 1200
        assert stats["final_window"]["max_position_error_m"] == (0.11 if outlier else 0.05)
        assert len(runner._diagnostic_samples) == 160 and runner._diagnostic_samples.maxlen == 256
        assert max(x["position_error_m"] for x in runner._diagnostic_samples) == 0.05
        sample = runner._diagnostic_samples[-1]
        assert sample["time_since_release_s"] == 8
        assert sample["final_target_world_enu_m"] == (0, 0, 1)
        assert sample["setpoint_world_enu_m"] == (0, 0, 1)
        assert sample["flight_phase"] == "fixed_target_hover" and sample["setpoint_transition_complete"]
        assert metrics.pulse_start_s is None and metrics.pulse_release_s is None
        assert not any(sample["tendon_commands_n"])
        # The same bounded recorder stays capped if fed a longer synthetic run.
        for i in range(9601, 18001):
            metrics.physics_steps = i
            runner._measure_diagnostic(i/1200, 0.05, 0.01, tip, tip, metrics)
        assert len(runner._diagnostic_samples) == 256
    finally: runner.close()
