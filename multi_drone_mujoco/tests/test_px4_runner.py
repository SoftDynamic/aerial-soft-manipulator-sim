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
        self.HOVER_RPM = 15000.0
        self._generated_xml = '<mujoco model="fake"/>'
        self.data = SimpleNamespace(ncon=0, ctrl=np.zeros(0))

    def reset(self, seed=None):
        self.pos[:] = (0.0, 0.0, 1.0)
        self.vel[:] = 0.0
        self.actions.clear()

    def step(self, action):
        self.actions.append(tuple(action))

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
    assert payload["schema_version"] == 4
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


def test_headless_replay_applies_tip_force_once_per_step_and_clears_it(tmp_path):
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
        interaction_mode="replay",
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

    def recording_step(action):
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

    def recording_step(action):
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
