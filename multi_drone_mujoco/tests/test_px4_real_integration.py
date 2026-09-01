"""Opt-in external PX4 integration preflight; ordinary tests never start PX4."""

import os
from pathlib import Path

import pytest

from multi_drone_mujoco.backends.px4_environment import inspect_px4_environment
from multi_drone_mujoco.backends.px4_environment import Px4EnvironmentMode
from multi_drone_mujoco.backends.px4_runner import Px4MujocoRunner, Px4RunnerConfig


pytestmark = pytest.mark.integration


def test_real_px4_environment_preflight():
    report = inspect_px4_environment(mode="local")
    if report.issues:
        pytest.skip("real PX4 unavailable: " + "; ".join(report.issues))
    if os.environ.get("RUN_PX4_INTEGRATION") != "1":
        pytest.skip("real PX4 available but RUN_PX4_INTEGRATION=1 was not requested")
    assert report.compatible


def test_real_px4_owned_handshake_and_disarmed_sensor_loop(tmp_path):
    if os.environ.get("RUN_PX4_INTEGRATION") != "1":
        pytest.skip("RUN_PX4_INTEGRATION=1 was not requested")
    autopilot_dir = os.environ.get("PX4_AUTOPILOT_DIR")
    simulator_host = os.environ.get("PX4_SIM_HOSTNAME")
    if not autopilot_dir or not simulator_host:
        pytest.skip(
            "PX4_AUTOPILOT_DIR and PX4_SIM_HOSTNAME are required for owned launch"
        )
    config = Px4RunnerConfig(
        mode=Px4EnvironmentMode.LOCAL,
        bind_host=os.environ.get("PX4_BIND_HOST", "127.0.0.1"),
        bind_port=int(os.environ.get("PX4_BIND_PORT", "4560")),
        launch_owned_px4=True,
        autopilot_dir=Path(autopilot_dir),
        simulator_host=simulator_host,
        duration_s=0.05,
        bootstrap_wall_interval_s=0.001,
        log_dir=tmp_path,
    )
    metrics = Px4MujocoRunner.create_default(config).run()
    assert metrics.outcome == "completed"
    assert metrics.physics_steps == 10
    assert metrics.peer_system_id == 1
    assert metrics.protocol_version == "2.0"
    assert metrics.timeout_count == 0
    assert metrics.disconnect_count == 0
