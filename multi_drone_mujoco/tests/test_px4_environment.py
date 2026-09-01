"""Gate P7 offline environment and process-ownership tests."""

import io
from pathlib import Path

import pytest

from multi_drone_mujoco.backends import px4_environment
from multi_drone_mujoco.backends.px4_environment import (
    OwnedProcessConfig,
    PX4_COMPATIBLE_COMMIT,
    PX4_COMPATIBLE_TAG,
    Px4EnvironmentMode,
    Px4LaunchError,
    Px4OwnedProcessLauncher,
    inspect_px4_environment,
    px4_owned_launch_config,
    read_px4_parameter_file,
    px4_source_launch_command,
)


def test_offline_environment_has_no_px4_or_pymavlink_requirements(monkeypatch):
    monkeypatch.setattr(px4_environment.importlib.util, "find_spec", lambda name: None)
    report = inspect_px4_environment({})
    assert report.mode is Px4EnvironmentMode.OFFLINE
    assert report.autopilot_dir is None
    assert not report.compatible
    assert report.ready


def test_remote_mode_does_not_require_local_checkout_binary_or_launcher(monkeypatch):
    monkeypatch.setattr(px4_environment.importlib.util, "find_spec", lambda name: object())
    report = inspect_px4_environment(
        {
            "PX4_BIND_HOST": "0.0.0.0",
            "PX4_BIND_PORT": "4560",
            "PX4_SIM_HOSTNAME": "192.0.2.10",
        },
        mode="remote",
    )
    assert report.mode is Px4EnvironmentMode.REMOTE
    assert report.autopilot_dir is report.binary is None
    assert report.bind_host == "0.0.0.0"
    assert report.bind_port == 4560
    assert report.remote_simulator_host == "192.0.2.10"
    assert report.ready


def test_local_empty_environment_reports_local_and_listener_inputs():
    report = inspect_px4_environment({}, mode="local")
    assert "PX4_AUTOPILOT_DIR is not set for local mode" in report.issues
    assert "PX4_BINARY is not set" in "\n".join(report.issues)
    assert "PX4_BIND_HOST is not set" in report.issues
    assert "PX4_BIND_PORT is not set" in report.issues
    assert "PX4_SIM_HOSTNAME is not set for the local PX4 process" in report.issues


def test_verified_checkout_and_command_generation_are_read_only(tmp_path, monkeypatch):
    binary = tmp_path / "build" / "px4_sitl_default" / "bin" / "px4"
    binary.parent.mkdir(parents=True)
    binary.touch()

    def fake_git(directory, *args):
        if args[0] == "describe":
            return PX4_COMPATIBLE_TAG
        return PX4_COMPATIBLE_COMMIT

    monkeypatch.setattr(px4_environment, "_git_value", fake_git)
    monkeypatch.setattr(px4_environment.importlib.util, "find_spec", lambda name: object())
    report = inspect_px4_environment(
        {
            "PX4_AUTOPILOT_DIR": str(tmp_path),
            "PX4_BIND_HOST": "127.0.0.1",
            "PX4_BIND_PORT": "4560",
            "PX4_SIM_HOSTNAME": "127.0.0.1",
        },
        mode="local",
    )
    assert report.compatible and not report.issues
    assert report.binary == binary
    assert px4_source_launch_command(tmp_path) == (
        "make", "-C", str(tmp_path), "px4_sitl", "none_iris"
    )


def test_launcher_only_stops_the_process_it_started(tmp_path):
    class FakeProcess:
        def __init__(self):
            self.running = True
            self.terminated = 0

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated += 1
            self.running = False

        def wait(self, timeout):
            return 0

    process = FakeProcess()
    calls = []

    def factory(command, **kwargs):
        calls.append((command, kwargs["cwd"], kwargs["env"]["PX4_SIM_HOSTNAME"]))
        return process

    config = OwnedProcessConfig(
        ("make", "px4_sitl", "none_iris"),
        Path(tmp_path),
        {"PX4_SIM_HOSTNAME": "192.0.2.10"},
        1.0,
    )
    launcher = Px4OwnedProcessLauncher(config, factory)
    launcher.close()  # An external/unowned process is never searched for or killed.
    launcher.start()
    assert launcher.owns_running_process
    with pytest.raises(RuntimeError, match="already running"):
        launcher.start()
    launcher.close()
    launcher.close()
    assert process.terminated == 1
    assert calls[0][:2] == (config.command, str(tmp_path))


def test_fixed_local_config_is_direct_external_simulator_launch(tmp_path):
    binary = tmp_path / "build" / "px4_sitl_default" / "bin" / "px4"
    rootfs = tmp_path / "build" / "px4_sitl_default" / "rootfs"
    binary.parent.mkdir(parents=True)
    rootfs.mkdir(parents=True)
    binary.touch(mode=0o755)
    config = px4_owned_launch_config(tmp_path, "127.0.0.1", instance=2)
    assert config.command == (str(binary), "-i", "2")
    assert config.cwd == rootfs
    assert config.environment == {
        "PX4_SYS_AUTOSTART": "10016",
        "PX4_SIM_HOSTNAME": "127.0.0.1",
    }
    assert "gazebo" not in " ".join(config.command).lower()
    assert "jmavsim" not in " ".join(config.command).lower()


def test_parameter_file_is_read_only_and_becomes_scoped_environment(tmp_path):
    parameters = tmp_path / "vehicle.params"
    parameters.write_text("# derived\nMPC_THR_HOVER 0.666666667\nCA_ROTOR_COUNT 4\n")
    values = read_px4_parameter_file(parameters)
    binary = tmp_path / "build" / "px4_sitl_default" / "bin" / "px4"
    rootfs = tmp_path / "build" / "px4_sitl_default" / "rootfs"
    binary.parent.mkdir(parents=True)
    rootfs.mkdir(parents=True)
    binary.touch(mode=0o755)
    config = px4_owned_launch_config(
        tmp_path, "127.0.0.1", parameter_overrides=values
    )
    assert config.environment["PX4_PARAM_MPC_THR_HOVER"] == "0.666666667"
    assert config.environment["PX4_PARAM_CA_ROTOR_COUNT"] == "4"


def test_launcher_rejects_missing_binary_and_wrong_checkout(tmp_path, monkeypatch):
    missing = OwnedProcessConfig(
        (str(tmp_path / "missing-px4"),), tmp_path, {}, 0.1
    )
    with pytest.raises(Px4LaunchError, match="does not exist"):
        Px4OwnedProcessLauncher(missing).start()

    binary = tmp_path / "px4"
    binary.touch(mode=0o755)
    wrong = OwnedProcessConfig(
        (str(binary),), tmp_path, {}, 0.1,
        autopilot_dir=tmp_path, expected_commit=PX4_COMPATIBLE_COMMIT,
    )
    monkeypatch.setattr(px4_environment, "_git_value", lambda *args: "wrong")
    with pytest.raises(Px4LaunchError, match="checkout mismatch"):
        Px4OwnedProcessLauncher(wrong).start()


def test_launcher_reports_early_exit_and_bounds_logs(tmp_path):
    class ExitedProcess:
        stdout = io.BytesIO(b"x" * 200 + b"\ncritical startup failure\n")

        def poll(self):
            return 23

        def wait(self, timeout):
            return 23

    config = OwnedProcessConfig(
        ("px4",), tmp_path, {}, 0.1,
        startup_timeout_s=0.1, startup_pattern="ready", max_log_bytes=64,
    )
    launcher = Px4OwnedProcessLauncher(config, lambda *args, **kwargs: ExitedProcess())
    with pytest.raises(Px4LaunchError) as error:
        launcher.start()
    message = str(error.value)
    assert "exit code 23" in message
    assert "command: px4" in message
    assert "critical startup failure" in message
    assert len(launcher.log_text.encode()) <= 64


def test_launcher_timeout_terminates_only_owned_process(tmp_path):
    class WaitingProcess:
        stdout = None

        def __init__(self):
            self.running = True
            self.terminated = 0

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.terminated += 1
            self.running = False

        def kill(self):
            self.running = False

        def wait(self, timeout):
            return 0

    process = WaitingProcess()
    config = OwnedProcessConfig(
        ("px4",), tmp_path, {}, 0.1,
        startup_timeout_s=0.02, startup_pattern="never",
    )
    launcher = Px4OwnedProcessLauncher(config, lambda *args, **kwargs: process)
    with pytest.raises(Px4LaunchError, match="startup timed out"):
        launcher.start()
    assert process.terminated == 1
    assert not launcher.owns_running_process
