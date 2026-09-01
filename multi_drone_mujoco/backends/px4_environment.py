"""Read-only PX4 environment inspection and explicitly owned local launcher."""

from dataclasses import dataclass
from collections import deque
from enum import Enum
import importlib.util
import math
import os
from pathlib import Path
import pty
import shlex
import signal
import subprocess
import threading
import time
from typing import Callable, Deque, Dict, Mapping, Optional, Tuple


PX4_COMPATIBLE_TAG = "v1.17.0"
PX4_COMPATIBLE_COMMIT = "d6f12ad1c4f70ad3230afd7d86e971421e02fef4"


class Px4EnvironmentMode(str, Enum):
    """Deployment modes with intentionally different local requirements."""

    OFFLINE = "offline"
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class Px4EnvironmentReport:
    mode: Px4EnvironmentMode
    autopilot_dir: Optional[Path]
    binary: Optional[Path]
    bind_host: Optional[str]
    bind_port: Optional[int]
    remote_simulator_host: Optional[str]
    pymavlink_available: bool
    git_tag: Optional[str]
    git_commit: Optional[str]
    compatible: bool
    issues: Tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.issues


def _git_value(directory: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ("git", "-C", str(directory)) + args,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def inspect_px4_environment(
    environ: Optional[Mapping[str, str]] = None,
    mode: Optional[str] = None,
) -> Px4EnvironmentReport:
    """Inspect one deployment mode without opening sockets or writing.

    Offline mode has no PX4 requirements. Remote mode validates only the local
    simulator listener and Python protocol dependency. Only local mode inspects
    a PX4 source checkout and executable.
    """
    values = os.environ if environ is None else environ
    mode_value = mode if mode is not None else values.get("PX4_MODE", "offline")
    try:
        selected_mode = Px4EnvironmentMode(str(mode_value).lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in Px4EnvironmentMode)
        raise ValueError(f"mode must be one of: {choices}") from exc

    issues = []
    autopilot_dir = None
    binary = None
    if selected_mode is Px4EnvironmentMode.LOCAL:
        directory_value = values.get("PX4_AUTOPILOT_DIR")
        autopilot_dir = Path(directory_value).expanduser() if directory_value else None
        if autopilot_dir is None:
            issues.append("PX4_AUTOPILOT_DIR is not set for local mode")
        elif not autopilot_dir.is_dir():
            issues.append(f"PX4_AUTOPILOT_DIR is not a directory: {autopilot_dir}")

        binary_value = values.get("PX4_BINARY")
        binary = Path(binary_value).expanduser() if binary_value else None
        if binary is None and autopilot_dir is not None:
            candidate = autopilot_dir / "build" / "px4_sitl_default" / "bin" / "px4"
            if candidate.is_file():
                binary = candidate
        if binary is None:
            issues.append(
                "PX4_BINARY is not set and no standard SITL binary was found "
                "for local mode"
            )
        elif not binary.is_file():
            issues.append(f"PX4_BINARY is not a file: {binary}")

    bind_host = values.get("PX4_BIND_HOST")
    bind_port = None
    if selected_mode is not Px4EnvironmentMode.OFFLINE:
        if bind_host is None or not bind_host.strip():
            bind_host = None
            issues.append("PX4_BIND_HOST is not set")
        port_value = values.get("PX4_BIND_PORT")
        if port_value is None:
            issues.append("PX4_BIND_PORT is not set")
        else:
            try:
                bind_port = int(port_value)
            except ValueError:
                issues.append("PX4_BIND_PORT is not an integer")
            else:
                if not 1 <= bind_port <= 65535:
                    issues.append("PX4_BIND_PORT is outside [1, 65535]")
                    bind_port = None

    remote_simulator_host = values.get("PX4_SIM_HOSTNAME")
    if remote_simulator_host is not None and not remote_simulator_host.strip():
        remote_simulator_host = None
    if (
        selected_mode is Px4EnvironmentMode.LOCAL
        and remote_simulator_host is None
    ):
        issues.append("PX4_SIM_HOSTNAME is not set for the local PX4 process")

    git_tag = None
    git_commit = None
    if selected_mode is Px4EnvironmentMode.LOCAL and autopilot_dir is not None and autopilot_dir.is_dir():
        git_tag = _git_value(autopilot_dir, "describe", "--tags", "--exact-match", "HEAD")
        git_commit = _git_value(autopilot_dir, "rev-parse", "HEAD")
    compatible = (
        git_tag == PX4_COMPATIBLE_TAG and git_commit == PX4_COMPATIBLE_COMMIT
    )
    if selected_mode is Px4EnvironmentMode.LOCAL and autopilot_dir is not None and autopilot_dir.is_dir() and not compatible:
        issues.append(
            f"PX4 checkout is not verified {PX4_COMPATIBLE_TAG} "
            f"({PX4_COMPATIBLE_COMMIT})"
        )
    pymavlink_available = importlib.util.find_spec("pymavlink") is not None
    if selected_mode is not Px4EnvironmentMode.OFFLINE and not pymavlink_available:
        issues.append("optional Python dependency pymavlink is not installed")
    return Px4EnvironmentReport(
        mode=selected_mode,
        autopilot_dir=autopilot_dir,
        binary=binary,
        bind_host=bind_host,
        bind_port=bind_port,
        remote_simulator_host=remote_simulator_host,
        pymavlink_available=pymavlink_available,
        git_tag=git_tag,
        git_commit=git_commit,
        compatible=compatible,
        issues=tuple(issues),
    )


def px4_source_launch_command(autopilot_dir: Path) -> Tuple[str, ...]:
    """Return, but do not execute, the official custom-simulator SITL target."""
    directory = Path(autopilot_dir)
    if not directory.is_dir():
        raise ValueError(f"autopilot_dir is not a directory: {directory}")
    return ("make", "-C", str(directory), "px4_sitl", "none_iris")


@dataclass(frozen=True)
class OwnedProcessConfig:
    """Explicit command/cwd for a process that this launcher will own."""

    command: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    shutdown_timeout_s: float
    startup_timeout_s: float = 5.0
    startup_pattern: Optional[str] = None
    max_log_bytes: int = 64 * 1024
    autopilot_dir: Optional[Path] = None
    expected_commit: Optional[str] = None
    use_pty: bool = False

    def __post_init__(self) -> None:
        command = tuple(self.command)
        if not command or any(not isinstance(value, str) or not value for value in command):
            raise ValueError("command must contain non-empty strings")
        object.__setattr__(self, "command", command)
        cwd = Path(self.cwd)
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")
        object.__setattr__(self, "cwd", cwd)
        for name in ("shutdown_timeout_s", "startup_timeout_s"):
            timeout = float(getattr(self, name))
            if not math.isfinite(timeout) or timeout <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, timeout)
        if self.startup_pattern is not None and not self.startup_pattern:
            raise ValueError("startup_pattern must be non-empty when provided")
        if (
            isinstance(self.max_log_bytes, bool)
            or not isinstance(self.max_log_bytes, int)
            or self.max_log_bytes <= 0
        ):
            raise ValueError("max_log_bytes must be a positive integer")
        if self.autopilot_dir is not None:
            object.__setattr__(self, "autopilot_dir", Path(self.autopilot_dir))
        if self.expected_commit is not None and self.autopilot_dir is None:
            raise ValueError("expected_commit requires autopilot_dir")
        if not isinstance(self.use_pty, bool):
            raise ValueError("use_pty must be bool")


class Px4LaunchError(RuntimeError):
    """An owned PX4 process could not reach its declared startup state."""


def px4_owned_launch_config(
    autopilot_dir: Path,
    simulator_host: str,
    *,
    instance: int = 0,
    startup_timeout_s: float = 10.0,
    shutdown_timeout_s: float = 5.0,
    max_log_bytes: int = 64 * 1024,
    parameter_overrides: Optional[Mapping[str, str]] = None,
) -> OwnedProcessConfig:
    """Build a fixed-version, external-simulator launch configuration.

    The simulator TCP port is PX4's conventional ``4560 + instance``.  This
    helper starts neither Gazebo nor jMAVSim and does not modify PX4 source.
    """
    directory = Path(autopilot_dir)
    binary = directory / "build" / "px4_sitl_default" / "bin" / "px4"
    rootfs = directory / "build" / "px4_sitl_default" / "rootfs"
    if not isinstance(simulator_host, str) or not simulator_host.strip():
        raise ValueError("simulator_host must be a non-empty string")
    if isinstance(instance, bool) or not isinstance(instance, int) or instance < 0:
        raise ValueError("instance must be a non-negative integer")
    environment = {
        "PX4_SYS_AUTOSTART": "10016",
        "PX4_SIM_HOSTNAME": simulator_host,
    }
    for name, value in (parameter_overrides or {}).items():
        if not name or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in name):
            raise ValueError(f"invalid PX4 parameter name: {name!r}")
        environment[f"PX4_PARAM_{name}"] = str(value)
    return OwnedProcessConfig(
        command=(str(binary), "-i", str(instance)),
        cwd=rootfs,
        environment=environment,
        shutdown_timeout_s=shutdown_timeout_s,
        startup_timeout_s=startup_timeout_s,
        startup_pattern="simulator_mavlink",
        max_log_bytes=max_log_bytes,
        autopilot_dir=directory,
        expected_commit=PX4_COMPATIBLE_COMMIT,
        use_pty=True,
    )


def read_px4_parameter_file(path: Path) -> Dict[str, str]:
    """Read the repository's simple ``NAME VALUE`` parameter format."""
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid PX4 parameter line {line_number}: {raw_line}")
        name, value = fields
        if name in values:
            raise ValueError(f"duplicate PX4 parameter {name} on line {line_number}")
        if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in name):
            raise ValueError(f"invalid PX4 parameter name on line {line_number}: {name}")
        values[name] = value
    return values


class Px4OwnedProcessLauncher:
    """Optional launcher that only terminates the exact process it created."""

    def __init__(
        self,
        config: OwnedProcessConfig,
        popen_factory: Callable = subprocess.Popen,
    ):
        self.config = config
        self._popen_factory = popen_factory
        self._process = None
        self._owns_process_group = False
        self._reader_thread = None
        self._reader_stream = None
        self._log_chunks: Deque[bytes] = deque()
        self._log_size = 0
        self._log_lock = threading.Lock()

    @property
    def owns_running_process(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def log_text(self) -> str:
        with self._log_lock:
            value = b"".join(self._log_chunks)
        return value.decode("utf-8", errors="replace")

    @property
    def command_display(self) -> str:
        return shlex.join(self.config.command)

    def start(self) -> None:
        if self.owns_running_process:
            raise RuntimeError("owned PX4 process is already running")
        self._validate_target()
        with self._log_lock:
            self._log_chunks.clear()
            self._log_size = 0
        environment = os.environ.copy()
        environment.update(self.config.environment)
        kwargs = {
            "cwd": str(self.config.cwd),
            "env": environment,
            "start_new_session": True,
        }
        master_fd = None
        slave_fd = None
        if self.config.use_pty:
            master_fd, slave_fd = pty.openpty()
            kwargs.update({
                "stdin": subprocess.DEVNULL,
                "stdout": slave_fd,
                "stderr": slave_fd,
            })
        else:
            kwargs.update({
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            })
        try:
            self._process = self._popen_factory(self.config.command, **kwargs)
        except (OSError, subprocess.SubprocessError) as exc:
            for descriptor in (master_fd, slave_fd):
                if descriptor is not None:
                    os.close(descriptor)
            raise Px4LaunchError(
                f"PX4 launch failed: {self.command_display}: {exc}"
            ) from exc
        if slave_fd is not None:
            os.close(slave_fd)
        self._owns_process_group = self._popen_factory is subprocess.Popen
        if master_fd is not None:
            stdout = os.fdopen(master_fd, "rb", buffering=0)
        else:
            stdout = getattr(self._process, "stdout", None)
        self._reader_stream = stdout
        if stdout is not None:
            self._reader_thread = threading.Thread(
                target=self._read_output,
                args=(stdout,),
                name="px4-owned-log-reader",
                daemon=True,
            )
            self._reader_thread.start()
        if self.config.startup_pattern is not None:
            self._wait_for_startup()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            self._signal_owned_process(process, signal.SIGTERM)
            try:
                process.wait(timeout=self.config.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                self._signal_owned_process(process, signal.SIGKILL)
                process.wait(timeout=self.config.shutdown_timeout_s)
        stdout = self._reader_stream
        self._reader_stream = None
        if stdout is not None:
            try:
                stdout.close()
            except OSError:
                pass
        reader = self._reader_thread
        self._reader_thread = None
        if reader is not None:
            reader.join(timeout=self.config.shutdown_timeout_s)
        self._owns_process_group = False

    def _validate_target(self) -> None:
        executable = Path(self.config.command[0])
        if executable.is_absolute():
            if not executable.is_file():
                raise Px4LaunchError(f"PX4 binary does not exist: {executable}")
            if not os.access(executable, os.X_OK):
                raise Px4LaunchError(f"PX4 binary is not executable: {executable}")
        if self.config.expected_commit is not None:
            commit = _git_value(self.config.autopilot_dir, "rev-parse", "HEAD")
            if commit != self.config.expected_commit:
                raise Px4LaunchError(
                    "PX4 checkout mismatch: expected "
                    f"{self.config.expected_commit}, got {commit or '<unknown>'}"
                )

    def _wait_for_startup(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        pattern = self.config.startup_pattern
        while time.monotonic() < deadline:
            if pattern in self.log_text:
                return
            return_code = self._process.poll()
            if return_code is not None:
                message = self._failure_message(
                    f"exited before startup (exit code {return_code})"
                )
                self.close()
                raise Px4LaunchError(message)
            time.sleep(0.01)
        message = self._failure_message(
            f"startup timed out after {self.config.startup_timeout_s:g} s"
        )
        self.close()
        raise Px4LaunchError(message)

    def _failure_message(self, reason: str) -> str:
        logs = self.log_text.strip() or "<no output>"
        return f"PX4 {reason}\ncommand: {self.command_display}\nlog:\n{logs}"

    def _read_output(self, stream) -> None:
        while True:
            try:
                chunk = stream.read(4096)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode()
            self._append_log(chunk)

    def _append_log(self, chunk: bytes) -> None:
        limit = self.config.max_log_bytes
        if len(chunk) >= limit:
            chunk = chunk[-limit:]
        with self._log_lock:
            self._log_chunks.append(chunk)
            self._log_size += len(chunk)
            while self._log_size > limit and self._log_chunks:
                excess = self._log_size - limit
                first = self._log_chunks[0]
                if len(first) <= excess:
                    self._log_chunks.popleft()
                    self._log_size -= len(first)
                else:
                    self._log_chunks[0] = first[excess:]
                    self._log_size -= excess

    def _signal_owned_process(self, process, sig: signal.Signals) -> None:
        if self._owns_process_group:
            try:
                os.killpg(process.pid, sig)
                return
            except ProcessLookupError:
                return
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
