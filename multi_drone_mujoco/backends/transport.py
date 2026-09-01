"""Replaceable PX4 simulator transports with a deterministic in-memory fake."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Deque, List, Optional, Protocol, runtime_checkable

from multi_drone_mujoco.backends.contracts import NormalizedMotorCommand
from multi_drone_mujoco.backends.sensors import SensorFrame


class TransportError(RuntimeError):
    """Base simulator transport error."""


class TransportTimeout(TransportError):
    """No actuator packet arrived within the bounded wall-clock timeout."""


class TransportDisconnected(TransportError):
    """Transport operation attempted without a live connection."""


def _mavlink_int16_milli(value: float) -> int:
    """Scale SI acceleration to MAVLink milligravity field without overflow."""
    return max(-32768, min(32767, int(round(float(value) * 1000.0))))


class TransportState(Enum):
    DISCONNECTED = "disconnected"
    LISTENING = "listening"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True)
class TransportHealth:
    state: TransportState
    detail: str = ""
    sent_packets: int = 0
    received_packets: int = 0
    peer_system_id: Optional[int] = None
    peer_component_id: Optional[int] = None
    protocol_version: Optional[str] = None
    handshake_duration_s: Optional[float] = None
    timeout_count: int = 0
    disconnect_count: int = 0


@runtime_checkable
class SimulatorTransport(Protocol):
    """Bounded transport contract used by the lockstep coordinator."""

    def connect(self) -> None:
        pass

    def send_sensor_frame(self, frame: SensorFrame) -> None:
        pass

    def receive_actuator(self, timeout_s: float) -> NormalizedMotorCommand:
        pass

    def poll_actuator(self) -> Optional[NormalizedMotorCommand]:
        pass

    @property
    def health(self) -> TransportHealth:
        pass

    def close(self) -> None:
        pass


class FakeTransport:
    """Deterministic no-network transport for unit and integration tests."""

    def __init__(self, actuator_packets=()):
        self.sent_sensor_frames: List[SensorFrame] = []
        self._actuator_packets: Deque[NormalizedMotorCommand] = deque(actuator_packets)
        self._state = TransportState.DISCONNECTED
        self._detail = ""
        self._received = 0
        self._timeout_count = 0

    @property
    def health(self) -> TransportHealth:
        return TransportHealth(
            self._state,
            self._detail,
            len(self.sent_sensor_frames),
            self._received,
            timeout_count=self._timeout_count,
        )

    def connect(self) -> None:
        if self._state == TransportState.CLOSED:
            raise TransportDisconnected("transport is closed")
        self._state = TransportState.CONNECTED
        self._detail = ""

    def disconnect(self, detail: str = "fake peer disconnected") -> None:
        if self._state != TransportState.CLOSED:
            self._state = TransportState.DISCONNECTED
            self._detail = detail

    def queue_actuator(self, command: NormalizedMotorCommand) -> None:
        self._actuator_packets.append(command)

    def send_sensor_frame(self, frame: SensorFrame) -> None:
        self._require_connected()
        self.sent_sensor_frames.append(frame)

    def receive_actuator(self, timeout_s: float) -> NormalizedMotorCommand:
        self._require_connected()
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        if not self._actuator_packets:
            self._timeout_count += 1
            raise TransportTimeout(
                f"no fake actuator packet available within {timeout_s:g} s"
            )
        self._received += 1
        return self._actuator_packets.popleft()

    def poll_actuator(self) -> Optional[NormalizedMotorCommand]:
        self._require_connected()
        if not self._actuator_packets:
            return None
        self._received += 1
        return self._actuator_packets.popleft()

    def close(self) -> None:
        self._state = TransportState.CLOSED
        self._detail = ""
        self._actuator_packets.clear()

    def _require_connected(self) -> None:
        if self._state != TransportState.CONNECTED:
            raise TransportDisconnected(f"transport is {self._state.value}")


@dataclass(frozen=True)
class MavlinkTransportConfig:
    """No-default network configuration for a PX4 simulator endpoint."""

    bind_host: str
    bind_port: int
    heartbeat_timeout_s: float
    source_system: int
    source_component: int
    max_reconnect_attempts: int
    tcp_role: str = "listen"

    def __post_init__(self) -> None:
        if not isinstance(self.bind_host, str) or not self.bind_host.strip():
            raise ValueError("bind_host must be a non-empty string")
        if isinstance(self.bind_port, bool) or not isinstance(self.bind_port, int) or not 1 <= self.bind_port <= 65535:
            raise ValueError("bind_port must be in [1, 65535]")
        timeout = float(self.heartbeat_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("heartbeat_timeout_s must be finite and positive")
        object.__setattr__(self, "heartbeat_timeout_s", timeout)
        for name in ("source_system", "source_component"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
                raise ValueError(f"{name} must be in [1, 255]")
        if (
            isinstance(self.max_reconnect_attempts, bool)
            or not isinstance(self.max_reconnect_attempts, int)
            or self.max_reconnect_attempts < 0
        ):
            raise ValueError("max_reconnect_attempts must be a non-negative integer")
        if self.tcp_role not in ("listen", "connect"):
            raise ValueError("tcp_role must be 'listen' or 'connect'")


class SimulatorMavlinkTransport:
    """TCP MAVLink simulator transport; imports ``pymavlink`` only on connect."""

    def __init__(self, config: MavlinkTransportConfig):
        self.config = config
        self._connection = None
        self._mavutil = None
        self._state = TransportState.DISCONNECTED
        self._detail = ""
        self._sent = 0
        self._received = 0
        self._peer_system_id = None
        self._peer_component_id = None
        self._protocol_version = None
        self._handshake_duration_s = None
        self._handshake_started_s = None
        self._timeout_count = 0
        self._disconnect_count = 0
        self._sensor_reset_pending = True

    @property
    def health(self) -> TransportHealth:
        return TransportHealth(
            self._state,
            self._detail,
            self._sent,
            self._received,
            self._peer_system_id,
            self._peer_component_id,
            self._protocol_version,
            self._handshake_duration_s,
            self._timeout_count,
            self._disconnect_count,
        )

    @property
    def endpoint(self) -> str:
        """pymavlink endpoint; standard PX4 SITL uses the ``tcpin`` role."""
        scheme = "tcpin" if self.config.tcp_role == "listen" else "tcp"
        return f"{scheme}:{self.config.bind_host}:{self.config.bind_port}"

    def connect(self) -> None:
        if self._state == TransportState.CLOSED:
            raise TransportDisconnected("transport is closed")
        if self.config.tcp_role == "listen":
            self.open_listener()
            self.accept_peer()
            return
        self._state = TransportState.CONNECTING
        self._handshake_started_s = time.monotonic()
        self._load_mavlink()
        try:
            self._connection = self._open_connection()
            self._finish_handshake()
        except Exception as exc:
            self._connect_error(exc)

    def open_listener(self) -> None:
        """Bind the standard simulator TCP listener without waiting for PX4."""
        if self.config.tcp_role != "listen":
            raise TransportError("open_listener requires tcp_role='listen'")
        if self._state == TransportState.CLOSED:
            raise TransportDisconnected("transport is closed")
        if self._state == TransportState.LISTENING:
            return
        if self._state == TransportState.CONNECTED:
            raise TransportError("transport is already connected")
        self._state = TransportState.CONNECTING
        self._handshake_started_s = time.monotonic()
        self._load_mavlink()
        try:
            self._connection = self._open_connection()
        except Exception as exc:
            self._connect_error(exc)
        self._state = TransportState.LISTENING
        self._sensor_reset_pending = True
        self._detail = (
            f"waiting for PX4 TCP peer on {self.config.bind_host}:"
            f"{self.config.bind_port}"
        )

    def accept_peer(self, timeout_s: Optional[float] = None) -> None:
        """Accept PX4 after :meth:`open_listener` using a bounded wait."""
        if self._state != TransportState.LISTENING or self._connection is None:
            raise TransportDisconnected(
                f"transport is {self._state.value}; listener is not open"
            )
        timeout = (
            self.config.heartbeat_timeout_s if timeout_s is None else float(timeout_s)
        )
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout
        try:
            while getattr(self._connection, "port", None) is None:
                self._connection.recv()
                if time.monotonic() >= deadline:
                    self._timeout_count += 1
                    raise TransportTimeout(
                        "PX4 TCP simulator peer handshake timed out"
                    )
                time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
            self._finish_handshake()
        except Exception as exc:
            self._connect_error(exc)

    def _load_mavlink(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            self._state = TransportState.ERROR
            self._detail = "optional dependency pymavlink is not installed"
            raise RuntimeError(
                "PX4 MAVLink transport requires optional dependency 'pymavlink'; "
                "install the project's px4 extra"
            ) from exc
        self._mavutil = mavutil

    def _open_connection(self):
        return self._mavutil.mavlink_connection(
            self.endpoint,
            source_system=self.config.source_system,
            source_component=self.config.source_component,
            autoreconnect=False,
            retries=1,
        )

    def _finish_handshake(self) -> None:
        self._connection.mav.heartbeat_send(
            self._mavutil.mavlink.MAV_TYPE_GCS,
            self._mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            self._mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        self._protocol_version = str(
            getattr(self._connection, "WIRE_PROTOCOL_VERSION", "unknown")
        )
        if self._handshake_started_s is not None:
            self._handshake_duration_s = time.monotonic() - self._handshake_started_s
        self._state = TransportState.CONNECTED
        self._detail = ""

    def _connect_error(self, exc: Exception) -> None:
        self._state = TransportState.ERROR
        self._detail = str(exc)
        self._close_connection()
        if isinstance(exc, TransportError):
            raise exc
        raise TransportError(f"PX4 MAVLink connect failed: {exc}") from exc

    def reconnect(self) -> None:
        self._close_connection()
        last_error = None
        for _ in range(self.config.max_reconnect_attempts + 1):
            self._state = TransportState.DISCONNECTED
            try:
                self.connect()
                return
            except TransportError as exc:
                last_error = exc
        raise TransportDisconnected(
            f"PX4 reconnect exhausted: {last_error}"
        ) from last_error

    def send_sensor_frame(self, frame: SensorFrame) -> None:
        self._require_connected()
        mav = self._connection.mav
        try:
            if frame.imu_updated or frame.barometer_updated or frame.magnetometer_updated:
                xacc, yacc, zacc = frame.specific_force_frd_m_s2
                xgyro, ygyro, zgyro = frame.angular_velocity_frd_rad_s
                xmag, ymag, zmag = frame.magnetic_field_frd_gauss
                pressure_alt = frame.barometric_altitude_m
                pressure_hpa = 1013.25 * max(0.0, 1.0 - pressure_alt / 44330.0) ** 5.255
                fields_updated = 0
                if frame.imu_updated:
                    fields_updated |= 0x003F
                if frame.magnetometer_updated:
                    fields_updated |= 0x01C0
                if frame.barometer_updated:
                    fields_updated |= 0x1A00
                if self._sensor_reset_pending:
                    fields_updated |= 1 << 31
                mav.hil_sensor_send(
                    frame.timestamp_us, xacc, yacc, zacc, xgyro, ygyro, zgyro,
                    xmag, ymag, zmag, pressure_hpa, 0.0, pressure_alt, 20.0,
                    fields_updated,
                )
                self._sensor_reset_pending = False
            if frame.gps_updated:
                vn, ve, vd = frame.velocity_ned_m_s
                horizontal_speed = math.hypot(vn, ve)
                course = math.degrees(math.atan2(ve, vn)) % 360.0
                mav.hil_gps_send(
                    frame.timestamp_us, 3,
                    int(round(frame.gps_latitude_deg * 1e7)),
                    int(round(frame.gps_longitude_deg * 1e7)),
                    int(round(frame.gps_altitude_m * 1000.0)),
                    100, 100, int(round(horizontal_speed * 100.0)),
                    int(round(vn * 100.0)), int(round(ve * 100.0)),
                    int(round(vd * 100.0)), int(round(course * 100.0)), 10,
                )
            q = frame.quaternion_ned_frd_wxyz
            rollspeed, pitchspeed, yawspeed = frame.angular_velocity_frd_rad_s
            lat = int(round(frame.gps_latitude_deg * 1e7))
            lon = int(round(frame.gps_longitude_deg * 1e7))
            alt = int(round(frame.gps_altitude_m * 1000.0))
            vx, vy, vz = (int(round(value * 100.0)) for value in frame.velocity_ned_m_s)
            xacc, yacc, zacc = (
                _mavlink_int16_milli(value)
                for value in frame.linear_acceleration_ned_m_s2
            )
            mav.hil_state_quaternion_send(
                frame.timestamp_us, q, rollspeed, pitchspeed, yawspeed,
                lat, lon, alt, vx, vy, vz, 0, 0, xacc, yacc, zacc,
            )
        except Exception as exc:
            self._state = TransportState.ERROR
            self._detail = str(exc)
            self._disconnect_count += 1
            raise TransportError(f"MAVLink sensor send failed: {exc}") from exc
        self._sent += 1

    def receive_actuator(self, timeout_s: float) -> NormalizedMotorCommand:
        self._require_connected()
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        message = self._connection.recv_match(
            type="HIL_ACTUATOR_CONTROLS", blocking=True, timeout=timeout_s
        )
        if message is None:
            if (
                self.config.tcp_role == "listen"
                and getattr(self._connection, "port", None) is None
            ):
                self._state = TransportState.DISCONNECTED
                self._detail = "PX4 TCP peer disconnected"
                self._disconnect_count += 1
                raise TransportDisconnected(self._detail)
            self._timeout_count += 1
            raise TransportTimeout(
                f"PX4 actuator receive timed out after {timeout_s:g} s"
            )
        return self._decode_actuator(message)

    def poll_actuator(self) -> Optional[NormalizedMotorCommand]:
        """Return an actuator packet if already available, without a timeout."""
        self._require_connected()
        message = self._connection.recv_match(
            type="HIL_ACTUATOR_CONTROLS", blocking=False
        )
        if message is None:
            return None
        return self._decode_actuator(message)

    def _decode_actuator(self, message) -> NormalizedMotorCommand:
        controls = tuple(float(value) for value in message.controls[:4])
        armed_flag = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        sequence = int(message.get_seq())
        self._peer_system_id = int(message.get_srcSystem())
        self._peer_component_id = int(message.get_srcComponent())
        self._protocol_version = str(
            getattr(self._connection, "WIRE_PROTOCOL_VERSION", self._protocol_version)
        )
        self._received += 1
        return NormalizedMotorCommand(
            controls,
            bool(int(message.mode) & armed_flag),
            int(message.time_usec),
            sequence,
        )

    def close(self) -> None:
        self._close_connection()
        self._state = TransportState.CLOSED
        self._detail = ""

    def _close_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            close = getattr(connection, "close", None)
            if close is not None:
                close()

    def _require_connected(self) -> None:
        if self._state != TransportState.CONNECTED or self._connection is None:
            raise TransportDisconnected(f"transport is {self._state.value}")
