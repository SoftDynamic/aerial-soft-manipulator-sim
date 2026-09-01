"""Optional MAVLink GCS channel for PX4 flight commands and telemetry."""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Px4FlightControlConfig:
    bind_host: str
    bind_port: int
    target_system: int = 1
    target_component: int = 1
    source_system: int = 255
    source_component: int = 190

    def __post_init__(self) -> None:
        if not isinstance(self.bind_host, str) or not self.bind_host.strip():
            raise ValueError("bind_host must be a non-empty string")
        if isinstance(self.bind_port, bool) or not isinstance(self.bind_port, int) or not 1 <= self.bind_port <= 65535:
            raise ValueError("bind_port must be in [1, 65535]")
        for name in (
            "target_system", "target_component", "source_system", "source_component"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
                raise ValueError(f"{name} must be in [1, 255]")


@dataclass(frozen=True)
class Px4FlightControlHealth:
    opened: bool
    heartbeat_count: int
    armed: bool
    custom_mode: Optional[int]
    system_status: Optional[int]
    last_command_ack: Optional[Tuple[int, int]]
    estimator_health_flags: Optional[int]
    estimator_timeout_flags: Optional[int]
    local_position_ned_m: Optional[Tuple[float, float, float]]
    warnings: Tuple[str, ...]


class Px4MavlinkFlightControl:
    """Non-blocking position-setpoint/arm interface on PX4's GCS UDP link."""

    POSITION_ONLY_TYPE_MASK = 0b110111111000
    PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
    PX4_CUSTOM_MAIN_MODE_AUTO = 4
    PX4_CUSTOM_SUB_MODE_AUTO_LAND = 6

    def __init__(self, config: Px4FlightControlConfig):
        self.config = config
        self._connection = None
        self._mavutil = None
        self._heartbeat_count = 0
        self._armed = False
        self._custom_mode = None
        self._system_status = None
        self._last_command_ack = None
        self._estimator_health_flags = None
        self._estimator_timeout_flags = None
        self._local_position = None
        self._warnings: List[str] = []

    @property
    def endpoint(self) -> str:
        return f"udpin:{self.config.bind_host}:{self.config.bind_port}"

    @property
    def health(self) -> Px4FlightControlHealth:
        return Px4FlightControlHealth(
            opened=self._connection is not None,
            heartbeat_count=self._heartbeat_count,
            armed=self._armed,
            custom_mode=self._custom_mode,
            system_status=self._system_status,
            last_command_ack=self._last_command_ack,
            estimator_health_flags=self._estimator_health_flags,
            estimator_timeout_flags=self._estimator_timeout_flags,
            local_position_ned_m=self._local_position,
            warnings=tuple(self._warnings),
        )

    def open(self) -> None:
        if self._connection is not None:
            raise RuntimeError("PX4 flight-control channel is already open")
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                "PX4 flight control requires the project's optional px4 extra"
            ) from exc
        self._mavutil = mavutil
        self._connection = mavutil.mavlink_connection(
            self.endpoint,
            source_system=self.config.source_system,
            source_component=self.config.source_component,
            autoreconnect=False,
            retries=1,
        )

    def send_position_setpoint_enu(
        self,
        timestamp_us: int,
        position_enu_m,
    ) -> None:
        self._require_open()
        values = tuple(float(value) for value in position_enu_m)
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("position_enu_m must contain three finite values")
        east, north, up = values
        self._connection.mav.set_position_target_local_ned_send(
            int(timestamp_us // 1000),
            self.config.target_system,
            self.config.target_component,
            self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            self.POSITION_ONLY_TYPE_MASK,
            north,
            east,
            -up,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
        )

    def request_offboard_mode(self) -> None:
        self._require_open()
        self._connection.mav.command_long_send(
            self.config.target_system,
            self.config.target_component,
            self._mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MAIN_MODE_OFFBOARD,
            0.0, 0.0, 0.0, 0.0, 0.0,
        )

    def request_arm(self, armed: bool) -> None:
        self._require_open()
        self._connection.mav.command_long_send(
            self.config.target_system,
            self.config.target_component,
            self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0 if armed else 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    def request_auto_land(self) -> None:
        """Switch PX4 to its native controlled landing flight task."""
        self._require_open()
        self._connection.mav.command_long_send(
            self.config.target_system,
            self.config.target_component,
            self._mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MAIN_MODE_AUTO,
            self.PX4_CUSTOM_SUB_MODE_AUTO_LAND,
            0.0, 0.0, 0.0, 0.0,
        )

    def poll(self) -> int:
        """Drain currently available telemetry without advancing wall time."""
        self._require_open()
        count = 0
        while True:
            message = self._connection.recv_match(blocking=False)
            if message is None:
                break
            count += 1
            message_type = message.get_type()
            if message_type == "HEARTBEAT":
                self._heartbeat_count += 1
                self._armed = bool(
                    int(message.base_mode)
                    & self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self._custom_mode = int(message.custom_mode)
                self._system_status = int(message.system_status)
            elif message_type == "COMMAND_ACK":
                self._last_command_ack = (
                    int(message.command), int(message.result)
                )
            elif message_type == "ESTIMATOR_STATUS":
                # MAVLink ESTIMATOR_STATUS exposes a single estimator-status
                # bitmask named ``flags``.  Some generated/vendor dialects add
                # the uORB-style health/timeout fields, so preserve those when
                # present without assuming they exist in common.xml.
                self._estimator_health_flags = int(
                    getattr(message, "health_flags", message.flags)
                )
                timeout_flags = getattr(message, "timeout_flags", None)
                self._estimator_timeout_flags = (
                    None if timeout_flags is None else int(timeout_flags)
                )
            elif message_type == "LOCAL_POSITION_NED":
                self._local_position = (
                    float(message.x), float(message.y), float(message.z)
                )
            elif message_type == "STATUSTEXT":
                severity = int(message.severity)
                if severity <= self._mavutil.mavlink.MAV_SEVERITY_WARNING:
                    text = str(message.text).rstrip("\x00")
                    if text and text not in self._warnings:
                        self._warnings.append(text)
                        del self._warnings[:-100]
        return count

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _require_open(self) -> None:
        if self._connection is None:
            raise RuntimeError("PX4 flight-control channel is not open")
