"""Bounded deterministic lockstep between sensor and actuator packets."""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Iterable, Optional, Set, Tuple

from multi_drone_mujoco.backends.contracts import (
    NormalizedMotorCommand,
    finite_tuple,
)
from multi_drone_mujoco.backends.sensors import SensorFrame
from multi_drone_mujoco.backends.transport import (
    SimulatorTransport,
    TransportDisconnected,
    TransportTimeout,
)


class LockstepStatus(Enum):
    READY = "ready"
    PAUSED = "paused"
    TIMED_OUT = "timed_out"
    DISCONNECTED = "disconnected"
    FUTURE_PACKET = "future_packet"
    CLOSED = "closed"


@dataclass(frozen=True)
class LockstepConfig:
    actuator_timeout_s: float
    safe_normalized_values: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        timeout = float(self.actuator_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("actuator_timeout_s must be finite and positive")
        object.__setattr__(self, "actuator_timeout_s", timeout)
        object.__setattr__(
            self,
            "safe_normalized_values",
            finite_tuple("safe_normalized_values", self.safe_normalized_values, 4),
        )


@dataclass(frozen=True)
class LockstepBootstrapConfig:
    """Bounded startup stream that advances PX4 time but not MuJoCo physics."""

    max_frames: int
    wall_interval_s: float
    final_timeout_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer")
        for name in ("wall_interval_s", "final_timeout_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.final_timeout_s == 0.0:
            raise ValueError("final_timeout_s must be positive")


@dataclass(frozen=True)
class LockstepOutcome:
    command: NormalizedMotorCommand
    advance_physics: bool
    status: LockstepStatus
    detail: str = ""


class LockstepCoordinator:
    """Match actuator output to sensor simulation time without unbounded waits."""

    def __init__(self, transport: SimulatorTransport, config: LockstepConfig):
        self.transport = transport
        self.config = config
        self._paused = False
        self._single_steps = 0
        self._closed = False
        self._last_advanced_timestamp_us = -1
        self._last_advanced_sequence = -1
        self._last_sent_timestamp_us = -1
        self._last_sent_sequence = -1
        self._seen_packets: Set[Tuple[int, int]] = set()

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("lockstep coordinator is closed")
        self._paused = False
        self._single_steps = 0
        self._last_advanced_timestamp_us = -1
        self._last_advanced_sequence = -1
        self._last_sent_timestamp_us = -1
        self._last_sent_sequence = -1
        self._seen_packets.clear()

    @property
    def will_exchange(self) -> bool:
        """Whether the next call may consume/send a sensor frame."""
        return not self._closed and (not self._paused or self._single_steps > 0)

    @property
    def last_sent_timestamp_us(self) -> int:
        """Timestamp of the newest sensor frame accepted by the coordinator."""
        return self._last_sent_timestamp_us

    @property
    def last_sent_sequence(self) -> int:
        """Sequence of the newest sensor frame accepted by the coordinator."""
        return self._last_sent_sequence

    def connect(self) -> None:
        if self._closed:
            raise RuntimeError("lockstep coordinator is closed")
        self.transport.connect()

    def reconnect(self) -> None:
        """Explicit bounded recovery; no simulation time is advanced here."""
        if self._closed:
            raise RuntimeError("lockstep coordinator is closed")
        reconnect = getattr(self.transport, "reconnect", None)
        if reconnect is None:
            self.transport.connect()
        else:
            reconnect()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._single_steps = 0

    def request_single_step(self) -> None:
        if not self._paused:
            raise RuntimeError("single-step is only valid while paused")
        self._single_steps += 1

    def exchange(self, frame: SensorFrame) -> LockstepOutcome:
        if self._closed:
            return self._safe(frame, LockstepStatus.CLOSED, "coordinator is closed")
        if self._paused and self._single_steps == 0:
            return self._safe(frame, LockstepStatus.PAUSED, "simulation is paused")
        self._validate_next_frame(frame)
        if self._paused:
            self._single_steps -= 1
        try:
            self.transport.send_sensor_frame(frame)
            self._remember_sent(frame)
        except TransportDisconnected as exc:
            return self._safe(frame, LockstepStatus.DISCONNECTED, str(exc))

        deadline = time.monotonic() + self.config.actuator_timeout_s
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                packet = self.transport.receive_actuator(remaining)
            except TransportTimeout as exc:
                return self._safe(frame, LockstepStatus.TIMED_OUT, str(exc))
            except TransportDisconnected as exc:
                return self._safe(frame, LockstepStatus.DISCONNECTED, str(exc))

            identity = (packet.timestamp_us, packet.sequence)
            if identity in self._seen_packets:
                if remaining <= 0.0:
                    return self._safe(
                        frame, LockstepStatus.TIMED_OUT, "only duplicate packets received"
                    )
                continue
            self._seen_packets.add(identity)
            if packet.timestamp_us < frame.timestamp_us:
                if remaining <= 0.0:
                    return self._safe(
                        frame, LockstepStatus.TIMED_OUT, "only stale packets received"
                    )
                continue
            if packet.timestamp_us > frame.timestamp_us:
                return self._safe(
                    frame,
                    LockstepStatus.FUTURE_PACKET,
                    f"actuator timestamp {packet.timestamp_us} exceeds sensor "
                    f"timestamp {frame.timestamp_us}",
                )
            self._last_advanced_timestamp_us = frame.timestamp_us
            self._last_advanced_sequence = frame.sequence
            return LockstepOutcome(packet, True, LockstepStatus.READY)

    def bootstrap(
        self,
        frames: Iterable[SensorFrame],
        config: LockstepBootstrapConfig,
    ) -> NormalizedMotorCommand:
        """Pace startup sensors until PX4 publishes its first actuator packet."""
        if self._closed:
            raise RuntimeError("lockstep coordinator is closed")
        if self._paused:
            raise RuntimeError("cannot bootstrap while paused")
        poll = getattr(self.transport, "poll_actuator", None)
        if poll is None:
            raise RuntimeError("transport does not support non-blocking bootstrap")
        count = 0
        for frame in frames:
            if count >= config.max_frames:
                break
            self._validate_next_frame(frame)
            self.transport.send_sensor_frame(frame)
            self._remember_sent(frame)
            count += 1
            packet = poll()
            if packet is not None:
                return packet
            if config.wall_interval_s > 0.0:
                time.sleep(config.wall_interval_s)
        if count == 0:
            raise ValueError("bootstrap frames must not be empty")
        try:
            return self.transport.receive_actuator(config.final_timeout_s)
        except TransportTimeout as exc:
            raise TransportTimeout(
                f"PX4 produced no actuator packet during {count}-frame bootstrap: {exc}"
            ) from exc

    def close(self) -> None:
        if not self._closed:
            self.transport.close()
            self._closed = True
            self._single_steps = 0

    def _validate_next_frame(self, frame: SensorFrame) -> None:
        if frame.timestamp_us <= self._last_sent_timestamp_us:
            raise ValueError("sensor timestamp must follow the last sent frame")
        if frame.sequence <= self._last_sent_sequence:
            raise ValueError("sensor sequence must follow the last sent frame")

    def _remember_sent(self, frame: SensorFrame) -> None:
        self._last_sent_timestamp_us = frame.timestamp_us
        self._last_sent_sequence = frame.sequence

    def _safe(
        self, frame: SensorFrame, status: LockstepStatus, detail: str
    ) -> LockstepOutcome:
        command = NormalizedMotorCommand(
            self.config.safe_normalized_values,
            False,
            frame.timestamp_us,
            frame.sequence,
        )
        return LockstepOutcome(command, False, status, detail)
