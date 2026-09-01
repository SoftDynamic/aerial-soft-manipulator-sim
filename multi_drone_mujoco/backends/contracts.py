"""Immutable, unit-bearing contracts at controller/simulator boundaries."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Protocol, Tuple, runtime_checkable


MotorTuple = Tuple[float, float, float, float]


def finite_tuple(name: str, values: Any, size: int) -> Tuple[float, ...]:
    """Return a fixed-size tuple of finite real values."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} real values") from exc
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class SimulatorClock:
    """One simulation instant, independent of wall-clock time."""

    timestamp_us: int
    sequence: int

    def __post_init__(self) -> None:
        nonnegative_int("timestamp_us", self.timestamp_us)
        nonnegative_int("sequence", self.sequence)

    @classmethod
    def from_seconds(cls, seconds: float, sequence: int) -> "SimulatorClock":
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("seconds must be finite and non-negative")
        return cls(timestamp_us=int(round(seconds * 1_000_000)), sequence=sequence)


@dataclass(frozen=True)
class NormalizedMotorCommand:
    """Four dimensionless PX4 motor outputs in transport channel order."""

    values: MotorTuple
    armed: bool
    timestamp_us: int
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", finite_tuple("values", self.values, 4))
        if not isinstance(self.armed, bool):
            raise ValueError("armed must be bool")
        nonnegative_int("timestamp_us", self.timestamp_us)
        nonnegative_int("sequence", self.sequence)


@dataclass(frozen=True)
class PwmMotorCommand:
    """Four pulse-width commands in microseconds, in simulator rotor order."""

    pulse_width_us: MotorTuple
    armed: bool
    timestamp_us: int
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pulse_width_us",
            finite_tuple("pulse_width_us", self.pulse_width_us, 4),
        )
        if not isinstance(self.armed, bool):
            raise ValueError("armed must be bool")
        nonnegative_int("timestamp_us", self.timestamp_us)
        nonnegative_int("sequence", self.sequence)


@dataclass(frozen=True)
class RotorSpeedCommand:
    """Four non-negative rotor speeds in revolutions per minute."""

    rpm: MotorTuple
    timestamp_us: int
    sequence: int
    armed: bool = True

    def __post_init__(self) -> None:
        values = finite_tuple("rpm", self.rpm, 4)
        if any(value < 0.0 for value in values):
            raise ValueError("rpm values must be non-negative")
        object.__setattr__(self, "rpm", values)
        nonnegative_int("timestamp_us", self.timestamp_us)
        nonnegative_int("sequence", self.sequence)
        if not isinstance(self.armed, bool):
            raise ValueError("armed must be bool")


class ControllerState(Enum):
    """Backend lifecycle/health state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True)
class ControllerHealth:
    """Diagnosable backend status at a simulation instant."""

    state: ControllerState
    timestamp_us: int
    last_sequence: int
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, ControllerState):
            raise ValueError("state must be ControllerState")
        nonnegative_int("timestamp_us", self.timestamp_us)
        if isinstance(self.last_sequence, bool) or not isinstance(self.last_sequence, int):
            raise ValueError("last_sequence must be an integer")
        if self.last_sequence < -1:
            raise ValueError("last_sequence must be >= -1")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string")


@dataclass(frozen=True)
class ControllerStepResult:
    """A safe actuator command plus explicit permission to advance physics."""

    rotor_speed: RotorSpeedCommand
    advance_physics: bool
    health: ControllerHealth

    def __post_init__(self) -> None:
        if not isinstance(self.rotor_speed, RotorSpeedCommand):
            raise ValueError("rotor_speed must be RotorSpeedCommand")
        if not isinstance(self.advance_physics, bool):
            raise ValueError("advance_physics must be bool")
        if not isinstance(self.health, ControllerHealth):
            raise ValueError("health must be ControllerHealth")


@runtime_checkable
class ControllerBackend(Protocol):
    """Controller lifecycle independent of output representation/transport."""

    def reset(self, clock: SimulatorClock) -> None:
        """Reset all episode state at ``clock``."""

    def step(self, sensor_frame: Any) -> ControllerStepResult:
        """Produce a command and explicit lockstep advance decision."""

    @property
    def health(self) -> ControllerHealth:
        """Return immutable current health information."""

    def close(self) -> None:
        """Release owned resources; repeated calls must be harmless."""
