"""Timeout safety for rotor-speed controller outputs."""

from typing import Optional, Tuple

from multi_drone_mujoco.backends.contracts import (
    RotorSpeedCommand,
    finite_tuple,
    nonnegative_int,
)


class CommandWatchdog:
    """Hold a fresh command and otherwise emit an explicit disarmed command."""

    def __init__(self, timeout_us: int, safe_rpm=(0.0, 0.0, 0.0, 0.0)):
        nonnegative_int("timeout_us", timeout_us)
        if timeout_us == 0:
            raise ValueError("timeout_us must be positive")
        values = finite_tuple("safe_rpm", safe_rpm, 4)
        if any(value < 0.0 for value in values):
            raise ValueError("safe_rpm values must be non-negative")
        self.timeout_us = timeout_us
        self.safe_rpm: Tuple[float, ...] = values
        self._last: Optional[RotorSpeedCommand] = None
        self._closed = False

    def reset(self) -> None:
        self._last = None
        self._closed = False

    def accept(self, command: RotorSpeedCommand) -> None:
        if self._closed:
            raise RuntimeError("watchdog is closed")
        if self._last is not None:
            if command.timestamp_us <= self._last.timestamp_us:
                raise ValueError("command timestamp must increase monotonically")
            if command.sequence <= self._last.sequence:
                raise ValueError("command sequence must increase monotonically")
        self._last = command

    def output(self, clock_timestamp_us: int, sequence: int) -> RotorSpeedCommand:
        nonnegative_int("clock_timestamp_us", clock_timestamp_us)
        nonnegative_int("sequence", sequence)
        if self._closed or self._last is None:
            return self._safe(clock_timestamp_us, sequence)
        age_us = clock_timestamp_us - self._last.timestamp_us
        if age_us < 0:
            raise ValueError("clock_timestamp_us precedes the accepted command")
        if age_us > self.timeout_us:
            return self._safe(clock_timestamp_us, sequence)
        return self._last

    def close(self) -> None:
        self._last = None
        self._closed = True

    def _safe(self, timestamp_us: int, sequence: int) -> RotorSpeedCommand:
        return RotorSpeedCommand(
            rpm=self.safe_rpm,
            timestamp_us=timestamp_us,
            sequence=sequence,
            armed=False,
        )
