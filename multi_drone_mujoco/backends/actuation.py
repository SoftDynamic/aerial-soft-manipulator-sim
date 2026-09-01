"""Explicit normalized-output, PWM, and rotor-speed actuator models."""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from multi_drone_mujoco.backends.contracts import (
    NormalizedMotorCommand,
    PwmMotorCommand,
    RotorSpeedCommand,
    finite_tuple,
)


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _motor_order(values) -> Tuple[int, int, int, int]:
    result = tuple(values)
    if result != tuple(int(value) for value in result):
        raise ValueError("motor_order values must be integers")
    result = tuple(int(value) for value in result)
    if sorted(result) != [0, 1, 2, 3]:
        raise ValueError("motor_order must be a permutation of 0, 1, 2, 3")
    return result


def _motor_directions(values) -> Tuple[int, int, int, int]:
    result = tuple(values)
    if len(result) != 4 or any(value not in (-1, 1) for value in result):
        raise ValueError("motor_directions must contain four -1/+1 values")
    return tuple(int(value) for value in result)


@dataclass(frozen=True)
class NormalizedToPwmConfig:
    """Calibration from PX4 channel order to simulator PWM semantics.

    ``motor_order[i]`` is the input channel used for simulator rotor ``i``.
    """

    pwm_disarmed_us: float
    pwm_min_us: float
    pwm_max_us: float
    normalized_min: float
    normalized_max: float
    motor_order: Tuple[int, int, int, int]
    motor_directions: Tuple[int, int, int, int]
    clip_out_of_range: bool = True

    def __post_init__(self) -> None:
        for name in (
            "pwm_disarmed_us",
            "pwm_min_us",
            "pwm_max_us",
            "normalized_min",
            "normalized_max",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.pwm_disarmed_us > self.pwm_min_us:
            raise ValueError("pwm_disarmed_us must be <= pwm_min_us")
        if self.pwm_min_us >= self.pwm_max_us:
            raise ValueError("pwm_min_us must be less than pwm_max_us")
        if self.normalized_min >= self.normalized_max:
            raise ValueError("normalized_min must be less than normalized_max")
        object.__setattr__(self, "motor_order", _motor_order(self.motor_order))
        object.__setattr__(
            self, "motor_directions", _motor_directions(self.motor_directions)
        )
        if not isinstance(self.clip_out_of_range, bool):
            raise ValueError("clip_out_of_range must be bool")


class Px4NormalizedOutputToPwm:
    """Linear, visible PX4 normalized-output to PWM conversion."""

    def __init__(self, config: NormalizedToPwmConfig):
        self.config = config

    def convert(self, command: NormalizedMotorCommand) -> PwmMotorCommand:
        values = np.asarray(command.values, dtype=float)
        values = values[list(self.config.motor_order)]
        if not command.armed:
            pwm = np.full(4, self.config.pwm_disarmed_us, dtype=float)
        else:
            lower = self.config.normalized_min
            upper = self.config.normalized_max
            if not self.config.clip_out_of_range and (
                np.any(values < lower) or np.any(values > upper)
            ):
                raise ValueError("normalized motor output is outside configured range")
            values = np.clip(values, lower, upper)
            fraction = (values - lower) / (upper - lower)
            pwm = self.config.pwm_min_us + fraction * (
                self.config.pwm_max_us - self.config.pwm_min_us
            )
        return PwmMotorCommand(
            pulse_width_us=tuple(pwm),
            armed=command.armed,
            timestamp_us=command.timestamp_us,
            sequence=command.sequence,
        )


@dataclass(frozen=True)
class PwmToRotorSpeedConfig:
    """ESC/motor calibration in explicit microseconds, RPM, and seconds."""

    pwm_disarmed_us: float
    pwm_min_us: float
    pwm_max_us: float
    rpm_disarmed: float
    rpm_idle: float
    rpm_max: float
    motor_time_constant_s: float
    motor_directions: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        for name in (
            "pwm_disarmed_us",
            "pwm_min_us",
            "pwm_max_us",
            "rpm_disarmed",
            "rpm_idle",
            "rpm_max",
            "motor_time_constant_s",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.pwm_disarmed_us > self.pwm_min_us:
            raise ValueError("pwm_disarmed_us must be <= pwm_min_us")
        if self.pwm_min_us >= self.pwm_max_us:
            raise ValueError("pwm_min_us must be less than pwm_max_us")
        if not 0.0 <= self.rpm_disarmed <= self.rpm_idle < self.rpm_max:
            raise ValueError(
                "RPM values must satisfy 0 <= disarmed <= idle < max"
            )
        if self.motor_time_constant_s < 0.0:
            raise ValueError("motor_time_constant_s must be non-negative")
        object.__setattr__(
            self, "motor_directions", _motor_directions(self.motor_directions)
        )


class PwmToRotorSpeedModel:
    """PWM to RPM model with an optional exact first-order motor response."""

    def __init__(self, config: PwmToRotorSpeedConfig):
        self.config = config
        self._rpm = np.full(4, config.rpm_disarmed, dtype=float)

    @property
    def rotor_directions(self) -> Tuple[int, int, int, int]:
        """CW/CCW sign metadata in simulator rotor order."""
        return self.config.motor_directions

    def reset(self) -> None:
        self._rpm[:] = self.config.rpm_disarmed

    def convert(self, command: PwmMotorCommand, dt_s: float) -> RotorSpeedCommand:
        dt_s = _finite("dt_s", dt_s)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if not command.armed:
            target = np.full(4, self.config.rpm_disarmed, dtype=float)
        else:
            pwm = np.clip(
                np.asarray(command.pulse_width_us, dtype=float),
                self.config.pwm_min_us,
                self.config.pwm_max_us,
            )
            fraction = (pwm - self.config.pwm_min_us) / (
                self.config.pwm_max_us - self.config.pwm_min_us
            )
            target = self.config.rpm_idle + fraction * (
                self.config.rpm_max - self.config.rpm_idle
            )
        if self.config.motor_time_constant_s == 0.0 or not command.armed:
            self._rpm = target
        else:
            alpha = -math.expm1(-dt_s / self.config.motor_time_constant_s)
            self._rpm = self._rpm + alpha * (target - self._rpm)
        self._rpm = np.clip(
            self._rpm, self.config.rpm_disarmed, self.config.rpm_max
        )
        return RotorSpeedCommand(
            rpm=tuple(self._rpm),
            timestamp_us=command.timestamp_us,
            sequence=command.sequence,
            armed=command.armed,
        )
