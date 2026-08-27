"""Pure contraction-acceleration control for four ideal tendon drives."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import numpy as np


_NUM_TENDONS = 4


def _finite_real(name: str, value) -> float:
    """Normalize one finite real scalar while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _four_finite_reals(name: str, values) -> Tuple[float, ...]:
    """Return a defensive four-value copy without flattening its shape."""
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (4,)") from exc
    if array.shape != (_NUM_TENDONS,):
        raise ValueError(
            f"{name} must have shape (4,), got {array.shape}"
        )
    return tuple(
        _finite_real(f"{name}[{index}]", value)
        for index, value in enumerate(array)
    )


def _finite_calculation(name: str, index: int, value: float) -> float:
    """Reject overflow in derived controller values."""
    if not math.isfinite(value):
        raise ValueError(f"{name}[{index}] must be finite")
    return float(value)


@dataclass(frozen=True)
class TendonAccelerationControllerConfig:
    """Explicit scalar parameters for a four-tendon PD servo.

    Gains and limits apply independently and identically to all four manifest-
    ordered channels.  No value in this configuration represents tendon mass,
    motor dynamics, spool geometry, or a hardware rating.
    """

    control_timestep_s: float
    kp_n_per_m: float
    kd_n_s_per_m: float
    max_acceleration_m_s2: float
    max_velocity_m_s: float
    min_contraction_m: float
    max_contraction_m: float
    max_tension_n: float
    preload_tension_n: float = 0.0

    def __post_init__(self):
        for field_name in (
            "control_timestep_s",
            "kp_n_per_m",
            "kd_n_s_per_m",
            "max_acceleration_m_s2",
            "max_velocity_m_s",
            "min_contraction_m",
            "max_contraction_m",
            "max_tension_n",
            "preload_tension_n",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_real(field_name, getattr(self, field_name)),
            )

        if self.control_timestep_s <= 0.0:
            raise ValueError("control_timestep_s must be positive")
        if self.kp_n_per_m < 0.0:
            raise ValueError("kp_n_per_m must be non-negative")
        if self.kd_n_s_per_m < 0.0:
            raise ValueError("kd_n_s_per_m must be non-negative")
        if self.max_acceleration_m_s2 <= 0.0:
            raise ValueError("max_acceleration_m_s2 must be positive")
        if self.max_velocity_m_s <= 0.0:
            raise ValueError("max_velocity_m_s must be positive")
        if self.min_contraction_m >= self.max_contraction_m:
            raise ValueError(
                "min_contraction_m must be less than max_contraction_m"
            )
        if not self.min_contraction_m <= 0.0 <= self.max_contraction_m:
            raise ValueError(
                "contraction limits must contain the reset reference 0"
            )
        if self.max_tension_n <= 0.0:
            raise ValueError("max_tension_n must be positive")
        if not 0.0 <= self.preload_tension_n <= self.max_tension_n:
            raise ValueError(
                "preload_tension_n must be in [0, max_tension_n]"
            )


@dataclass(frozen=True)
class TendonAccelerationControllerState:
    """Immutable manifest-ordered controller state and diagnostics."""

    neutral_tendon_length_m: Tuple[float, ...]
    tendon_length_m: Tuple[float, ...]
    tendon_velocity_m_s: Tuple[float, ...]
    contraction_actual_m: Tuple[float, ...]
    contraction_velocity_actual_m_s: Tuple[float, ...]
    contraction_reference_m: Tuple[float, ...]
    contraction_velocity_reference_m_s: Tuple[float, ...]
    acceleration_command_m_s2: Tuple[float, ...]
    effective_acceleration_m_s2: Tuple[float, ...]
    position_error_m: Tuple[float, ...]
    velocity_error_m_s: Tuple[float, ...]
    unclipped_tension_n: Tuple[float, ...]
    tension_command_n: Tuple[float, ...]
    acceleration_was_clipped: Tuple[bool, ...]
    velocity_was_clipped: Tuple[bool, ...]
    contraction_was_clipped: Tuple[bool, ...]
    reference_velocity_was_projected: Tuple[bool, ...]
    tension_was_clipped_low: Tuple[bool, ...]
    tension_was_clipped_high: Tuple[bool, ...]


class TendonContractionAccelerationController:
    """Integrate four contraction accelerations and produce ideal tensions.

    Positive contraction, contraction velocity, and contraction acceleration
    all describe tendon shortening.  One call to :meth:`update` represents one
    configured control interval; there are no implicit physics substeps.
    """

    def __init__(self, config: TendonAccelerationControllerConfig):
        if not isinstance(config, TendonAccelerationControllerConfig):
            raise TypeError(
                "config must be a TendonAccelerationControllerConfig"
            )
        self._config = config
        self._state = None

    @property
    def config(self) -> TendonAccelerationControllerConfig:
        return self._config

    @property
    def is_reset(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> TendonAccelerationControllerState:
        if self._state is None:
            raise RuntimeError("controller must be reset before reading state")
        return self._state

    def reset(
        self,
        tendon_length_m,
        tendon_velocity_m_s,
    ) -> TendonAccelerationControllerState:
        """Replace all episode state using the current tendon measurements."""
        lengths = _four_finite_reals("tendon_length_m", tendon_length_m)
        if any(value < 0.0 for value in lengths):
            index = next(
                index for index, value in enumerate(lengths) if value < 0.0
            )
            raise ValueError(
                f"tendon_length_m[{index}] must be non-negative"
            )
        velocities = _four_finite_reals(
            "tendon_velocity_m_s", tendon_velocity_m_s
        )
        zeros = (0.0,) * _NUM_TENDONS
        actual_velocities = tuple(-value for value in velocities)
        preload = (self._config.preload_tension_n,) * _NUM_TENDONS
        false_flags = (False,) * _NUM_TENDONS
        self._state = TendonAccelerationControllerState(
            neutral_tendon_length_m=lengths,
            tendon_length_m=lengths,
            tendon_velocity_m_s=velocities,
            contraction_actual_m=zeros,
            contraction_velocity_actual_m_s=actual_velocities,
            contraction_reference_m=zeros,
            contraction_velocity_reference_m_s=actual_velocities,
            acceleration_command_m_s2=zeros,
            effective_acceleration_m_s2=zeros,
            position_error_m=zeros,
            velocity_error_m_s=zeros,
            unclipped_tension_n=preload,
            tension_command_n=preload,
            acceleration_was_clipped=false_flags,
            velocity_was_clipped=false_flags,
            contraction_was_clipped=false_flags,
            reference_velocity_was_projected=false_flags,
            tension_was_clipped_low=false_flags,
            tension_was_clipped_high=false_flags,
        )
        return self._state

    def update(
        self,
        tendon_contraction_acceleration_cmd_m_s2,
        tendon_length_m,
        tendon_velocity_m_s,
    ) -> Tuple[float, ...]:
        """Advance one control interval and return four tensions in newtons."""
        if self._state is None:
            raise RuntimeError("controller must be reset before update")

        acceleration = _four_finite_reals(
            "tendon_contraction_acceleration_cmd_m_s2",
            tendon_contraction_acceleration_cmd_m_s2,
        )
        lengths = _four_finite_reals("tendon_length_m", tendon_length_m)
        if any(value < 0.0 for value in lengths):
            index = next(
                index for index, value in enumerate(lengths) if value < 0.0
            )
            raise ValueError(
                f"tendon_length_m[{index}] must be non-negative"
            )
        velocities = _four_finite_reals(
            "tendon_velocity_m_s", tendon_velocity_m_s
        )

        config = self._config
        previous = self._state
        effective_acceleration = []
        acceleration_clipped = []
        velocity_reference = []
        velocity_clipped = []
        contraction_reference = []
        contraction_clipped = []
        velocity_projected = []

        for index, command in enumerate(acceleration):
            effective = min(
                config.max_acceleration_m_s2,
                max(-config.max_acceleration_m_s2, command),
            )
            effective_acceleration.append(float(effective))
            acceleration_clipped.append(effective != command)

            raw_velocity = _finite_calculation(
                "contraction_velocity_reference_m_s",
                index,
                previous.contraction_velocity_reference_m_s[index]
                + effective * config.control_timestep_s,
            )
            next_velocity = min(
                config.max_velocity_m_s,
                max(-config.max_velocity_m_s, raw_velocity),
            )
            velocity_clipped.append(next_velocity != raw_velocity)

            raw_contraction = _finite_calculation(
                "contraction_reference_m",
                index,
                previous.contraction_reference_m[index]
                + next_velocity * config.control_timestep_s,
            )
            next_contraction = min(
                config.max_contraction_m,
                max(config.min_contraction_m, raw_contraction),
            )
            contraction_clipped.append(next_contraction != raw_contraction)

            projected = False
            if (
                next_contraction == config.max_contraction_m
                and next_velocity > 0.0
            ) or (
                next_contraction == config.min_contraction_m
                and next_velocity < 0.0
            ):
                next_velocity = 0.0
                projected = True
            velocity_projected.append(projected)
            velocity_reference.append(float(next_velocity))
            contraction_reference.append(float(next_contraction))

        neutral_lengths = previous.neutral_tendon_length_m
        contraction_actual = tuple(
            _finite_calculation(
                "contraction_actual_m", index, neutral - current
            )
            for index, (neutral, current) in enumerate(
                zip(neutral_lengths, lengths)
            )
        )
        contraction_velocity_actual = tuple(
            -value for value in velocities
        )
        position_error = tuple(
            _finite_calculation(
                "position_error_m", index, reference - actual
            )
            for index, (reference, actual) in enumerate(
                zip(contraction_reference, contraction_actual)
            )
        )
        velocity_error = tuple(
            _finite_calculation(
                "velocity_error_m_s", index, reference - actual
            )
            for index, (reference, actual) in enumerate(
                zip(velocity_reference, contraction_velocity_actual)
            )
        )

        unclipped_tension = []
        tension = []
        tension_clipped_low = []
        tension_clipped_high = []
        for index, (position_item, velocity_item) in enumerate(
            zip(position_error, velocity_error)
        ):
            raw_tension = _finite_calculation(
                "unclipped_tension_n",
                index,
                config.preload_tension_n
                + config.kp_n_per_m * position_item
                + config.kd_n_s_per_m * velocity_item,
            )
            clipped_tension = min(
                config.max_tension_n, max(0.0, raw_tension)
            )
            unclipped_tension.append(raw_tension)
            tension.append(float(clipped_tension))
            tension_clipped_low.append(raw_tension < 0.0)
            tension_clipped_high.append(raw_tension > config.max_tension_n)

        self._state = TendonAccelerationControllerState(
            neutral_tendon_length_m=neutral_lengths,
            tendon_length_m=lengths,
            tendon_velocity_m_s=velocities,
            contraction_actual_m=contraction_actual,
            contraction_velocity_actual_m_s=(
                contraction_velocity_actual
            ),
            contraction_reference_m=tuple(contraction_reference),
            contraction_velocity_reference_m_s=tuple(velocity_reference),
            acceleration_command_m_s2=acceleration,
            effective_acceleration_m_s2=tuple(effective_acceleration),
            position_error_m=position_error,
            velocity_error_m_s=velocity_error,
            unclipped_tension_n=tuple(unclipped_tension),
            tension_command_n=tuple(tension),
            acceleration_was_clipped=tuple(acceleration_clipped),
            velocity_was_clipped=tuple(velocity_clipped),
            contraction_was_clipped=tuple(contraction_clipped),
            reference_velocity_was_projected=tuple(velocity_projected),
            tension_was_clipped_low=tuple(tension_clipped_low),
            tension_was_clipped_high=tuple(tension_clipped_high),
        )
        return self._state.tension_command_n
