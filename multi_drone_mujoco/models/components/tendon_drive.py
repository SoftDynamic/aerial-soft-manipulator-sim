"""Ideal tendon-tension actuation helpers."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Tuple


def _finite_real(name: str, value) -> float:
    """Normalize one finite real value to a built-in float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


BASELINE_MAX_TENSION_N = 0.01


@dataclass(frozen=True)
class IdealTendonTensionSpec:
    """Configure an ideal, internally instantaneous tendon tension source.

    ``max_tension`` is expressed in newtons.  This abstraction is not a
    muscle, physical motor, FOC loop, spool, or tendon-length servo.
    """

    max_tension: float

    def __post_init__(self):
        normalized = _finite_real(
            "Ideal tendon tension max_tension",
            self.max_tension,
        )
        if normalized <= 0:
            raise ValueError(
                "Ideal tendon tension max_tension must be positive"
            )
        object.__setattr__(self, "max_tension", normalized)


def validate_ideal_tendon_tension_command(
    tendon_tension_cmd_n,
    spec: IdealTendonTensionSpec,
    expected_size: int = 4,
) -> Tuple[float, ...]:
    """Return an immutable tension command in newtons after validation."""
    try:
        raw_values = tuple(tendon_tension_cmd_n)
    except TypeError as exc:
        raise ValueError(
            "Ideal tendon tension command must be a sequence"
        ) from exc
    if len(raw_values) != expected_size:
        raise ValueError(
            "Ideal tendon tension command must contain "
            f"{expected_size} values"
        )

    values = tuple(
        _finite_real(f"Ideal tendon tension command[{index}]", value)
        for index, value in enumerate(raw_values)
    )
    if any(value < 0 for value in values):
        raise ValueError("Ideal tendon tension command must be non-negative")
    if any(value > spec.max_tension for value in values):
        raise ValueError(
            "Ideal tendon tension command must not exceed max_tension"
        )
    return values


def _float_text(value: float) -> str:
    return format(value, ".17g")


def build_ideal_tendon_tension_actuator_xml(
    tendon_names,
    spec: IdealTendonTensionSpec,
) -> Tuple[str, Tuple[str, ...]]:
    """Build one direct tendon motor per name, preserving input order."""
    tendon_names = tuple(tendon_names)
    actuator_names = tuple(
        f"t{index}_tension_N"
        for index in range(len(tendon_names))
    )
    maximum = _float_text(spec.max_tension)
    actuator_xml = "\n".join(
        f'    <motor name="{actuator_name}" tendon="{tendon_name}" '
        f'gear="-1" ctrllimited="true" ctrlrange="0 {maximum}"/>'
        for tendon_name, actuator_name in zip(
            tendon_names,
            actuator_names,
        )
    )
    return actuator_xml, actuator_names
