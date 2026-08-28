"""Calibrated static effort feedforward in two tendon bending modes."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _positive(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _two(name, values):
    try:
        result = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite values") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result


@dataclass(frozen=True)
class BendingModalStaticFeedforwardConfig:
    """A bounded-domain 2-D static plant model, ``effort = K_static q_ref``.

    ``stiffness_n_per_m`` contains the x/y diagonal of the measured modal
    stiffness in N/m.  This is static equilibrium feedforward, not an
    acceleration inverse-dynamics or tendon-mass model.
    """

    stiffness_n_per_m: tuple
    max_reference_l1_m: float
    max_actual_l1_m: float

    def __post_init__(self):
        stiffness = tuple(_two("stiffness_n_per_m", self.stiffness_n_per_m))
        if any(value <= 0.0 for value in stiffness):
            raise ValueError("stiffness_n_per_m values must be positive")
        object.__setattr__(self, "stiffness_n_per_m", stiffness)
        object.__setattr__(
            self,
            "max_reference_l1_m",
            _positive("max_reference_l1_m", self.max_reference_l1_m),
        )
        object.__setattr__(
            self,
            "max_actual_l1_m",
            _positive("max_actual_l1_m", self.max_actual_l1_m),
        )


@dataclass(frozen=True)
class BendingModalStaticFeedforwardState:
    position_reference_m: tuple
    position_actual_m: tuple
    static_bending_effort_n: tuple
    reference_l1_m: float
    actual_l1_m: float
    out_of_envelope: bool
    envelope_reasons: tuple


class BendingModalStaticFeedforward:
    """Evaluate the calibrated diagonal static stiffness without inversion."""

    def __init__(self, config):
        if not isinstance(config, BendingModalStaticFeedforwardConfig):
            raise TypeError("config must be BendingModalStaticFeedforwardConfig")
        self.config = config
        self._state = None

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("static feedforward has not been evaluated")
        return self._state

    def update(self, position_reference_m, position_actual_m):
        reference = _two("position_reference_m", position_reference_m)
        actual = _two("position_actual_m", position_actual_m)
        reference_l1 = float(np.sum(np.abs(reference)))
        actual_l1 = float(np.sum(np.abs(actual)))
        tolerance = 64.0 * np.finfo(float).eps
        reasons = []
        if reference_l1 > self.config.max_reference_l1_m + tolerance:
            reasons.append("reference")
        if actual_l1 > self.config.max_actual_l1_m + tolerance:
            reasons.append("actual")
        with np.errstate(over="raise", invalid="raise"):
            effort = np.asarray(self.config.stiffness_n_per_m) * reference
        if not np.all(np.isfinite(effort)):
            raise FloatingPointError("non-finite static feedforward effort")
        self._state = BendingModalStaticFeedforwardState(
            position_reference_m=tuple(reference),
            position_actual_m=tuple(actual),
            static_bending_effort_n=tuple(effort),
            reference_l1_m=reference_l1,
            actual_l1_m=actual_l1,
            out_of_envelope=bool(reasons),
            envelope_reasons=tuple(reasons),
        )
        return self._state.static_bending_effort_n
