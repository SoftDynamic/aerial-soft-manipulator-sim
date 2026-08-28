"""Explicit two-dimensional PD feedback in tendon bending coordinates."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _gain(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
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
class BendingModalPDConfig:
    kp_n_per_m: float
    kd_n_s_per_m: float

    def __post_init__(self):
        object.__setattr__(self, "kp_n_per_m", _gain("kp_n_per_m", self.kp_n_per_m))
        object.__setattr__(self, "kd_n_s_per_m", _gain("kd_n_s_per_m", self.kd_n_s_per_m))


@dataclass(frozen=True)
class BendingModalPDState:
    position_reference_m: tuple
    velocity_reference_m_s: tuple
    position_actual_m: tuple
    velocity_actual_m_s: tuple
    position_error_m: tuple
    velocity_error_m_s: tuple
    desired_bending_effort_n: tuple


class BendingModalPD:
    def __init__(self, config):
        if not isinstance(config, BendingModalPDConfig):
            raise TypeError("config must be BendingModalPDConfig")
        self.config = config
        self._state = None

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("modal PD has not been evaluated")
        return self._state

    def update(self, position_reference_m, velocity_reference_m_s, position_actual_m, velocity_actual_m_s):
        q_ref = _two("position_reference_m", position_reference_m)
        v_ref = _two("velocity_reference_m_s", velocity_reference_m_s)
        q = _two("position_actual_m", position_actual_m)
        v = _two("velocity_actual_m_s", velocity_actual_m_s)
        position_error = q_ref - q
        velocity_error = v_ref - v
        effort = self.config.kp_n_per_m * position_error + self.config.kd_n_s_per_m * velocity_error
        self._state = BendingModalPDState(
            tuple(q_ref), tuple(v_ref), tuple(q), tuple(v), tuple(position_error), tuple(velocity_error), tuple(effort)
        )
        return self._state.desired_bending_effort_n
