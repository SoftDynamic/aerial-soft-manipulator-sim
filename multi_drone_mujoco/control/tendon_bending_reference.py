"""Semi-implicit reference integration in 2-D tendon bending coordinates."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from multi_drone_mujoco.control.tendon_modal_coordinates import TendonModalCoordinates


def _finite(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(name, values, size):
    try:
        array = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def project_l1_ball(values, radius):
    """Euclidean projection of a two-vector onto ``|x|+|y| <= radius``."""
    vector = _vector("values", values, 2)
    bound = _finite("radius", radius)
    if bound <= 0.0:
        raise ValueError("radius must be positive")
    absolute = np.abs(vector)
    if float(np.sum(absolute)) <= bound:
        return vector.copy(), False
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    rho = max(index for index in range(2) if ordered[index] > (cumulative[index] - bound) / (index + 1))
    theta = (cumulative[rho] - bound) / (rho + 1)
    return np.sign(vector) * np.maximum(absolute - theta, 0.0), True


@dataclass(frozen=True)
class BendingReferenceConfig:
    control_timestep_s: float
    max_acceleration_m_s2: float
    max_velocity_m_s: float
    max_contraction_m: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.control_timestep_s <= 0.0:
            raise ValueError("control_timestep_s must be positive")
        for name in ("max_acceleration_m_s2", "max_velocity_m_s", "max_contraction_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BendingReferenceState:
    requested_bending_acceleration_m_s2: tuple
    effective_bending_acceleration_m_s2: tuple
    bending_velocity_reference_m_s: tuple
    bending_contraction_reference_m: tuple
    tendon_acceleration_reference_m_s2: tuple
    tendon_velocity_reference_m_s: tuple
    tendon_contraction_reference_m: tuple
    acceleration_was_projected: bool
    velocity_was_projected: bool
    contraction_was_projected: bool
    common_residuals: tuple
    twist_candidate_residuals: tuple


class BendingModalReferenceIntegrator:
    """Integrate q_x/q_y tendon-modal acceleration in m/s^2.

    q_x/q_y are tendon modal contractions, not UAV/tip Cartesian position,
    curvature, or motor/spool coordinates.  Integration is performed once per
    configured control interval: velocity first, then contraction.
    """

    def __init__(self, config, coordinates):
        if not isinstance(config, BendingReferenceConfig):
            raise TypeError("config must be BendingReferenceConfig")
        if not isinstance(coordinates, TendonModalCoordinates):
            raise TypeError("coordinates must be TendonModalCoordinates")
        self.config = config
        self.coordinates = coordinates
        self._state = None

    @property
    def state(self):
        if self._state is None:
            raise RuntimeError("reference integrator must be reset before use")
        return self._state

    def _state_from(self, requested, effective, velocity, contraction, flags):
        tendon_a = self.coordinates.bending_to_tendon(effective)
        tendon_v = self.coordinates.bending_to_tendon(velocity)
        tendon_s = self.coordinates.bending_to_tendon(contraction)
        modal_a = self.coordinates.tendon_to_modal(tendon_a)
        modal_v = self.coordinates.tendon_to_modal(tendon_v)
        modal_s = self.coordinates.tendon_to_modal(tendon_s)
        return BendingReferenceState(
            requested_bending_acceleration_m_s2=tuple(requested),
            effective_bending_acceleration_m_s2=tuple(effective),
            bending_velocity_reference_m_s=tuple(velocity),
            bending_contraction_reference_m=tuple(contraction),
            tendon_acceleration_reference_m_s2=tuple(tendon_a),
            tendon_velocity_reference_m_s=tuple(tendon_v),
            tendon_contraction_reference_m=tuple(tendon_s),
            acceleration_was_projected=bool(flags[0]),
            velocity_was_projected=bool(flags[1]),
            contraction_was_projected=bool(flags[2]),
            common_residuals=(float(modal_a[0]), float(modal_v[0]), float(modal_s[0])),
            twist_candidate_residuals=(float(modal_a[3]), float(modal_v[3]), float(modal_s[3])),
        )

    def reset(self, tendon_velocity_m_s):
        contraction_velocity = -_vector("tendon_velocity_m_s", tendon_velocity_m_s, 4)
        velocity = self.coordinates.tendon_to_bending(contraction_velocity)
        zeros = np.zeros(2)
        self._state = self._state_from(zeros, zeros, velocity, zeros, (False, False, False))
        return self._state

    def update_bending(self, bending_acceleration_m_s2):
        previous = self.state
        requested = _vector("bending_acceleration_m_s2", bending_acceleration_m_s2, 2)
        effective, a_clipped = project_l1_ball(requested, self.config.max_acceleration_m_s2)
        raw_velocity = np.asarray(previous.bending_velocity_reference_m_s) + effective * self.config.control_timestep_s
        velocity, v_clipped = project_l1_ball(raw_velocity, self.config.max_velocity_m_s)
        previous_q = np.asarray(previous.bending_contraction_reference_m)
        raw_contraction = previous_q + velocity * self.config.control_timestep_s
        contraction, q_clipped = project_l1_ball(raw_contraction, self.config.max_contraction_m)
        if q_clipped:
            velocity = (contraction - previous_q) / self.config.control_timestep_s
        self._state = self._state_from(requested, effective, velocity, contraction, (a_clipped, v_clipped, q_clipped))
        return self._state

    def update_tendon(self, tendon_acceleration_m_s2):
        projection = self.coordinates.project_to_bending(tendon_acceleration_m_s2)
        return self.update_bending(projection.bending_coordinates), projection
