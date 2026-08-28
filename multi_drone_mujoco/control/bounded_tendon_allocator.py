"""Lexicographic bounded allocation of bending effort to four tensions."""

from dataclasses import dataclass
import itertools
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
        result = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _project_l1(values, radius):
    vector = _vector("desired_bending_effort_n", values, 2)
    absolute = np.abs(vector)
    if float(np.sum(absolute)) <= radius:
        return vector.copy()
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    rho = max(index for index in range(2) if ordered[index] > (cumulative[index] - radius) / (index + 1))
    theta = (cumulative[rho] - radius) / (rho + 1)
    return np.sign(vector) * np.maximum(absolute - theta, 0.0)


@dataclass(frozen=True)
class BoundedTendonAllocatorConfig:
    max_tension_n: float
    bias_tension_n: tuple = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        maximum = _finite("max_tension_n", self.max_tension_n)
        bias = tuple(_vector("bias_tension_n", self.bias_tension_n, 4))
        if maximum <= 0.0:
            raise ValueError("max_tension_n must be positive")
        if any(value < 0.0 or value > maximum for value in bias):
            raise ValueError("bias_tension_n must lie within tension bounds")
        object.__setattr__(self, "max_tension_n", maximum)
        object.__setattr__(self, "bias_tension_n", bias)


@dataclass(frozen=True)
class BoundedTendonAllocation:
    requested_bending_effort_n: tuple
    primary_optimal_bending_effort_n: tuple
    achieved_bending_effort_n: tuple
    tension_n: tuple
    bias_tension_n: tuple
    primary_residual_n: float
    secondary_objective_n2: float
    lower_bound_active: tuple
    upper_bound_active: tuple
    dynamic_common_offset_n: float
    physical_common_effort_n: float
    twist_candidate_effort_n: float
    status: str


class AllocationNumericalError(RuntimeError):
    pass


class BoundedTendonAllocator:
    """Solve primary bounded LS, then minimum bias distance exactly.

    Primary effort is the Euclidean projection onto the exact feasible diamond.
    For each secondary active set, the free variables are the orthogonal
    projection of the bias onto ``A_F t_F = rhs``.  An explicit SVD sets the
    null-space displacement to zero, rather than accepting an arbitrary least-
    squares representative.
    """

    def __init__(self, config, coordinates):
        if not isinstance(config, BoundedTendonAllocatorConfig):
            raise TypeError("config must be BoundedTendonAllocatorConfig")
        if not isinstance(coordinates, TendonModalCoordinates):
            raise TypeError("coordinates must be TendonModalCoordinates")
        self.config = config
        self.coordinates = coordinates

    @staticmethod
    def _minimum_bias_free_solution(matrix, rhs, bias, tolerance):
        if matrix.shape[1] == 0:
            return np.empty(0) if np.linalg.norm(rhs) <= tolerance else None
        shifted_rhs = rhs - matrix @ bias
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
        rank_tolerance = 64 * np.finfo(float).eps * max(matrix.shape) * max(1.0, singular_values[0] if singular_values.size else 0.0)
        rank = int(np.sum(singular_values > rank_tolerance))
        if rank:
            delta = vt[:rank].T @ ((u[:, :rank].T @ shifted_rhs) / singular_values[:rank])
        else:
            delta = np.zeros(matrix.shape[1])
        candidate = bias + delta
        if np.linalg.norm(matrix @ candidate - rhs) > tolerance:
            return None
        return candidate

    def allocate(self, desired_bending_effort_n):
        requested = _vector("desired_bending_effort_n", desired_bending_effort_n, 2)
        maximum = self.config.max_tension_n
        primary = _project_l1(requested, 2.0 * maximum)
        matrix = self.coordinates.bending_g
        bias = np.asarray(self.config.bias_tension_n)
        tolerance = 256 * np.finfo(float).eps * max(1.0, maximum, float(np.linalg.norm(primary)))
        candidates = []
        for states in itertools.product((-1, 0, 1), repeat=4):
            states_array = np.asarray(states)
            free = states_array == 0
            fixed = ~free
            tension = np.empty(4)
            tension[states_array == -1] = 0.0
            tension[states_array == 1] = maximum
            rhs = primary - matrix[:, fixed] @ tension[fixed]
            solution = self._minimum_bias_free_solution(matrix[:, free], rhs, bias[free], tolerance)
            if solution is None or np.any(solution < -tolerance) or np.any(solution > maximum + tolerance):
                continue
            tension[free] = np.clip(solution, 0.0, maximum)
            if np.linalg.norm(matrix @ tension - primary) > tolerance:
                continue
            secondary = float(np.dot(tension - bias, tension - bias))
            candidates.append((secondary, tuple(tension), tension.copy()))
        if not candidates:
            raise AllocationNumericalError("no feasible secondary allocation found")
        candidates.sort(key=lambda item: (item[0], item[1]))
        tension = candidates[0][2]
        achieved = matrix @ tension
        modal_effort = self.coordinates.g @ tension
        mean_bias = float(np.mean(bias))
        return BoundedTendonAllocation(
            requested_bending_effort_n=tuple(requested),
            primary_optimal_bending_effort_n=tuple(primary),
            achieved_bending_effort_n=tuple(achieved),
            tension_n=tuple(float(value) for value in tension),
            bias_tension_n=tuple(float(value) for value in bias),
            primary_residual_n=float(np.linalg.norm(requested - primary)),
            secondary_objective_n2=candidates[0][0],
            lower_bound_active=tuple(bool(value <= tolerance) for value in tension),
            upper_bound_active=tuple(bool(value >= maximum - tolerance) for value in tension),
            dynamic_common_offset_n=float(np.mean(tension) - mean_bias),
            physical_common_effort_n=float(modal_effort[0]),
            twist_candidate_effort_n=float(modal_effort[3]),
            status=("optimal_feasible" if np.linalg.norm(requested - primary) <= tolerance else "saturated_best_effort"),
        )
