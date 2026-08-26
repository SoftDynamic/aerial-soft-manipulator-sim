"""C1-B4 active-hover direction symmetry helpers.

The formal trajectories reuse the C1-B3 production-model runner.  This module
adds only immutable direction metadata, manifest-derived channel permutations,
coordinate-aware symmetry metrics, and robust SO(3) comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import mujoco
import numpy as np

from multi_drone_mujoco.tests._scaled_stablesoft_active_coupled import (
    MAX_TENSION_N,
    _make_active_production_env,
)
from multi_drone_mujoco.tests._scaled_stablesoft_active_hover import (
    CORNERS,
    CTRL_FREQ,
    DURATION_S,
    EARLY_END_S,
    FORCED_END_S,
    FORMAL_TENSION_N,
    RELEASE_TICK,
    C1B3RunConfig,
    _normalized_quaternion,
    _relative_quaternion,
    _rotation_vector,
    _run_active_hover,
    convergence_metrics as active_hover_convergence_metrics,
)


SIM_FREQUENCIES = (2400, 4800)
DIRECTION_CASES = ("plus_x", "minus_x", "plus_y", "minus_y")
B4_CASE_CORNERS = MappingProxyType(
    {
        "baseline": (),
        "plus_x": ("xp_yp", "xp_yn"),
        "minus_x": ("xn_yp", "xn_yn"),
        "plus_y": ("xp_yp", "xn_yp"),
        "minus_y": ("xp_yn", "xn_yn"),
        "common_mode_max": CORNERS,
    }
)
B4_CASE_TENSION_N = MappingProxyType(
    {
        "baseline": FORMAL_TENSION_N,
        "plus_x": FORMAL_TENSION_N,
        "minus_x": FORMAL_TENSION_N,
        "plus_y": FORMAL_TENSION_N,
        "minus_y": FORMAL_TENSION_N,
        "common_mode_max": MAX_TENSION_N,
    }
)
B4_FORMAL_CASES = tuple(B4_CASE_CORNERS)

WINDOW_END_S = MappingProxyType(
    {"forced": FORCED_END_S, "early": EARLY_END_S, "full": DURATION_S}
)
WINDOW_RELATIVE_FRACTION = MappingProxyType(
    {"forced": 5e-4, "early": 5e-4, "full": 2e-3}
)

DIRECTION_PRIMARY_AXIS = MappingProxyType(
    {
        "plus_x": (0, 1.0, 1),
        "minus_x": (0, -1.0, 1),
        "plus_y": (1, 1.0, 0),
        "minus_y": (1, -1.0, 0),
    }
)
RELEASE_PRIMARY_ACTIVITY_FLOOR_M = 8e-4
ORTHOGONAL_LEAKAGE_ABSOLUTE_FLOOR_M = 1e-10
ORTHOGONAL_LEAKAGE_RELATIVE_FRACTION = 5e-4

DIRECTION_ACTIVITY_FLOORS = MappingProxyType(
    {
        "uav_position_m": 2e-2,
        "uav_attitude_rad": 5e-2,
        "tip_body_m": 8e-4,
        "joint_rotation_rad": 3e-3,
        "composite_rotation_rad": 2e-2,
        "tendon_excursion_m": 7e-5,
    }
)

COMMON_MODE_DISPLACEMENT_LIMIT_M = 1e-10
COMMON_MODE_ANGLE_LIMIT_RAD = 1e-10
COMMON_MODE_TENDON_LIMIT_M = 1e-10
COMMON_MODE_FORCE_SPREAD_LIMIT_N = 1e-15


def _freeze(values, *, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def mixed_limit(absolute_floor, relative_fraction, activity):
    values = (float(absolute_floor), float(relative_fraction), float(activity))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("mixed-limit inputs must be finite and non-negative")
    return max(values[0], values[1] * values[2])


@dataclass(frozen=True)
class C1B4RunConfig(C1B3RunConfig):
    """Independent C1-B4 case configuration accepted by the shared runner."""

    def __post_init__(self):
        if self.sim_freq not in SIM_FREQUENCIES:
            raise ValueError("C1-B4 sim_freq must be 2400 or 4800")
        if self.case not in B4_CASE_CORNERS:
            raise ValueError(f"unknown C1-B4 case: {self.case}")
        if self.ctrl_freq != CTRL_FREQ:
            raise ValueError("C1-B4 ctrl_freq must be 240")
        if self.duration_s != DURATION_S:
            raise ValueError("C1-B4 duration_s must be 3.0")
        if self.integrator != "implicitfast":
            raise ValueError("C1-B4 integrator must be implicitfast")
        if not math.isfinite(self.tension_n):
            raise ValueError("C1-B4 tension must be finite")
        expected = B4_CASE_TENSION_N[self.case]
        if self.tension_n != expected:
            raise ValueError(
                f"C1-B4 {self.case} tension must be exactly {expected} N"
            )


@dataclass(frozen=True)
class TransformSpec:
    name: str
    matrix: np.ndarray
    corner_map_items: tuple

    @property
    def determinant(self):
        return float(np.linalg.det(self.matrix))

    @property
    def axial_matrix(self):
        return _freeze(self.determinant * self.matrix)

    @property
    def corner_map(self):
        return MappingProxyType(dict(self.corner_map_items))


@dataclass(frozen=True)
class SymmetryRelation:
    name: str
    reference_case: str
    candidate_case: str
    transform_name: str


@dataclass(frozen=True)
class AcceptanceMetric:
    case: str
    sim_freq: int
    window: str
    field: str
    activity: float
    error: float
    absolute_floor: float
    relative_fraction: float
    limit: float
    unit: str
    passed: bool


@dataclass(frozen=True)
class DirectionAnchor:
    case: str
    sim_freq: int
    release_vector_m: tuple
    signed_primary_m: float
    primary_floor_m: float
    orthogonal_leakage_m: float
    orthogonal_limit_m: float
    passed: bool


@dataclass(frozen=True)
class LocalFrameTopologyProof:
    manifest_corners: tuple
    maximum_body_quaternion_identity_error: float
    maximum_body_xy_offset_m: float
    maximum_inertia_xy_difference: float
    maximum_joint_anchor_error_m: float
    maximum_joint_damping_axis_spread: float
    joint_stiffness_values: tuple
    routing_corner_signs_match: bool
    actuator_control_ranges_match: bool
    actuator_force_limits_disabled: bool

    @property
    def passed(self):
        return (
            self.manifest_corners == CORNERS
            and self.maximum_body_quaternion_identity_error == 0.0
            and self.maximum_body_xy_offset_m == 0.0
            and self.maximum_inertia_xy_difference == 0.0
            and self.maximum_joint_anchor_error_m == 0.0
            and self.maximum_joint_damping_axis_spread == 0.0
            and len(set(self.joint_stiffness_values)) == 1
            and self.routing_corner_signs_match
            and self.actuator_control_ranges_match
            and self.actuator_force_limits_disabled
        )


CORNER_COORDINATES = MappingProxyType(
    {
        "xp_yp": (1, 1),
        "xp_yn": (1, -1),
        "xn_yp": (-1, 1),
        "xn_yn": (-1, -1),
    }
)


def _corner_map_for_matrix(matrix):
    by_coordinates = {value: key for key, value in CORNER_COORDINATES.items()}
    result = []
    for corner, coordinates in CORNER_COORDINATES.items():
        vector = np.array([coordinates[0], coordinates[1], 0.0])
        transformed = np.asarray(matrix) @ vector
        target_coordinates = tuple(int(round(value)) for value in transformed[:2])
        if target_coordinates not in by_coordinates:
            raise AssertionError(
                f"transform does not map corner {corner}: {transformed}"
            )
        result.append((corner, by_coordinates[target_coordinates]))
    return tuple(result)


def _transform(name, matrix):
    matrix = _freeze(matrix)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-15):
        raise AssertionError(f"non-orthogonal transform: {name}")
    determinant = float(np.linalg.det(matrix))
    if not math.isclose(abs(determinant), 1.0, abs_tol=1e-15):
        raise AssertionError(f"invalid transform determinant: {name}")
    return TransformSpec(name, matrix, _corner_map_for_matrix(matrix))


TRANSFORMS = MappingProxyType(
    {
        "Sx": _transform("Sx", np.diag((-1.0, 1.0, 1.0))),
        "Sy": _transform("Sy", np.diag((1.0, -1.0, 1.0))),
        "Rz_pos_90": _transform(
            "Rz_pos_90",
            np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        ),
        "Rz_neg_90": _transform(
            "Rz_neg_90",
            np.array(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        ),
        "Rz_180": _transform("Rz_180", np.diag((-1.0, -1.0, 1.0))),
    }
)

SYMMETRY_RELATIONS = (
    SymmetryRelation("plus_x_to_minus_x", "plus_x", "minus_x", "Sx"),
    SymmetryRelation("plus_y_to_minus_y", "plus_y", "minus_y", "Sy"),
    SymmetryRelation("plus_x_to_plus_y", "plus_x", "plus_y", "Rz_pos_90"),
    SymmetryRelation("plus_y_to_minus_x", "plus_y", "minus_x", "Rz_pos_90"),
    SymmetryRelation("plus_x_to_minus_y", "plus_x", "minus_y", "Rz_neg_90"),
)


def run_c1_b4(config, baseline=None):
    if not isinstance(config, C1B4RunConfig):
        raise TypeError("config must be C1B4RunConfig")
    return _run_active_hover(
        config,
        B4_CASE_CORNERS[config.case],
        baseline=baseline,
    )


def _corner_from_tendon_name(tendon_name):
    matches = [
        corner
        for corner in CORNERS
        if tendon_name.endswith(f"_{corner}_tendon")
    ]
    if len(matches) != 1:
        raise AssertionError(f"cannot resolve tendon corner: {tendon_name}")
    return matches[0]


def manifest_corner_order(trace):
    return tuple(_corner_from_tendon_name(item.tendon_name) for item in trace.bindings)


def manifest_permutation(reference, candidate, transform):
    if not isinstance(transform, TransformSpec):
        raise TypeError("transform must be TransformSpec")
    candidate_by_corner = {
        _corner_from_tendon_name(binding.tendon_name): index
        for index, binding in enumerate(candidate.bindings)
    }
    if set(candidate_by_corner) != set(CORNERS):
        raise AssertionError("candidate manifest does not contain four unique corners")
    permutation = [
        candidate_by_corner[transform.corner_map[corner]]
        for corner in manifest_corner_order(reference)
    ]
    if sorted(permutation) != list(range(len(CORNERS))):
        raise AssertionError(f"manifest permutation is not bijective: {permutation}")
    return _freeze(permutation, dtype=int)


def validate_rotation_matrix(matrix, *, atol=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite")
    orthogonal_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    if orthogonal_error > atol or abs(determinant - 1.0) > atol:
        raise ValueError(
            "matrix is not a proper rotation: "
            f"orthogonal_error={orthogonal_error}, determinant={determinant}"
        )
    return matrix


def rotation_matrix_to_quaternion(matrix):
    matrix = validate_rotation_matrix(matrix)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix.reshape(9))
    return _freeze(_normalized_quaternion(quaternion))


def quaternion_geodesic(left, right):
    relative = _relative_quaternion(left, right)
    angle = float(np.linalg.norm(_rotation_vector(relative)))
    if not math.isfinite(angle):
        raise ValueError("SO(3) geodesic is non-finite")
    tolerance = 64 * np.finfo(float).eps
    if angle < -tolerance or angle > math.pi + tolerance:
        raise ValueError(f"SO(3) geodesic outside [0, pi]: {angle}")
    return min(math.pi, max(0.0, angle))


def so3_geodesic(left, right):
    return quaternion_geodesic(
        rotation_matrix_to_quaternion(left),
        rotation_matrix_to_quaternion(right),
    )


def transformed_rotation(matrix, transform):
    matrix = validate_rotation_matrix(matrix)
    expected = transform.matrix @ matrix @ transform.matrix.T
    return _freeze(validate_rotation_matrix(expected))


def _quaternion_matrices(quaternions):
    result = []
    for quaternion in np.asarray(quaternions):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        result.append(matrix.reshape(3, 3))
    return _freeze(result)


def joint_rotation_matrices(trace):
    samples = [
        _quaternion_matrices(quaternions)
        for quaternions in trace.relative["joint_quaternion"]
    ]
    return _freeze(samples)


def composite_rotation_matrices(trace):
    return _quaternion_matrices(trace.relative["composite_quaternion"])


def body_linear_velocity(trace):
    return _freeze(
        np.einsum(
            "nji,nj->ni",
            trace.control["uav_rotation_matrix"],
            trace.control["uav_linear_velocity_world_m_s"],
        )
    )


def per_channel_power(trace, *, physics=False):
    values = trace.physics if physics else trace.control
    return _freeze(-values["actuator_force"] * values["tendon_velocity"])


def direction_anchor(trace):
    if trace.config.case not in DIRECTION_PRIMARY_AXIS:
        raise ValueError("direction anchor requires a directional case")
    axis, sign, cross_axis = DIRECTION_PRIMARY_AXIS[trace.config.case]
    vector = trace.relative["root_to_tip_uav_body_m"][RELEASE_TICK]
    primary = float(sign * vector[axis])
    leakage = abs(float(vector[cross_axis]))
    leakage_limit = mixed_limit(
        ORTHOGONAL_LEAKAGE_ABSOLUTE_FLOOR_M,
        ORTHOGONAL_LEAKAGE_RELATIVE_FRACTION,
        abs(primary),
    )
    return DirectionAnchor(
        case=trace.config.case,
        sim_freq=trace.config.sim_freq,
        release_vector_m=tuple(float(value) for value in vector),
        signed_primary_m=primary,
        primary_floor_m=RELEASE_PRIMARY_ACTIVITY_FLOOR_M,
        orthogonal_leakage_m=leakage,
        orthogonal_limit_m=leakage_limit,
        passed=(
            primary >= RELEASE_PRIMARY_ACTIVITY_FLOOR_M
            and leakage <= leakage_limit
        ),
    )


def direction_activity(trace):
    if trace.config.case not in DIRECTION_CASES:
        raise ValueError("activity metrics require a directional case")
    mask = trace.control["time"] <= FORCED_END_S + 1e-12
    relative = trace.relative
    actual = {
        "uav_position_m": float(
            np.max(np.linalg.norm(relative["uav_position_world_m"][mask], axis=1))
        ),
        "uav_attitude_rad": float(
            np.max(relative["uav_quaternion_geodesic_rad"][mask])
        ),
        "tip_body_m": float(
            np.max(
                np.linalg.norm(relative["root_to_tip_uav_body_m"][mask], axis=1)
            )
        ),
        "joint_rotation_rad": float(
            np.max(
                np.linalg.norm(
                    relative["joint_rotation_vector_rad"][mask], axis=-1
                )
            )
        ),
        "composite_rotation_rad": float(
            np.max(
                np.linalg.norm(
                    relative["composite_rotation_vector_rad"][mask], axis=1
                )
            )
        ),
        "tendon_excursion_m": float(
            np.max(np.abs(relative["tendon_excursion_m"][mask]))
        ),
    }
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "activity": value,
                    "limit": DIRECTION_ACTIVITY_FLOORS[name],
                    "passed": value >= DIRECTION_ACTIVITY_FLOORS[name],
                }
            )
            for name, value in actual.items()
        }
    )


def _window_mask(trace, window):
    if window not in WINDOW_END_S:
        raise ValueError(f"unknown C1-B4 window: {window}")
    return trace.control["time"] <= WINDOW_END_S[window] + 1e-12


def _metric(
    reference,
    window,
    field,
    activity,
    error,
    absolute_floor,
    unit,
    *,
    relative_fraction=None,
):
    if relative_fraction is None:
        relative_fraction = WINDOW_RELATIVE_FRACTION[window]
    limit = mixed_limit(absolute_floor, relative_fraction, activity)
    return AcceptanceMetric(
        case=reference.config.case,
        sim_freq=reference.config.sim_freq,
        window=window,
        field=field,
        activity=float(activity),
        error=float(error),
        absolute_floor=float(absolute_floor),
        relative_fraction=float(relative_fraction),
        limit=float(limit),
        unit=unit,
        passed=bool(error <= limit),
    )


def _vector_metric(
    reference,
    window,
    field,
    reference_values,
    candidate_values,
    matrix,
    absolute_floor,
    unit,
    *,
    excursion=False,
):
    mask = _window_mask(reference, window)
    left = np.asarray(reference_values)[mask]
    right = np.asarray(candidate_values)[mask]
    expected = left @ np.asarray(matrix).T
    error = float(np.max(np.linalg.norm(right - expected, axis=-1)))
    scale_values = left - left[0] if excursion else left
    activity = float(np.max(np.linalg.norm(scale_values, axis=-1)))
    return _metric(
        reference, window, field, activity, error, absolute_floor, unit
    )


def _channel_metric(
    reference,
    candidate,
    transform,
    window,
    field,
    reference_values,
    candidate_values,
    absolute_floor,
    unit,
    *,
    excursion=False,
    relative_fraction=None,
):
    mask = _window_mask(reference, window)
    permutation = manifest_permutation(reference, candidate, transform)
    left = np.asarray(reference_values)[mask]
    right = np.asarray(candidate_values)[mask][:, permutation]
    error = float(np.max(np.abs(right - left)))
    scale_values = left - left[0] if excursion else left
    activity = float(np.max(np.abs(scale_values)))
    return _metric(
        reference,
        window,
        field,
        activity,
        error,
        absolute_floor,
        unit,
        relative_fraction=relative_fraction,
    )


def _scalar_metric(
    reference,
    window,
    field,
    reference_values,
    candidate_values,
    absolute_floor,
    unit,
    *,
    excursion=False,
):
    mask = _window_mask(reference, window)
    left = np.asarray(reference_values)[mask]
    right = np.asarray(candidate_values)[mask]
    error = float(np.max(np.abs(right - left)))
    scale_values = left - left[0] if excursion else left
    activity = float(np.max(np.abs(scale_values)))
    return _metric(
        reference, window, field, activity, error, absolute_floor, unit
    )


def _so3_metric(
    reference,
    window,
    field,
    reference_matrices,
    candidate_matrices,
    transform,
):
    mask = _window_mask(reference, window)
    left = np.asarray(reference_matrices)[mask]
    right = np.asarray(candidate_matrices)[mask]
    errors = []
    activity = []
    identity = np.eye(3)
    for left_matrix, right_matrix in zip(
        left.reshape((-1, 3, 3)), right.reshape((-1, 3, 3))
    ):
        expected = transformed_rotation(left_matrix, transform)
        errors.append(so3_geodesic(right_matrix, expected))
        activity.append(so3_geodesic(left_matrix, identity))
    return _metric(
        reference,
        window,
        field,
        max(activity, default=0.0),
        max(errors, default=0.0),
        5e-7,
        "rad",
    )


def symmetry_metrics(reference, candidate, transform, window):
    if reference.config.sim_freq != candidate.config.sim_freq:
        raise ValueError("symmetry traces must use the same simulation frequency")
    if reference.config.case not in DIRECTION_CASES:
        raise ValueError("reference must be a directional trace")
    control_left, control_right = reference.control, candidate.control
    relative_left, relative_right = reference.relative, candidate.relative
    polar = transform.matrix
    axial = transform.axial_matrix
    metrics = {}

    vector_specs = (
        ("uav_position_world_m", relative_left["uav_position_world_m"], relative_right["uav_position_world_m"], polar, 1e-9, "m", False),
        ("uav_linear_velocity_world_m_s", relative_left["uav_linear_velocity_world_m_s"], relative_right["uav_linear_velocity_world_m_s"], polar, 1e-8, "m/s", False),
        ("root_to_tip_uav_body_m", relative_left["root_to_tip_uav_body_m"], relative_right["root_to_tip_uav_body_m"], polar, 1e-9, "m", False),
        ("tip_minus_uav_world_m", relative_left["tip_minus_uav_world_m"], relative_right["tip_minus_uav_world_m"], polar, 1e-9, "m", False),
        ("body_linear_velocity_m_s", body_linear_velocity(reference), body_linear_velocity(candidate), polar, 1e-8, "m/s", False),
        ("system_com_world_m", control_left["system_com_world_m"], control_right["system_com_world_m"], polar, 1e-9, "m", True),
        ("uav_inertial_com_world_m", control_left["uav_inertial_com_world_m"], control_right["uav_inertial_com_world_m"], polar, 1e-9, "m", True),
        ("root_world_m", control_left["root_world_m"], control_right["root_world_m"], polar, 1e-9, "m", True),
        ("tip_world_m", control_left["tip_world_m"], control_right["tip_world_m"], polar, 1e-9, "m", True),
        ("uav_rotation_vector_rad", relative_left["uav_rotation_vector_rad"], relative_right["uav_rotation_vector_rad"], axial, 5e-7, "rad", False),
        ("joint_rotation_vector_rad", relative_left["joint_rotation_vector_rad"], relative_right["joint_rotation_vector_rad"], axial, 5e-7, "rad", False),
        ("composite_rotation_vector_rad", relative_left["composite_rotation_vector_rad"], relative_right["composite_rotation_vector_rad"], axial, 5e-7, "rad", False),
        ("uav_angular_velocity_world_rad_s", control_left["uav_angular_velocity_world_rad_s"], control_right["uav_angular_velocity_world_rad_s"], axial, 1e-7, "rad/s", False),
        ("system_linear_momentum_ns", control_left["system_linear_momentum_ns"], control_right["system_linear_momentum_ns"], polar, 1e-10, "N*s", False),
        ("p_uav_ns", control_left["p_uav_ns"], control_right["p_uav_ns"], polar, 1e-10, "N*s", False),
        ("p_arm_ns", control_left["p_arm_ns"], control_right["p_arm_ns"], polar, 1e-10, "N*s", False),
        ("system_angular_momentum_com_nms", control_left["system_angular_momentum_com_nms"], control_right["system_angular_momentum_com_nms"], axial, 1e-12, "N*m*s", False),
        ("h_uav_com_nms", control_left["h_uav_com_nms"], control_right["h_uav_com_nms"], axial, 1e-12, "N*m*s", False),
        ("h_arm_com_nms", control_left["h_arm_com_nms"], control_right["h_arm_com_nms"], axial, 1e-12, "N*m*s", False),
        ("rotor_force_world_n", control_left["rotor_force_world_n"], control_right["rotor_force_world_n"], polar, 1e-10, "N", True),
        ("rotor_torque_world_nm", control_left["rotor_torque_world_nm"], control_right["rotor_torque_world_nm"], axial, 1e-12, "N*m", False),
    )
    for field, left, right, matrix, floor, unit, excursion in vector_specs:
        metrics[field] = _vector_metric(
            reference,
            window,
            field,
            left,
            right,
            matrix,
            floor,
            unit,
            excursion=excursion,
        )

    metrics["uav_so3_geodesic_rad"] = _so3_metric(
        reference,
        window,
        "uav_so3_geodesic_rad",
        relative_left["uav_rotation_matrix"],
        relative_right["uav_rotation_matrix"],
        transform,
    )
    metrics["joint_so3_geodesic_rad"] = _so3_metric(
        reference,
        window,
        "joint_so3_geodesic_rad",
        joint_rotation_matrices(reference),
        joint_rotation_matrices(candidate),
        transform,
    )
    metrics["composite_so3_geodesic_rad"] = _so3_metric(
        reference,
        window,
        "composite_so3_geodesic_rad",
        composite_rotation_matrices(reference),
        composite_rotation_matrices(candidate),
        transform,
    )

    channel_specs = (
        ("command_n", control_left["command"], control_right["command"], 1e-15, "N", False, 0.0),
        ("actuator_force_n", control_left["actuator_force"], control_right["actuator_force"], 1e-15, "N", False, 0.0),
        ("tendon_length_m", control_left["tendon_length"], control_right["tendon_length"], 1e-10, "m", True, None),
        ("tendon_velocity_m_s", control_left["tendon_velocity"], control_right["tendon_velocity"], 1e-8, "m/s", False, None),
        ("tendon_excursion_m", relative_left["tendon_excursion_m"], relative_right["tendon_excursion_m"], 1e-10, "m", False, None),
        ("per_channel_power_w", per_channel_power(reference), per_channel_power(candidate), 1e-12, "W", False, None),
    )
    for field, left, right, floor, unit, excursion, fraction in channel_specs:
        metrics[field] = _channel_metric(
            reference,
            candidate,
            transform,
            window,
            field,
            left,
            right,
            floor,
            unit,
            excursion=excursion,
            relative_fraction=fraction,
        )

    scalar_specs = (
        ("mechanical_energy_j", control_left["mechanical_energy_j"], control_right["mechanical_energy_j"], True),
        ("tendon_work_j", control_left["tendon_work_j"], control_right["tendon_work_j"], False),
        ("rotor_work_j", control_left["rotor_work_j"], control_right["rotor_work_j"], False),
        ("damping_dissipation_j", control_left["damping_dissipation_j"], control_right["damping_dissipation_j"], False),
        ("ledger_residual_j", control_left["ledger_residual_j"], control_right["ledger_residual_j"], False),
        (
            "rotor_generalized_power_w",
            np.einsum(
                "ni,ni->n",
                control_left["rotor_generalized_force"],
                control_left["qvel"],
            ),
            np.einsum(
                "ni,ni->n",
                control_right["rotor_generalized_force"],
                control_right["qvel"],
            ),
            False,
        ),
    )
    for field, left, right, excursion in scalar_specs:
        unit = "W" if field.endswith("power_w") else "J"
        floor = 1e-12 if unit == "W" else 1e-9
        metrics[field] = _scalar_metric(
            reference,
            window,
            field,
            left,
            right,
            floor,
            unit,
            excursion=excursion,
        )
    return MappingProxyType(metrics)


def signed_tendon_velocity_extrema_metric(
    reference, candidate, transform, window
):
    end = WINDOW_END_S[window]
    left_mask = reference.physics["time"] <= end + 1e-12
    right_mask = candidate.physics["time"] <= end + 1e-12
    permutation = manifest_permutation(reference, candidate, transform)
    left = reference.physics["tendon_velocity"][left_mask]
    right = candidate.physics["tendon_velocity"][right_mask][:, permutation]
    left_extrema = np.stack((left.min(axis=0), left.max(axis=0)))
    right_extrema = np.stack((right.min(axis=0), right.max(axis=0)))
    error = float(np.max(np.abs(right_extrema - left_extrema)))
    activity = float(np.max(np.abs(left)))
    return _metric(
        reference,
        window,
        "tendon_velocity_signed_extrema_m_s",
        activity,
        error,
        1e-8,
        "m/s",
    )


def common_mode_metrics(trace):
    if trace.config.case != "common_mode_max":
        raise ValueError("common-mode metrics require common_mode_max")
    relative = trace.relative
    hold = slice(60, RELEASE_TICK)
    return MappingProxyType(
        {
            "minimum_hold_command_n": float(np.min(trace.control["command"][hold])),
            "maximum_hold_command_n": float(np.max(trace.control["command"][hold])),
            "maximum_force_spread_n": float(
                np.max(np.ptp(trace.control["actuator_force"], axis=1))
            ),
            "maximum_lateral_uav_displacement_m": float(
                np.max(np.linalg.norm(relative["uav_position_world_m"][:, :2], axis=1))
            ),
            "maximum_lateral_tip_body_m": float(
                np.max(
                    np.linalg.norm(
                        relative["root_to_tip_uav_body_m"][:, :2], axis=1
                    )
                )
            ),
            "maximum_bend_rad": float(
                np.max(
                    np.linalg.norm(
                        relative["composite_rotation_vector_rad"][:, :2], axis=1
                    )
                )
            ),
            "maximum_twist_rad": float(
                np.max(np.abs(relative["composite_rotation_vector_rad"][:, 2]))
            ),
            "maximum_tendon_excursion_m": float(
                np.max(np.abs(relative["tendon_excursion_m"]))
            ),
            "maximum_tendon_channel_spread_m": float(
                np.max(np.ptp(relative["tendon_excursion_m"], axis=1))
            ),
        }
    )


def prove_local_frame_topology(sim_freq=2400):
    config = C1B4RunConfig(sim_freq, "baseline")
    env = None
    try:
        env = _make_active_production_env(config)
        model = env.model
        manifest = env.SCALED_STABLESOFT_ARM_MANIFEST
        body_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in manifest.segment_body_names
            ],
            dtype=int,
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in manifest.ball_joint_names
            ],
            dtype=int,
        )
        manifest_corners = tuple(
            _corner_from_tendon_name(name) for name in manifest.tendon_names
        )
        damping_spread = 0.0
        for joint_id in joint_ids:
            address = int(model.jnt_dofadr[joint_id])
            damping_spread = max(
                damping_spread,
                float(np.ptp(model.dof_damping[address:address + 3])),
            )
        routing_signs_match = True
        for site_name in manifest.routing_site_names:
            corner = next(
                (name for name in CORNERS if f"_{name}_" in site_name),
                None,
            )
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            x_sign, y_sign = CORNER_COORDINATES[corner]
            position = model.site_pos[site_id]
            routing_signs_match = routing_signs_match and (
                np.sign(position[0]) == x_sign
                and np.sign(position[1]) == y_sign
                and abs(abs(position[0]) - abs(position[1])) <= 1e-15
            )
        actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in manifest.tendon_actuator_names
            ],
            dtype=int,
        )
        return LocalFrameTopologyProof(
            manifest_corners=manifest_corners,
            maximum_body_quaternion_identity_error=float(
                np.max(
                    np.abs(
                        model.body_quat[body_ids]
                        - np.array([1.0, 0.0, 0.0, 0.0])
                    )
                )
            ),
            maximum_body_xy_offset_m=float(
                np.max(np.abs(model.body_pos[body_ids, :2]))
            ),
            maximum_inertia_xy_difference=float(
                np.max(
                    np.abs(
                        model.body_inertia[body_ids, 0]
                        - model.body_inertia[body_ids, 1]
                    )
                )
            ),
            maximum_joint_anchor_error_m=float(
                np.max(np.abs(model.jnt_pos[joint_ids]))
            ),
            maximum_joint_damping_axis_spread=damping_spread,
            joint_stiffness_values=tuple(
                float(value) for value in model.jnt_stiffness[joint_ids]
            ),
            routing_corner_signs_match=bool(routing_signs_match),
            actuator_control_ranges_match=bool(
                np.allclose(
                    model.actuator_ctrlrange[actuator_ids],
                    np.tile([0.0, MAX_TENSION_N], (len(actuator_ids), 1)),
                    rtol=0.0,
                    atol=1e-15,
                )
            ),
            actuator_force_limits_disabled=bool(
                not np.any(model.actuator_forcelimited[actuator_ids])
            ),
        )
    finally:
        if env is not None:
            env.close()


def metric_diagnostics(metric, relation=None):
    return {
        "relation": None if relation is None else relation.name,
        "case": metric.case,
        "sim_freq": metric.sim_freq,
        "window": metric.window,
        "field": metric.field,
        "activity": metric.activity,
        "error": metric.error,
        "absolute_floor": metric.absolute_floor,
        "relative_fraction": metric.relative_fraction,
        "limit": metric.limit,
        "unit": metric.unit,
        "passed": metric.passed,
    }
