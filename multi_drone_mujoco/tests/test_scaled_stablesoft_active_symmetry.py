"""C1-B4 active UAV-arm direction symmetry acceptance tests."""

from dataclasses import FrozenInstanceError
import math
from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_active_symmetry import (
    B4_CASE_CORNERS,
    B4_CASE_TENSION_N,
    B4_FORMAL_CASES,
    COMMON_MODE_ANGLE_LIMIT_RAD,
    COMMON_MODE_DISPLACEMENT_LIMIT_M,
    COMMON_MODE_FORCE_SPREAD_LIMIT_N,
    COMMON_MODE_TENDON_LIMIT_M,
    CORNERS,
    DIRECTION_CASES,
    RELEASE_TICK,
    SIM_FREQUENCIES,
    SYMMETRY_RELATIONS,
    TRANSFORMS,
    WINDOW_END_S,
    C1B4RunConfig,
    active_hover_convergence_metrics,
    common_mode_metrics,
    direction_activity,
    direction_anchor,
    manifest_corner_order,
    manifest_permutation,
    metric_diagnostics,
    prove_local_frame_topology,
    quaternion_geodesic,
    run_c1_b4,
    signed_tendon_velocity_extrema_metric,
    so3_geodesic,
    symmetry_metrics,
    transformed_rotation,
    validate_rotation_matrix,
)


EXPECTED_PERMUTATIONS = MappingProxyType(
    {
        "Sx": (2, 3, 0, 1),
        "Sy": (1, 0, 3, 2),
        "Rz_pos_90": (2, 0, 3, 1),
        "Rz_neg_90": (1, 3, 0, 2),
        "Rz_180": (3, 2, 1, 0),
    }
)


@pytest.fixture(scope="module")
def c1_b4_results():
    results = {}
    for sim_freq in SIM_FREQUENCIES:
        baseline = run_c1_b4(C1B4RunConfig(sim_freq, "baseline"))
        results[(sim_freq, "baseline")] = baseline
        for case in DIRECTION_CASES:
            results[(sim_freq, case)] = run_c1_b4(
                C1B4RunConfig(sim_freq, case), baseline=baseline
            )
        results[(sim_freq, "common_mode_max")] = run_c1_b4(
            C1B4RunConfig(
                sim_freq,
                "common_mode_max",
                tension_n=B4_CASE_TENSION_N["common_mode_max"],
            ),
            baseline=baseline,
        )
    assert len(results) == 12
    return MappingProxyType(results)


@pytest.fixture(scope="module")
def local_frame_proof():
    return prove_local_frame_topology(2400)


@pytest.fixture(scope="module")
def symmetry_results(c1_b4_results):
    results = {}
    for sim_freq in SIM_FREQUENCIES:
        for relation in SYMMETRY_RELATIONS:
            reference = c1_b4_results[(sim_freq, relation.reference_case)]
            candidate = c1_b4_results[(sim_freq, relation.candidate_case)]
            transform = TRANSFORMS[relation.transform_name]
            for window in WINDOW_END_S:
                results[(sim_freq, relation.name, window)] = MappingProxyType(
                    {
                        "trajectory": symmetry_metrics(
                            reference, candidate, transform, window
                        ),
                        "signed_extrema": signed_tendon_velocity_extrema_metric(
                            reference, candidate, transform, window
                        ),
                    }
                )
    return MappingProxyType(results)


def _relation(name):
    return next(item for item in SYMMETRY_RELATIONS if item.name == name)


def _metric_failure(metric, relation):
    return metric_diagnostics(metric, relation)


def _axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(
        quaternion, axis, float(angle)
    )
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return quaternion, matrix.reshape(3, 3)


def test_c1_b4_config_and_formal_matrix_are_independent_from_c1_b3():
    assert B4_FORMAL_CASES == (
        "baseline",
        "plus_x",
        "minus_x",
        "plus_y",
        "minus_y",
        "common_mode_max",
    )
    assert B4_CASE_CORNERS["plus_x"] == ("xp_yp", "xp_yn")
    assert B4_CASE_CORNERS["minus_x"] == ("xn_yp", "xn_yn")
    assert B4_CASE_CORNERS["plus_y"] == ("xp_yp", "xn_yp")
    assert B4_CASE_CORNERS["minus_y"] == ("xp_yn", "xn_yn")
    assert B4_CASE_CORNERS["common_mode_max"] == CORNERS
    assert B4_CASE_TENSION_N["common_mode_max"] == 0.01
    with pytest.raises(ValueError):
        C1B4RunConfig(2400, "single_xp_yp")
    with pytest.raises(ValueError):
        C1B4RunConfig(9600, "plus_x")
    with pytest.raises(ValueError):
        C1B4RunConfig(2400, "common_mode_max")


def test_c1_b4_fixture_runs_exactly_twelve_named_traces(c1_b4_results):
    assert len(c1_b4_results) == 12
    assert set(c1_b4_results) == {
        (sim_freq, case)
        for sim_freq in SIM_FREQUENCIES
        for case in B4_FORMAL_CASES
    }
    assert not {
        case for _, case in c1_b4_results if case in {"single_xp_yp", "D1", "D2"}
    }
    for key, trace in c1_b4_results.items():
        assert key == (trace.config.sim_freq, trace.config.case)
        assert isinstance(trace.config, C1B4RunConfig)
        assert trace.config.tension_n == B4_CASE_TENSION_N[trace.config.case]


def test_c1_b4_local_joint_frames_and_actuator_ranges_prove_symmetry_basis(
    local_frame_proof,
):
    assert local_frame_proof.passed, local_frame_proof
    assert local_frame_proof.manifest_corners == CORNERS
    assert local_frame_proof.joint_stiffness_values == pytest.approx(
        (0.003086,) * 8, abs=1e-15
    )


@pytest.mark.parametrize("transform_name", tuple(EXPECTED_PERMUTATIONS))
def test_c1_b4_manifest_permutations_are_name_resolved_bijections(
    c1_b4_results, transform_name
):
    trace = c1_b4_results[(2400, "baseline")]
    transform = TRANSFORMS[transform_name]
    permutation = manifest_permutation(trace, trace, transform)

    assert manifest_corner_order(trace) == CORNERS
    assert tuple(permutation) == EXPECTED_PERMUTATIONS[transform_name]
    assert sorted(permutation.tolist()) == list(range(4))
    assert not permutation.flags.writeable


def test_c1_b4_transform_group_inverse_and_chain_contracts():
    identity = {corner: corner for corner in CORNERS}

    def compose(left, right):
        return {
            corner: left.corner_map[right.corner_map[corner]]
            for corner in CORNERS
        }

    sx, sy = TRANSFORMS["Sx"], TRANSFORMS["Sy"]
    positive, negative = TRANSFORMS["Rz_pos_90"], TRANSFORMS["Rz_neg_90"]
    assert compose(sx, sx) == identity
    assert compose(sy, sy) == identity
    assert compose(positive, negative) == identity
    current = identity
    for _ in range(4):
        current = {
            corner: positive.corner_map[current[corner]] for corner in CORNERS
        }
    assert current == identity
    assert positive.corner_map["xp_yp"] == "xn_yp"
    assert positive.corner_map["xn_yp"] == "xn_yn"


def test_c1_b4_transform_metadata_is_orthogonal_and_immutable():
    for transform in TRANSFORMS.values():
        np.testing.assert_allclose(
            transform.matrix.T @ transform.matrix,
            np.eye(3),
            rtol=0.0,
            atol=1e-15,
        )
        assert abs(abs(transform.determinant) - 1.0) <= 1e-15
        assert not transform.matrix.flags.writeable
        assert not transform.axial_matrix.flags.writeable
        with pytest.raises(ValueError):
            transform.matrix[0, 0] = 2.0


def test_c1_b4_so3_identity_and_quaternion_double_cover_are_zero():
    identity = np.eye(3)
    quaternion = np.array([0.9, 0.1, -0.2, 0.3])
    quaternion /= np.linalg.norm(quaternion)

    assert so3_geodesic(identity, identity) <= 1e-15
    assert quaternion_geodesic(quaternion, -quaternion) <= 1e-15


@pytest.mark.parametrize("angle", (1e-8, 0.125, math.pi - 1e-9))
def test_c1_b4_so3_known_angles_are_finite_stable_and_bounded(angle):
    _, rotation = _axis_angle_matrix([1.0, 2.0, -3.0], angle)
    distance = so3_geodesic(rotation, np.eye(3))

    assert math.isfinite(distance)
    assert 0.0 <= distance <= math.pi
    assert distance == pytest.approx(angle, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("transform_name", tuple(TRANSFORMS))
def test_c1_b4_conjugated_orientation_is_proper_and_has_zero_geodesic(
    transform_name,
):
    _, rotation = _axis_angle_matrix([0.3, -0.5, 0.8], 0.7)
    expected = transformed_rotation(rotation, TRANSFORMS[transform_name])

    validate_rotation_matrix(expected)
    np.testing.assert_allclose(expected.T @ expected, np.eye(3), atol=1e-12)
    assert np.linalg.det(expected) == pytest.approx(1.0, abs=1e-12)
    assert so3_geodesic(expected, expected) <= 1e-15


def test_c1_b4_improper_reflection_cannot_enter_so3_directly():
    with pytest.raises(ValueError, match="proper rotation"):
        validate_rotation_matrix(TRANSFORMS["Sx"].matrix)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", B4_FORMAL_CASES)
def test_c1_b4_manifest_command_ramp_hold_release_is_exact(
    c1_b4_results, sim_freq, case
):
    trace = c1_b4_results[(sim_freq, case)]
    control = trace.control
    by_corner = {
        corner: index
        for index, corner in enumerate(manifest_corner_order(trace))
    }
    expected = np.zeros_like(control["command"])
    active_corners = B4_CASE_CORNERS[case]
    for tick in range(RELEASE_TICK):
        scale = (
            1.0
            if tick >= 60
            else 0.5 * (1.0 - np.cos(np.pi * tick / 60.0))
        )
        for corner in active_corners:
            expected[tick, by_corner[corner]] = trace.config.tension_n * scale

    np.testing.assert_allclose(control["command"], expected, atol=1e-15)
    np.testing.assert_allclose(control["actuator_force"], expected, atol=1e-15)
    np.testing.assert_array_equal(control["command"][RELEASE_TICK:], 0.0)
    np.testing.assert_allclose(control["time"], control["tick"] / 240.0, atol=1e-12)


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", DIRECTION_CASES)
def test_c1_b4_absolute_body_tip_direction_anchor_has_sign_and_low_leakage(
    c1_b4_results, sim_freq, case
):
    anchor = direction_anchor(c1_b4_results[(sim_freq, case)])

    assert anchor.passed, anchor
    assert anchor.signed_primary_m >= anchor.primary_floor_m
    assert anchor.orthogonal_leakage_m <= anchor.orthogonal_limit_m


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", DIRECTION_CASES)
def test_c1_b4_directional_activity_prevents_zero_response_false_pass(
    c1_b4_results, sim_freq, case
):
    metrics = direction_activity(c1_b4_results[(sim_freq, case)])

    assert all(item["passed"] for item in metrics.values()), {
        "sim_freq": sim_freq,
        "case": case,
        "metrics": {key: dict(value) for key, value in metrics.items()},
    }


@pytest.mark.parametrize("window", tuple(WINDOW_END_S))
@pytest.mark.parametrize("relation_name", tuple(item.name for item in SYMMETRY_RELATIONS))
@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b4_same_frequency_polar_axial_so3_scalar_and_channel_symmetry(
    symmetry_results, sim_freq, relation_name, window
):
    relation = _relation(relation_name)
    result = symmetry_results[(sim_freq, relation_name, window)]
    failures = [
        _metric_failure(metric, relation)
        for metric in result["trajectory"].values()
        if not metric.passed
    ]
    if not result["signed_extrema"].passed:
        failures.append(_metric_failure(result["signed_extrema"], relation))

    assert not failures, failures


@pytest.mark.parametrize("relation_name", tuple(item.name for item in SYMMETRY_RELATIONS))
@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b4_uav_joint_and_composite_so3_use_matrix_conjugation(
    symmetry_results, sim_freq, relation_name
):
    metrics = symmetry_results[(sim_freq, relation_name, "full")]["trajectory"]
    names = (
        "uav_so3_geodesic_rad",
        "joint_so3_geodesic_rad",
        "composite_so3_geodesic_rad",
    )
    assert all(metrics[name].passed for name in names), {
        name: metric_diagnostics(metrics[name], _relation(relation_name))
        for name in names
    }


@pytest.mark.parametrize("relation_name", tuple(item.name for item in SYMMETRY_RELATIONS))
@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b4_tendon_permutation_uses_one_unsigned_channel_mapping(
    symmetry_results, sim_freq, relation_name
):
    metrics = symmetry_results[(sim_freq, relation_name, "full")]["trajectory"]
    names = (
        "command_n",
        "actuator_force_n",
        "tendon_length_m",
        "tendon_velocity_m_s",
        "tendon_excursion_m",
        "per_channel_power_w",
    )
    assert all(metrics[name].passed for name in names), {
        name: metric_diagnostics(metrics[name], _relation(relation_name))
        for name in names
    }


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
def test_c1_b4_maximum_common_mode_is_structural_zero_and_fully_tensioned(
    c1_b4_results, local_frame_proof, sim_freq
):
    trace = c1_b4_results[(sim_freq, "common_mode_max")]
    metrics = common_mode_metrics(trace)

    assert metrics["minimum_hold_command_n"] == pytest.approx(0.01, abs=1e-15)
    assert metrics["maximum_hold_command_n"] == pytest.approx(0.01, abs=1e-15)
    assert metrics["maximum_force_spread_n"] <= COMMON_MODE_FORCE_SPREAD_LIMIT_N
    assert (
        metrics["maximum_lateral_uav_displacement_m"]
        <= COMMON_MODE_DISPLACEMENT_LIMIT_M
    )
    assert (
        metrics["maximum_lateral_tip_body_m"]
        <= COMMON_MODE_DISPLACEMENT_LIMIT_M
    )
    assert metrics["maximum_bend_rad"] <= COMMON_MODE_ANGLE_LIMIT_RAD
    assert metrics["maximum_twist_rad"] <= COMMON_MODE_ANGLE_LIMIT_RAD
    assert metrics["maximum_tendon_excursion_m"] <= COMMON_MODE_TENDON_LIMIT_M
    assert (
        metrics["maximum_tendon_channel_spread_m"]
        <= COMMON_MODE_TENDON_LIMIT_M
    )
    assert local_frame_proof.actuator_control_ranges_match
    assert local_frame_proof.actuator_force_limits_disabled


@pytest.mark.parametrize("case", B4_FORMAL_CASES)
def test_c1_b4_2400_converges_to_4800_reference(c1_b4_results, case):
    candidate = c1_b4_results[(2400, case)]
    reference = c1_b4_results[(4800, case)]
    metrics = active_hover_convergence_metrics(candidate, reference)

    assert metrics["passed"], {
        "case": case,
        "candidate_frequency_hz": 2400,
        "reference_frequency_hz": 4800,
        "metrics": dict(metrics),
    }


@pytest.mark.parametrize("sim_freq", SIM_FREQUENCIES)
@pytest.mark.parametrize("case", B4_FORMAL_CASES)
def test_c1_b4_physics_rate_safety_and_read_only_contract(
    c1_b4_results, sim_freq, case
):
    trace = c1_b4_results[(sim_freq, case)]
    safety = trace.safety
    diagnostics = {
        "case": case,
        "sim_freq": sim_freq,
        "status": trace.status,
        "safety": safety,
    }

    assert trace.status == "completed", diagnostics
    assert safety.first_failure is None, diagnostics
    assert safety.completed_steps == safety.requested_steps == 3 * sim_freq
    assert safety.maximum_contacts == 0
    assert safety.minimum_tendon_margin_m >= -1e-9, diagnostics
    assert safety.maximum_abs_qvel_native < 1000.0, diagnostics
    assert safety.maximum_abs_qacc_native < 2e5, diagnostics
    assert safety.maximum_force_command_error_n <= 1e-15
    assert np.all(trace.physics["warnings"] == 0)
    assert np.all(trace.physics["contacts"] == 0)
    for values in (
        *trace.physics.values(),
        *trace.control.values(),
        *trace.relative.values(),
    ):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values)), diagnostics


def test_c1_b4_config_trace_metrics_and_metadata_are_immutable(
    c1_b4_results, symmetry_results
):
    trace = c1_b4_results[(2400, "plus_x")]
    result = symmetry_results[(2400, "plus_x_to_minus_x", "forced")]
    metric = result["trajectory"]["uav_position_world_m"]

    with pytest.raises(FrozenInstanceError):
        trace.config.case = "minus_x"
    with pytest.raises((TypeError, AttributeError)):
        c1_b4_results[(2400, "new")] = trace
    with pytest.raises(FrozenInstanceError):
        metric.limit = 1.0
    with pytest.raises(TypeError):
        result["new"] = metric
