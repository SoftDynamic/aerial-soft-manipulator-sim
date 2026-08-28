"""Gate C2-C2-B fixed-root modal calibration and rank decision."""

import itertools

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_modal_coordinates import TendonModalCoordinates
from multi_drone_mujoco.tests._scaled_stablesoft_c2_c2_probe import (
    CORNERS,
    FORMAL_TENSIONS_N,
    VectorTensionProbeConfig,
    maximum_principal_angle_rad,
    response_matrix,
    retained_projector,
    run_vector_tension_probe,
    tail_mean_contraction,
)


FREQUENCIES = (2400, 4800)
MODAL_TENSION_N = 0.002
MODAL_COMMANDS = {
    "plus_x": (0.002, 0.002, 0.0, 0.0),
    "minus_x": (0.0, 0.0, 0.002, 0.002),
    "plus_y": (0.002, 0.0, 0.002, 0.0),
    "minus_y": (0.0, 0.002, 0.0, 0.002),
    "common": (0.002, 0.002, 0.002, 0.002),
    "checker_plus": (0.002, 0.0, 0.0, 0.002),
    "checker_minus": (0.0, 0.002, 0.002, 0.0),
}


@pytest.fixture(scope="module")
def calibration():
    zero = {}
    singles = {}
    modal = {}
    repeat = {}
    for frequency in FREQUENCIES:
        zero[frequency] = run_vector_tension_probe(
            VectorTensionProbeConfig(frequency, (0.0,) * 4)
        )
        for tension in FORMAL_TENSIONS_N:
            for index, corner in enumerate(CORNERS):
                command = [0.0] * 4
                command[index] = tension
                singles[frequency, tension, corner] = run_vector_tension_probe(
                    VectorTensionProbeConfig(frequency, tuple(command))
                )
        for name, command in MODAL_COMMANDS.items():
            modal[frequency, name] = run_vector_tension_probe(
                VectorTensionProbeConfig(frequency, command)
            )
        repeat[frequency] = run_vector_tension_probe(
            VectorTensionProbeConfig(frequency, (MODAL_TENSION_N, 0.0, 0.0, 0.0))
        )
    matrices = {
        (frequency, tension): response_matrix(
            zero[frequency],
            [singles[frequency, tension, corner] for corner in CORNERS],
            tension,
        )
        for frequency in FREQUENCIES
        for tension in FORMAL_TENSIONS_N
    }
    return {"zero": zero, "singles": singles, "modal": modal, "repeat": repeat, "matrices": matrices}


def _frequency_uncertainty(matrices):
    return max(
        np.linalg.norm(matrices[2400, tension] - matrices[4800, tension], 2)
        for tension in FORMAL_TENSIONS_N
    )


def test_all_calibration_probes_are_safe_and_preserve_formal_commands(calibration):
    traces = [*calibration["zero"].values(), *calibration["singles"].values(), *calibration["modal"].values(), *calibration["repeat"].values()]
    for trace in traces:
        assert trace.safety.warning_events == ()
        assert trace.safety.maximum_contacts == 0
        assert trace.safety.completed_physics_steps == trace.config.number_of_physics_steps
        assert trace.safety.minimum_tendon_range_margin_m >= -1e-9
        np.testing.assert_allclose(trace.commanded_tension_n[-1], trace.config.target_tension_n, rtol=0.0, atol=1e-15)


def test_repeated_single_column_runs_are_bitwise_deterministic(calibration):
    for frequency in FREQUENCIES:
        original = calibration["singles"][frequency, MODAL_TENSION_N, CORNERS[0]]
        repeated = calibration["repeat"][frequency]
        for left, right in ((original.tendon_length_m, repeated.tendon_length_m), (original.tendon_velocity_m_s, repeated.tendon_velocity_m_s), (original.tip_position_m, repeated.tip_position_m)):
            np.testing.assert_array_equal(left, right)


def test_retained_bending_subspace_is_frequency_and_working_point_invariant(calibration):
    matrices = calibration["matrices"]
    uncertainty = _frequency_uncertainty(matrices)
    decompositions = {key: retained_projector(matrix) for key, matrix in matrices.items()}
    minimum_gap = min(values[0][1] - values[0][2] for values in decompositions.values())
    angle_bound = 2.0 * uncertainty / minimum_gap
    assert minimum_gap > uncertainty
    for singular_values, _ in decompositions.values():
        assert singular_values[1] > uncertainty
        assert abs(singular_values[1] / singular_values[0] - 1.0) <= uncertainty / singular_values[0]
        retained_ratio = singular_values[1] / singular_values[0]
        rejected_ratio = singular_values[2] / singular_values[1]
        assert retained_ratio - rejected_ratio > uncertainty / singular_values[0]
    for first, second in itertools.combinations(decompositions, 2):
        angle = maximum_principal_angle_rad(decompositions[first][1], decompositions[second][1])
        assert angle <= angle_bound


def test_third_chord_singular_value_is_quadratic_bending_shortening_not_a_common_mode(calibration):
    matrices = calibration["matrices"]
    for frequency in FREQUENCIES:
        sigma3 = np.asarray([retained_projector(matrices[frequency, tension])[0][2] for tension in FORMAL_TENSIONS_N])
        normalized = sigma3 / np.asarray(FORMAL_TENSIONS_N)
        np.testing.assert_allclose(normalized, normalized[0], rtol=0.003, atol=0.0)
        common = calibration["modal"][frequency, "common"]
        np.testing.assert_array_equal(common.contraction_m, 0.0)
        np.testing.assert_array_equal(common.tip_position_m - common.tip_position_m[0], 0.0)
        np.testing.assert_array_equal(common.composite_rotation_vectors_rad, 0.0)


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_plus_minus_x_y_have_correct_sign_symmetry_and_no_cross_axis(frequency, calibration):
    modal = calibration["modal"]
    transform = TendonModalCoordinates.from_tendon_names(modal[frequency, "plus_x"].tendon_names)
    values = {name: transform.tendon_to_modal(tail_mean_contraction(modal[frequency, name])) for name in ("plus_x", "minus_x", "plus_y", "minus_y")}
    assert values["plus_x"][1] > 0.0 and values["minus_x"][1] < 0.0
    assert values["plus_y"][2] > 0.0 and values["minus_y"][2] < 0.0
    np.testing.assert_allclose(values["plus_x"][1], -values["minus_x"][1], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(values["plus_y"][2], -values["minus_y"][2], rtol=0.0, atol=1e-15)
    assert values["plus_x"][2] == values["minus_x"][2] == 0.0
    assert values["plus_y"][1] == values["minus_y"][1] == 0.0


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_checker_uses_explicit_common_offset_and_has_no_resolved_twist(frequency, calibration):
    plus = calibration["modal"][frequency, "checker_plus"]
    minus = calibration["modal"][frequency, "checker_minus"]
    assert plus.config.target_tension_n == (0.002, 0.0, 0.0, 0.002)
    assert minus.config.target_tension_n == (0.0, 0.002, 0.002, 0.0)
    assert np.mean(plus.config.target_tension_n) == np.mean(minus.config.target_tension_n) == 0.001
    common_removed = 0.5 * (plus.contraction_m - minus.contraction_m)
    rotation_removed = 0.5 * (plus.composite_rotation_vectors_rad - minus.composite_rotation_vectors_rad)
    np.testing.assert_array_equal(common_removed, 0.0)
    np.testing.assert_array_equal(rotation_removed, 0.0)
    np.testing.assert_array_equal(plus.tip_position_m - plus.tip_position_m[0], 0.0)
    np.testing.assert_array_equal(minus.tip_position_m - minus.tip_position_m[0], 0.0)
