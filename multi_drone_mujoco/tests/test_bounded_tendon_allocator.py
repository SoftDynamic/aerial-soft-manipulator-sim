"""Gate C2-C2-E lexicographic bounded allocator contracts."""

import itertools

import numpy as np
import pytest

from multi_drone_mujoco.control.bounded_tendon_allocator import (
    BoundedTendonAllocator,
    BoundedTendonAllocatorConfig,
)
from multi_drone_mujoco.control.tendon_modal_coordinates import TendonModalCoordinates


CORNERS = ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
NAMES = tuple(f"arm_{corner}_tendon" for corner in CORNERS)
MAXIMUM = 0.01


def _allocator(bias=(0.0,) * 4, names=NAMES):
    return BoundedTendonAllocator(BoundedTendonAllocatorConfig(MAXIMUM, bias), TendonModalCoordinates.from_tendon_names(names))


def _project_diamond(effort):
    effort = np.asarray(effort, dtype=float)
    if np.sum(np.abs(effort)) <= 2 * MAXIMUM:
        return effort
    ordered = np.sort(np.abs(effort))[::-1]
    theta = (ordered.sum() - 2 * MAXIMUM) / 2 if ordered[1] > (ordered.sum() - 2 * MAXIMUM) / 2 else ordered[0] - 2 * MAXIMUM
    return np.sign(effort) * np.maximum(np.abs(effort) - theta, 0.0)


def _independent_pair_oracle(effort, bias):
    fx, fy = effort
    bias = np.asarray(bias)
    result = np.zeros(4)
    for positive, negative, difference in ((0, 3, 0.5 * (fx + fy)), (1, 2, 0.5 * (fx - fy))):
        low = max(0.0, -difference)
        high = min(MAXIMUM, MAXIMUM - difference)
        negative_value = np.clip(0.5 * (bias[positive] + bias[negative] - difference), low, high)
        result[negative] = negative_value
        result[positive] = negative_value + difference
    return result


@pytest.mark.parametrize("effort", ((0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.01, 0.01), (0.03, 0.0), (0.02, 0.02), (-0.03, 0.01)))
@pytest.mark.parametrize("bias", ((0.0,) * 4, (0.002,) * 4, (0.005,) * 4, (0.001, 0.003, 0.004, 0.002)))
def test_boundaries_vertices_infeasible_targets_and_bias_match_independent_oracle(effort, bias):
    allocation = _allocator(bias).allocate(effort)
    primary = _project_diamond(effort)
    expected = _independent_pair_oracle(primary, bias)
    np.testing.assert_allclose(allocation.primary_optimal_bending_effort_n, primary, atol=1e-14)
    np.testing.assert_allclose(allocation.tension_n, expected, atol=1e-14)
    assert all(0.0 <= value <= MAXIMUM for value in allocation.tension_n)


def test_random_feasible_and_infeasible_efforts_match_closed_form_secondary_oracle():
    random = np.random.default_rng(20260828)
    for _ in range(250):
        effort = random.uniform(-0.04, 0.04, size=2)
        bias = random.uniform(0.0, MAXIMUM, size=4)
        allocation = _allocator(tuple(bias)).allocate(effort)
        primary = _project_diamond(effort)
        expected = _independent_pair_oracle(primary, bias)
        np.testing.assert_allclose(allocation.tension_n, expected, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(allocation.achieved_bending_effort_n, primary, rtol=0.0, atol=2e-14)
        assert allocation.secondary_objective_n2 == pytest.approx(float(np.sum((expected - bias) ** 2)), abs=1e-16)


def test_primary_multi_solution_uses_unique_minimum_bias_distance_solution():
    bias = (0.009, 0.001, 0.008, 0.002)
    allocation = _allocator(bias).allocate((0.0, 0.0))
    expected = _independent_pair_oracle((0.0, 0.0), bias)
    np.testing.assert_allclose(allocation.tension_n, expected, atol=1e-15)
    np.testing.assert_allclose(allocation.achieved_bending_effort_n, 0.0, atol=1e-15)


def test_manifest_permutations_preserve_physical_allocation():
    effort = (0.013, -0.004)
    bias = np.array([0.001, 0.002, 0.003, 0.004])
    reference = np.asarray(_allocator(tuple(bias)).allocate(effort).tension_n)
    for permutation in itertools.permutations(range(4)):
        names = tuple(NAMES[index] for index in permutation)
        candidate = _allocator(tuple(bias[list(permutation)]), names).allocate(effort)
        np.testing.assert_allclose(candidate.tension_n, reference[list(permutation)], atol=2e-14)


def test_checker_is_diagnostic_and_not_a_hidden_optimization_level():
    allocation = _allocator().allocate((0.01, 0.01))
    assert allocation.tension_n == pytest.approx((0.01, 0.0, 0.0, 0.0))
    assert allocation.twist_candidate_effort_n == pytest.approx(0.01)
    assert allocation.dynamic_common_offset_n == pytest.approx(0.0025)
    assert allocation.bias_tension_n == (0.0,) * 4


def test_dense_grid_never_finds_better_primary_or_secondary_objective():
    effort = np.array([0.027, -0.011])
    bias = np.array([0.003, 0.006, 0.002, 0.008])
    allocation = _allocator(tuple(bias)).allocate(effort)
    chosen = np.asarray(allocation.tension_n)
    grid = np.linspace(0.0, MAXIMUM, 21)
    matrix = _allocator().coordinates.bending_g
    chosen_primary = np.linalg.norm(matrix @ chosen - effort) ** 2
    chosen_secondary = np.linalg.norm(chosen - bias) ** 2
    for values in itertools.product(grid, repeat=4):
        candidate = np.asarray(values)
        primary = np.linalg.norm(matrix @ candidate - effort) ** 2
        if primary < chosen_primary - 1e-12:
            pytest.fail("dense independent grid found a better primary objective")
        if abs(primary - chosen_primary) <= 1e-12:
            assert np.linalg.norm(candidate - bias) ** 2 >= chosen_secondary - 1e-12


def test_status_and_three_common_terms_are_distinct():
    feasible = _allocator().allocate((0.005, 0.0))
    saturated = _allocator().allocate((0.05, 0.0))
    assert feasible.status == "optimal_feasible"
    assert saturated.status == "saturated_best_effort"
    assert feasible.bias_tension_n == (0.0,) * 4
    assert feasible.dynamic_common_offset_n > 0.0
    assert feasible.physical_common_effort_n == pytest.approx(4 * feasible.dynamic_common_offset_n)
