"""Gate C2-C2-C compatible modal reference integration contracts."""

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_bending_reference import (
    BendingModalReferenceIntegrator,
    BendingReferenceConfig,
    project_l1_ball,
)
from multi_drone_mujoco.control.tendon_modal_coordinates import TendonModalCoordinates


NAMES = tuple(f"arm_{corner}_tendon" for corner in ("xp_yp", "xp_yn", "xn_yp", "xn_yn"))


def _integrator(**overrides):
    values = dict(control_timestep_s=0.1, max_acceleration_m_s2=1.0, max_velocity_m_s=2.0, max_contraction_m=3.0)
    values.update(overrides)
    return BendingModalReferenceIntegrator(BendingReferenceConfig(**values), TendonModalCoordinates.from_tendon_names(NAMES))


def test_two_dimensional_acceleration_has_semi_implicit_metre_semantics():
    controller = _integrator()
    controller.reset(np.zeros(4))
    state = controller.update_bending((0.4, -0.2))
    assert state.bending_velocity_reference_m_s == pytest.approx((0.04, -0.02))
    assert state.bending_contraction_reference_m == pytest.approx((0.004, -0.002))
    assert state.tendon_acceleration_reference_m_s2 == pytest.approx((0.2, 0.6, -0.6, -0.2))


def test_zero_acceleration_preserves_existing_modal_velocity_reference():
    controller = _integrator()
    controller.reset(np.zeros(4))
    first = controller.update_bending((0.4, -0.2))
    second = controller.update_bending((0.0, 0.0))
    assert second.bending_velocity_reference_m_s == first.bending_velocity_reference_m_s
    np.testing.assert_allclose(second.bending_contraction_reference_m, np.asarray(first.bending_contraction_reference_m) + 0.1 * np.asarray(first.bending_velocity_reference_m_s))


def test_tendon_entry_projects_before_integrating_and_reports_projection():
    controller = _integrator()
    controller.reset(np.zeros(4))
    state, projection = controller.update_tendon((1.0, 0.0, 0.0, 0.0))
    assert projection.was_projected
    np.testing.assert_allclose(projection.projected, [0.5, 0.0, 0.0, -0.5])
    assert state.requested_bending_acceleration_m_s2 == pytest.approx((0.25, 0.25))


def test_every_state_remains_compatible_after_all_three_limit_projections():
    controller = _integrator(max_acceleration_m_s2=0.3, max_velocity_m_s=0.04, max_contraction_m=0.006)
    controller.reset((0.01, -0.02, 0.03, -0.04))
    for command in ((1.0, 1.0), (-1.0, 0.3), (0.0, 0.0), (0.8, -1.2)) * 20:
        state = controller.update_bending(command)
        for common, checker in zip(state.common_residuals, state.twist_candidate_residuals):
            assert abs(common) <= 1e-18
            assert abs(checker) <= 1e-18
        assert sum(abs(value) for value in state.effective_bending_acceleration_m_s2) <= 0.3 + 1e-15
        assert sum(abs(value) for value in state.bending_velocity_reference_m_s) <= 0.04 + 1e-15
        assert sum(abs(value) for value in state.bending_contraction_reference_m) <= 0.006 + 1e-15


def test_l1_projection_covers_edges_vertices_and_interior():
    for vector, expected in (((0.1, 0.2), (0.1, 0.2)), ((2.0, 0.0), (1.0, 0.0)), ((1.0, 1.0), (0.5, 0.5)), ((-2.0, 1.0), (-1.0, 0.0))):
        actual, _ = project_l1_ball(vector, 1.0)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
