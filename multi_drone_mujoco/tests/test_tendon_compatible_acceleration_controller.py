import numpy as np
import pytest

from multi_drone_mujoco.control.bounded_tendon_allocator import (
    AllocationNumericalError,
    BoundedTendonAllocatorConfig,
)
from multi_drone_mujoco.control.tendon_bending_reference import BendingReferenceConfig
from multi_drone_mujoco.control.tendon_compatible_acceleration_controller import (
    TendonCompatibleAccelerationController,
    TendonCompatibleAccelerationControllerConfig,
)
from multi_drone_mujoco.control.tendon_modal_pd import BendingModalPDConfig
from multi_drone_mujoco.control.tendon_modal_static_feedforward import (
    BendingModalStaticFeedforwardConfig,
)


NAMES = tuple(f"arm_{corner}_tendon" for corner in ("xp_yp", "xp_yn", "xn_yp", "xn_yn"))


def _config(**allocator_overrides):
    return TendonCompatibleAccelerationControllerConfig(
        reference=BendingReferenceConfig(0.01, 0.2, 0.02, 0.002),
        modal_pd=BendingModalPDConfig(100.0, 0.5),
        allocator=BoundedTendonAllocatorConfig(
            max_tension_n=allocator_overrides.get("max_tension_n", 0.01),
            bias_tension_n=allocator_overrides.get("bias_tension_n", (0.0,) * 4),
        ),
    )


def _feedforward_config():
    base = _config()
    return TendonCompatibleAccelerationControllerConfig(
        reference=base.reference,
        modal_pd=base.modal_pd,
        allocator=base.allocator,
        static_feedforward=BendingModalStaticFeedforwardConfig(
            (52.58, 52.58), 0.002, 0.003
        ),
    )


def _controller(names=NAMES):
    result = TendonCompatibleAccelerationController(_config(), names)
    result.reset((1.0,) * 4, (0.0,) * 4)
    return result


def test_composite_requires_exact_legacy_physical_limits():
    with pytest.raises(ValueError, match="exactly 0.01"):
        _config(max_tension_n=0.02)
    with pytest.raises(ValueError, match="preload must remain zero"):
        _config(bias_tension_n=(0.001,) * 4)


def test_disabled_is_normal_exact_zero_and_does_not_integrate_reference():
    controller = _controller()
    before = controller.reference.state
    assert controller.update_bending_acceleration((0.1, 0.0), (1.0,) * 4, (0.0,) * 4) == (0.0,) * 4
    assert controller.state.status == "disabled"
    assert not controller.state.fault_latched
    assert controller.reference.state == before


def test_reset_enabled_zero_acceleration_matches_zero_preload_semantics():
    controller = _controller()
    controller.enable()
    assert controller.update_bending_acceleration((0.0, 0.0), (1.0,) * 4, (0.0,) * 4) == pytest.approx((0.0,) * 4, abs=1e-15)
    assert controller.state.modal_pd.desired_bending_effort_n == pytest.approx((0.0, 0.0))
    assert controller.state.allocation.dynamic_common_offset_n == pytest.approx(0.0)


def test_feedforward_disabled_exactly_preserves_pd_only_command_path():
    first = _controller()
    second = TendonCompatibleAccelerationController(_config(), NAMES)
    second.reset((1.0,) * 4, (0.0,) * 4)
    first.enable()
    second.enable()
    commands = ((0.1, -0.05), (0.0, 0.0), (-0.03, 0.02))
    for command in commands:
        left = first.update_bending_acceleration(command, (1.0,) * 4, (0.0,) * 4)
        right = second.update_bending_acceleration(command, (1.0,) * 4, (0.0,) * 4)
        assert left == right
        assert first.state.static_feedforward is None
        assert not first.state.feedforward_out_of_envelope


def test_enabled_feedforward_adds_only_two_mode_effort_before_allocator():
    controller = TendonCompatibleAccelerationController(_feedforward_config(), NAMES)
    controller.reset((1.0,) * 4, (0.0,) * 4)
    controller.enable()
    controller.update_bending_acceleration((0.1, -0.05), (1.0,) * 4, (0.0,) * 4)
    pd = np.asarray(controller.state.modal_pd.desired_bending_effort_n)
    feedforward = np.asarray(controller.state.static_feedforward.static_bending_effort_n)
    np.testing.assert_allclose(
        controller.state.desired_total_bending_effort_n,
        pd + feedforward,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        controller.state.allocation.requested_bending_effort_n,
        pd + feedforward,
        atol=1e-15,
    )


def test_composite_uses_explicit_modal_state_and_bounded_nonnegative_allocation():
    controller = _controller()
    controller.enable()
    tension = controller.update_bending_acceleration((0.1, -0.05), (1.0,) * 4, (0.0,) * 4)
    assert all(0.0 <= value <= 0.01 for value in tension)
    np.testing.assert_allclose(controller.state.reference.bending_contraction_reference_m, (1e-5, -5e-6))
    np.testing.assert_allclose(controller.state.modal_pd.position_actual_m, (0.0, 0.0))
    np.testing.assert_allclose(controller.state.modal_pd.desired_bending_effort_n, (0.0015, -0.00075))
    np.testing.assert_allclose(controller.state.allocation.achieved_bending_effort_n, (0.0015, -0.00075), atol=1e-14)


def test_full_four_channel_input_is_projected_before_all_reference_states():
    controller = _controller()
    controller.enable()
    controller.update_tendon_acceleration((0.2, 0.1, 0.0, -0.3), (1.0,) * 4, (0.0,) * 4)
    state = controller.state
    assert np.linalg.norm(state.input_projection_residual_m_s2) > 0.0
    for values in (
        state.reference.tendon_acceleration_reference_m_s2,
        state.reference.tendon_velocity_reference_m_s,
        state.reference.tendon_contraction_reference_m,
    ):
        modal = controller.coordinates.tendon_to_modal(values)
        assert abs(modal[0]) <= 1e-15
        assert abs(modal[3]) <= 1e-15


def test_strict_four_channel_entry_rejects_common_and_checker_without_state_change():
    controller = _controller()
    controller.enable()
    before = controller.state
    for command in ((0.1,) * 4, (0.1, -0.1, -0.1, 0.1)):
        with pytest.raises(ValueError, match="outside the retained bending subspace"):
            controller.update_tendon_acceleration_strict(
                command, (1.0,) * 4, (0.0,) * 4
            )
        assert controller.state is before


def test_projecting_common_and_checker_requests_reports_residual_and_zero_reference():
    for command in ((0.1,) * 4, (0.1, -0.1, -0.1, 0.1)):
        controller = _controller()
        controller.enable()
        tension = controller.update_tendon_acceleration(
            command, (1.0,) * 4, (0.0,) * 4
        )
        assert tension == pytest.approx((0.0,) * 4, abs=1e-15)
        assert np.linalg.norm(controller.state.input_projection_residual_m_s2) > 0.0
        assert controller.state.reference.bending_contraction_reference_m == (0.0, 0.0)


def test_sustained_saturation_releases_when_measurement_reaches_reference():
    controller = TendonCompatibleAccelerationController(_feedforward_config(), NAMES)
    neutral = np.ones(4)
    controller.reset(neutral, np.zeros(4))
    controller.enable()
    for _ in range(4):
        controller.update_bending_acceleration(
            (0.2, 0.0), neutral, np.zeros(4)
        )
    assert controller.state.allocation.status == "saturated_best_effort"
    assert max(controller.state.tension_command_n) == pytest.approx(0.01)

    reference = np.asarray(
        controller.state.reference.bending_contraction_reference_m
    )
    velocity = np.asarray(
        controller.state.reference.bending_velocity_reference_m_s
    )
    next_reference = reference + velocity * controller.config.reference.control_timestep_s
    contraction = controller.coordinates.bending_to_tendon(next_reference)
    tendon_velocity = -controller.coordinates.bending_to_tendon(velocity)
    controller.update_bending_acceleration(
        (0.0, 0.0), neutral - contraction, tendon_velocity
    )
    assert controller.state.allocation.status == "optimal_feasible"
    assert max(controller.state.tension_command_n) < 0.01
    assert not controller.state.fault_latched


def test_direction_reversal_crosses_zero_without_leaving_bending_subspace():
    controller = TendonCompatibleAccelerationController(_feedforward_config(), NAMES)
    neutral = np.ones(4)
    controller.reset(neutral, np.zeros(4))
    controller.enable()
    for command, count in (
        ((0.1, 0.0), 10),
        ((-0.1, 0.0), 10),
        ((-0.02, 0.0), 24),
        ((0.02, 0.0), 24),
    ):
        for _ in range(count):
            controller.update_bending_acceleration(
                command, neutral, np.zeros(4)
            )
    reference = controller.state.reference
    assert reference.bending_contraction_reference_m[0] < 0.0
    assert reference.bending_velocity_reference_m_s[0] == pytest.approx(0.0)
    assert reference.common_residuals == pytest.approx((0.0,) * 3, abs=1e-18)
    assert reference.twist_candidate_residuals == pytest.approx(
        (0.0,) * 3, abs=1e-18
    )
    assert all(0.0 <= value <= 0.01 for value in controller.state.tension_command_n)


def test_manifest_permutation_preserves_named_corner_tensions():
    permutation = (2, 0, 3, 1)
    canonical = _controller()
    permuted = _controller(tuple(NAMES[index] for index in permutation))
    canonical.enable()
    permuted.enable()
    first = canonical.update_bending_acceleration((0.08, -0.03), (1.0,) * 4, (0.0,) * 4)
    second = permuted.update_bending_acceleration((0.08, -0.03), (1.0,) * 4, (0.0,) * 4)
    restored = np.empty(4)
    restored[list(permutation)] = second
    np.testing.assert_allclose(restored, first, atol=1e-14)


def test_allocator_failure_zeros_and_latches_until_explicit_clear(monkeypatch):
    controller = _controller()
    controller.enable()

    def fail(_):
        raise AllocationNumericalError("injected")

    monkeypatch.setattr(controller.allocator, "allocate", fail)
    assert controller.update_bending_acceleration((0.1, 0.0), (1.0,) * 4, (0.0,) * 4) == (0.0,) * 4
    fault_tick = controller.state.fault_tick
    assert controller.state.fault_latched
    assert "injected" in controller.state.fault_reason

    monkeypatch.undo()
    assert controller.update_bending_acceleration((0.0, 0.0), (1.0,) * 4, (0.0,) * 4) == (0.0,) * 4
    assert controller.state.status == "fault_latched"
    assert controller.state.fault_tick == fault_tick

    controller.clear_fault()
    command = controller.update_bending_acceleration((0.0, 0.0), (1.0,) * 4, (0.0,) * 4)
    assert any(value > 0.0 for value in command)
    assert not controller.state.fault_latched


def test_reset_also_clears_fault_and_disables_controller(monkeypatch):
    controller = _controller()
    controller.enable()
    monkeypatch.setattr(
        controller.allocator,
        "allocate",
        lambda _: (_ for _ in ()).throw(AllocationNumericalError("injected")),
    )
    controller.update_bending_acceleration((0.1, 0.0), (1.0,) * 4, (0.0,) * 4)
    monkeypatch.undo()
    state = controller.reset((1.0,) * 4, (0.0,) * 4)
    assert not state.enabled
    assert not state.fault_latched
    assert state.tension_command_n == (0.0,) * 4
    assert state.control_tick == 0


@pytest.mark.parametrize("method, command", [
    ("update_bending_acceleration", (np.nan, 0.0)),
    ("update_tendon_acceleration", (0.0, 0.0, np.inf, 0.0)),
])
def test_invalid_public_command_is_rejected_without_becoming_a_fault(method, command):
    controller = _controller()
    controller.enable()
    with pytest.raises(ValueError):
        getattr(controller, method)(command, (1.0,) * 4, (0.0,) * 4)
    assert not controller.fault_latched
