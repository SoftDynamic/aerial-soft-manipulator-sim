"""Gate C2-C2-G fixed-root modal closed-loop acceptance."""

import itertools

import numpy as np
import pytest

from multi_drone_mujoco.control.bounded_tendon_allocator import BoundedTendonAllocatorConfig
from multi_drone_mujoco.control.tendon_bending_reference import BendingReferenceConfig
from multi_drone_mujoco.control.tendon_compatible_acceleration_controller import TendonCompatibleAccelerationControllerConfig
from multi_drone_mujoco.control.tendon_modal_pd import BendingModalPDConfig
from multi_drone_mujoco.tests._scaled_stablesoft_c2_c2_closed_loop import (
    CompatibleClosedLoopConfig,
    run_compatible_closed_loop,
)


FREQUENCIES = (2400, 4800)
DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
TAIL_TICKS = 48


def _controller_config():
    return TendonCompatibleAccelerationControllerConfig(
        reference=BendingReferenceConfig(
            control_timestep_s=1.0 / 240.0,
            max_acceleration_m_s2=0.04,
            max_velocity_m_s=0.002,
            max_contraction_m=0.0003,
        ),
        modal_pd=BendingModalPDConfig(
            kp_n_per_m=400.0,
            kd_n_s_per_m=0.05,
        ),
        allocator=BoundedTendonAllocatorConfig(
            max_tension_n=0.01,
            bias_tension_n=(0.0, 0.0, 0.0, 0.0),
        ),
    )


@pytest.fixture(scope="module")
def closed_loops():
    config = _controller_config()
    return {
        (frequency, direction): run_compatible_closed_loop(
            CompatibleClosedLoopConfig(frequency, config, direction)
        )
        for frequency, direction in itertools.product(FREQUENCIES, DIRECTIONS)
    }


def _tail_q(trace):
    return np.mean(trace.q_actual_m[-TAIL_TICKS:], axis=0)


def test_fixed_root_runs_are_safe_bounded_and_numerically_healthy(closed_loops):
    for trace in closed_loops.values():
        assert trace.warning_events == ()
        np.testing.assert_array_equal(trace.contact_count, 0)
        assert np.min(trace.tendon_range_margin_m) >= -1e-9
        assert not np.any(trace.fault_latched)
        assert set(trace.allocator_status) == {"optimal_feasible"}
        assert np.all(np.isfinite(trace.q_actual_m))
        assert np.all(trace.tension_command_n >= 0.0)
        assert np.all(trace.tension_command_n <= 0.01)
        assert np.max(trace.tension_command_n) < 0.0031


def test_profile_reaches_exact_100_micrometre_modal_reference(closed_loops):
    for (_, direction), trace in closed_loops.items():
        expected = 100e-6 * np.asarray(direction)
        np.testing.assert_allclose(trace.q_reference_m[-1], expected, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(trace.q_velocity_reference_m_s[-1], 0.0, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(trace.common_reference_residual, 0.0, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(trace.twist_candidate_reference_residual, 0.0, rtol=0.0, atol=1e-18)


def test_all_signed_axes_track_without_geometric_opposition(closed_loops):
    for (_, direction), trace in closed_loops.items():
        direction = np.asarray(direction)
        orthogonal = np.asarray((-direction[1], direction[0]))
        tail = _tail_q(trace)
        along = float(tail @ direction)
        cross = float(tail @ orthogonal)
        assert 85e-6 <= along <= 92e-6
        assert abs(cross) <= 1e-12
        assert float(np.max(trace.q_actual_m @ direction)) <= 95e-6


def test_plus_minus_and_x_y_are_symmetric(closed_loops):
    for frequency in FREQUENCIES:
        plus_x = closed_loops[frequency, (1.0, 0.0)].q_actual_m
        minus_x = closed_loops[frequency, (-1.0, 0.0)].q_actual_m
        plus_y = closed_loops[frequency, (0.0, 1.0)].q_actual_m
        minus_y = closed_loops[frequency, (0.0, -1.0)].q_actual_m
        np.testing.assert_allclose(plus_x, -minus_x, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(plus_y, -minus_y, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(plus_x[:, 0], plus_y[:, 1], rtol=0.0, atol=1e-12)


def test_2400_and_4800_hz_closed_loops_are_equivalent(closed_loops):
    for direction in DIRECTIONS:
        low = closed_loops[2400, direction]
        high = closed_loops[4800, direction]
        np.testing.assert_allclose(low.q_reference_m, high.q_reference_m, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(low.q_actual_m, high.q_actual_m, rtol=0.0, atol=0.15e-6)
        np.testing.assert_allclose(low.tension_command_n, high.tension_command_n, rtol=0.0, atol=25e-6)


def test_dynamic_common_offset_is_not_a_fixed_preload(closed_loops):
    trace = closed_loops[2400, (1.0, 0.0)]
    # The v1 bias/preload is exactly zero.  Positive mean tension is generated
    # dynamically only because nonnegative tendons must realize signed effort.
    assert _controller_config().allocator.bias_tension_n == (0.0,) * 4
    assert np.max(np.mean(trace.tension_command_n, axis=1)) > 0.0
    np.testing.assert_allclose(
        np.sum(trace.tension_command_n, axis=1),
        4.0 * np.mean(trace.tension_command_n, axis=1),
        rtol=0.0,
        atol=1e-18,
    )
