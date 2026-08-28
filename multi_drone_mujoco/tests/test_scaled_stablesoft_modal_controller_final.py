"""Final fixed-root StableSoft modal-controller workspace contracts."""

import itertools
from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    scaled_stablesoft_bending_controller_config,
)
from multi_drone_mujoco.tests._scaled_stablesoft_c2_c2_closed_loop import (
    CompatibleClosedLoopConfig,
    run_compatible_closed_loop,
)
from multi_drone_mujoco.tests.test_scaled_stablesoft_c2_c2_closed_loop import (
    _controller_config as pd_only_controller_config,
)


DT = 1.0 / 240.0
FREQUENCIES = (2400, 4800)
CASES = MappingProxyType(
    {
        "axis_75_neg_x": ((-1.0, 0.0), 75e-6, 12),
        "axis_136_pos_y": ((0.0, 1.0), 136e-6, 20),
        "axis_250_neg_y": ((0.0, -1.0), 250e-6, 30),
        "diagonal_75_pm": ((2 ** -0.5, -(2 ** -0.5)), 75e-6, 20),
        "diagonal_136_mp": ((-(2 ** -0.5), 2 ** -0.5), 136e-6, 28),
        "diagonal_200_mm": ((-(2 ** -0.5), -(2 ** -0.5)), 200e-6, 36),
    }
)


def _run_case(frequency, case):
    direction, reference_m, pulse_ticks = case
    pulse_s = pulse_ticks * DT
    acceleration = reference_m / pulse_s ** 2
    return run_compatible_closed_loop(
        CompatibleClosedLoopConfig(
            frequency,
            scaled_stablesoft_bending_controller_config(),
            direction,
            acceleration,
            pulse_s,
            1.5,
        )
    )


@pytest.fixture(scope="module")
def final_traces():
    return MappingProxyType(
        {
            (frequency, name): _run_case(frequency, case)
            for frequency, (name, case) in itertools.product(
                FREQUENCIES, CASES.items()
            )
        }
    )


def _tracking_metrics(trace):
    direction = np.asarray(trace.config.direction)
    orthogonal = np.asarray((-direction[1], direction[0]))
    actual = trace.q_actual_m @ direction
    reference = float(trace.q_reference_m[-1] @ direction)
    profile_end = 2 * trace.config.pulse_ticks
    tail_error = reference - float(np.mean(actual[-48:]))
    hold_rms = float(
        np.sqrt(np.mean((actual[profile_end:] - reference) ** 2))
    )
    overshoot = max(0.0, float(np.max(actual)) - reference) / reference
    tolerance = max(1e-6, 0.02 * reference)
    settling = None
    for index in range(profile_end, len(actual)):
        if np.all(np.abs(actual[index:] - reference) <= tolerance):
            settling = float(
                trace.time_s[index] - trace.time_s[profile_end]
            )
            break
    return {
        "tail_error": tail_error,
        "hold_rms": hold_rms,
        "overshoot": overshoot,
        "settling": settling,
        "cross": float(np.max(np.abs(trace.q_actual_m @ orthogonal))),
    }


def test_final_workspace_tracks_axes_and_diagonals_with_frozen_accuracy(final_traces):
    for trace in final_traces.values():
        metrics = _tracking_metrics(trace)
        assert abs(metrics["tail_error"]) <= 0.025e-6
        assert metrics["hold_rms"] <= 1.9e-6
        assert metrics["overshoot"] <= 0.08
        assert metrics["settling"] is not None
        assert metrics["settling"] <= 0.38
        assert metrics["cross"] <= 1e-12


def test_final_workspace_is_bounded_unsaturated_and_inside_feedforward_domain(final_traces):
    for trace in final_traces.values():
        assert trace.warning_events == ()
        np.testing.assert_array_equal(trace.contact_count, 0)
        assert not np.any(trace.fault_latched)
        assert not np.any(trace.feedforward_out_of_envelope)
        assert not np.any(trace.allocator_saturated)
        assert np.max(trace.allocator_residual_n) <= 1e-15
        assert np.min(trace.tension_command_n) >= 0.0
        assert np.max(trace.tension_command_n) <= 7.8e-3
        assert np.min(trace.tendon_range_margin_m) >= -1e-9


def test_force_power_work_and_energy_diagnostics_close(final_traces):
    for trace in final_traces.values():
        np.testing.assert_allclose(
            trace.generalized_power_w, trace.actuator_power_w, atol=1e-15
        )
        np.testing.assert_allclose(
            trace.actuator_power_w, trace.tendon_power_w, atol=1e-15
        )
        np.testing.assert_allclose(
            trace.tendon_power_w, trace.command_tendon_power_w, atol=1e-15
        )
        assert np.max(np.abs(trace.energy_ledger_residual_j)) <= 4e-9
        assert trace.tendon_work_j[-1] > 0.0
        assert np.all(trace.spring_energy_j >= 0.0)


def test_final_workspace_converges_from_2400_to_4800_hz(final_traces):
    for name in CASES:
        low = final_traces[2400, name]
        high = final_traces[4800, name]
        np.testing.assert_array_equal(
            low.acceleration_command_m_s2,
            high.acceleration_command_m_s2,
        )
        assert np.max(np.abs(low.q_actual_m - high.q_actual_m)) <= 0.04e-6
        assert np.max(np.abs(low.tip_position_m - high.tip_position_m)) <= 1e-6
        assert np.max(
            np.abs(low.tension_command_n - high.tension_command_n)
        ) <= 5e-6
        assert abs(low.tendon_work_j[-1] - high.tendon_work_j[-1]) <= 1e-9


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_feedforward_removes_the_measured_pd_only_equilibrium_error(frequency):
    direction = (1.0, 0.0)
    pd_only = run_compatible_closed_loop(
        CompatibleClosedLoopConfig(
            frequency, pd_only_controller_config(), direction, duration_s=1.5
        )
    )
    final = run_compatible_closed_loop(
        CompatibleClosedLoopConfig(
            frequency,
            scaled_stablesoft_bending_controller_config(),
            direction,
            duration_s=1.5,
        )
    )
    pd_error = 100e-6 - float(np.mean(pd_only.q_actual_m[-48:, 0]))
    final_error = 100e-6 - float(np.mean(final.q_actual_m[-48:, 0]))
    assert pd_error >= 10e-6
    assert abs(final_error) <= 0.025e-6
    assert abs(final_error) <= 0.003 * pd_error
