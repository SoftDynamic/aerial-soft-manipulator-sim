"""Long-tail quasi-static training and holdout evidence for modal feedforward."""

from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_modal_coordinates import (
    TendonModalCoordinates,
)
from multi_drone_mujoco.tests._scaled_stablesoft_c2_c2_probe import (
    VectorTensionProbeConfig,
    run_vector_tension_probe,
    tail_mean_contraction,
)


CASES = MappingProxyType(
    {
        "train_2400_2mN_pos_x": (2400, 0.002, (1, 1, 0, 0)),
        "train_2400_5mN_pos_y": (2400, 0.005, (1, 0, 1, 0)),
        "train_2400_10mN_diag_pp": (2400, 0.010, (1, 0, 0, 0)),
        "train_2400_5mN_pos_x": (2400, 0.005, (1, 1, 0, 0)),
        "hold_4800_2mN_neg_x": (4800, 0.002, (0, 0, 1, 1)),
        "hold_4800_5mN_neg_y": (4800, 0.005, (0, 1, 0, 1)),
        "hold_4800_10mN_diag_mm": (4800, 0.010, (0, 0, 0, 1)),
        "hold_4800_5mN_diag_pm": (4800, 0.005, (0, 1, 0, 0)),
        "hold_4800_5mN_pos_x": (4800, 0.005, (1, 1, 0, 0)),
    }
)


@pytest.fixture(scope="module")
def quasistatic():
    traces = {}
    for name, (frequency, tension, mask) in CASES.items():
        command = tuple(tension * value for value in mask)
        traces[name] = run_vector_tension_probe(
            VectorTensionProbeConfig(
                frequency,
                command,
                duration_s=3.0,
                ramp_duration_s=0.1,
            )
        )
    traces["repeat_2400_2mN_pos_x"] = run_vector_tension_probe(
        VectorTensionProbeConfig(
            2400,
            (0.002, 0.002, 0.0, 0.0),
            duration_s=3.0,
            ramp_duration_s=0.1,
        )
    )
    return MappingProxyType(traces)


def _modal_sample(trace):
    coordinates = TendonModalCoordinates.from_tendon_names(trace.tendon_names)
    contraction = tail_mean_contraction(trace, 2.8)
    q = coordinates.tendon_to_bending(contraction)
    effort = coordinates.bending_g @ np.asarray(
        trace.config.target_tension_n
    )
    active = np.abs(effort) > 1e-15
    stiffness = np.abs(effort[active] / q[active])
    return q, effort, stiffness


def test_quasistatic_training_freezes_the_linear_diagonal_stiffness(quasistatic):
    training = [
        quasistatic[name]
        for name in CASES
        if name.startswith("train_")
    ]
    stiffness = np.concatenate([_modal_sample(trace)[2] for trace in training])
    assert 52.50 <= np.mean(stiffness) <= 52.65
    assert np.ptp(stiffness) <= 0.12


def test_signed_axis_diagonal_and_working_point_holdouts_support_linear_model(quasistatic):
    for name, trace in quasistatic.items():
        q, effort, stiffness = _modal_sample(trace)
        active = np.abs(effort) > 1e-15
        assert np.all(q[active] * effort[active] > 0.0), name
        if np.any(~active):
            assert np.max(np.abs(q[~active])) <= 1e-12, name
        assert np.all((52.50 <= stiffness) & (stiffness <= 52.65)), name
        assert trace.safety.warning_events == ()
        assert trace.safety.maximum_contacts == 0
        assert trace.safety.minimum_tendon_range_margin_m >= -1e-9


def test_quasistatic_frequency_error_is_below_working_point_variation(quasistatic):
    low = _modal_sample(quasistatic["train_2400_5mN_pos_x"])[2]
    high = _modal_sample(quasistatic["hold_4800_5mN_pos_x"])[2]
    assert np.max(np.abs(low - high)) <= 5e-5


def test_long_quasistatic_repeat_is_bitwise_deterministic(quasistatic):
    original = quasistatic["train_2400_2mN_pos_x"]
    repeated = quasistatic["repeat_2400_2mN_pos_x"]
    np.testing.assert_array_equal(original.tendon_length_m, repeated.tendon_length_m)
    np.testing.assert_array_equal(original.tendon_velocity_m_s, repeated.tendon_velocity_m_s)
    np.testing.assert_array_equal(original.tip_position_m, repeated.tip_position_m)
