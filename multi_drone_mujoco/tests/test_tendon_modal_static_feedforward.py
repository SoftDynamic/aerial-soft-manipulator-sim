"""Static modal feedforward and frozen StableSoft tuning contracts."""

import numpy as np
import pytest

from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    CONTROL_FREQUENCY_HZ,
    KD_N_S_PER_M,
    KP_N_PER_M,
    MAX_REFERENCE_L1_M,
    MAX_TENSION_N,
    STATIC_STIFFNESS_N_PER_M,
    make_scaled_stablesoft_bending_controller,
    scaled_stablesoft_bending_controller_config,
)
from multi_drone_mujoco.control.tendon_modal_static_feedforward import (
    BendingModalStaticFeedforward,
    BendingModalStaticFeedforwardConfig,
)


NAMES = tuple(
    f"arm_{corner}_tendon"
    for corner in ("xp_yp", "xp_yn", "xn_yp", "xn_yn")
)


def test_static_feedforward_is_diagonal_two_mode_effort_not_an_inverse():
    model = BendingModalStaticFeedforward(
        BendingModalStaticFeedforwardConfig((52.58, 52.60), 3e-4, 4e-4)
    )
    effort = model.update((100e-6, -50e-6), (90e-6, -45e-6))
    np.testing.assert_allclose(effort, (0.005258, -0.00263), atol=1e-15)
    assert not model.state.out_of_envelope
    assert model.state.envelope_reasons == ()


def test_static_feedforward_reports_reference_and_actual_out_of_envelope():
    model = BendingModalStaticFeedforward(
        BendingModalStaticFeedforwardConfig((52.58, 52.58), 3e-4, 4e-4)
    )
    model.update((200e-6, 101e-6), (250e-6, -151e-6))
    assert model.state.out_of_envelope
    assert model.state.envelope_reasons == ("reference", "actual")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"stiffness_n_per_m": (52.58,)},
        {"stiffness_n_per_m": (52.58, -1.0)},
        {"max_reference_l1_m": 0.0},
        {"max_actual_l1_m": np.nan},
    ),
)
def test_static_feedforward_rejects_invalid_calibration(kwargs):
    values = dict(
        stiffness_n_per_m=(52.58, 52.58),
        max_reference_l1_m=3e-4,
        max_actual_l1_m=4e-4,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        BendingModalStaticFeedforwardConfig(**values)


def test_frozen_scaled_stablesoft_factory_has_final_units_and_limits():
    config = scaled_stablesoft_bending_controller_config()
    assert CONTROL_FREQUENCY_HZ == 240
    assert config.reference.control_timestep_s == 1.0 / 240.0
    assert config.reference.max_contraction_m == MAX_REFERENCE_L1_M
    assert config.modal_pd.kp_n_per_m == KP_N_PER_M == 300.0
    assert config.modal_pd.kd_n_s_per_m == KD_N_S_PER_M == 0.02
    assert config.static_feedforward.stiffness_n_per_m == STATIC_STIFFNESS_N_PER_M
    assert config.allocator.max_tension_n == MAX_TENSION_N == 0.01
    assert config.allocator.bias_tension_n == (0.0,) * 4
    with pytest.raises(ValueError, match="must be 240"):
        scaled_stablesoft_bending_controller_config(120)


def test_factory_uses_runtime_names_and_starts_disabled_at_exact_zero():
    controller = make_scaled_stablesoft_bending_controller(NAMES)
    state = controller.reset((1.0,) * 4, (0.0,) * 4)
    assert state.status == "disabled"
    assert state.tension_command_n == (0.0,) * 4
    assert state.static_feedforward.static_bending_effort_n == (0.0,) * 2
    assert not state.feedforward_out_of_envelope
