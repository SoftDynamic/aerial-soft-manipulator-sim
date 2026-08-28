"""Gate C2-C2-D explicit modal PD contracts."""

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_modal_pd import BendingModalPD, BendingModalPDConfig


def test_modal_pd_computes_two_dimensional_effort_directly():
    controller = BendingModalPD(BendingModalPDConfig(100.0, 0.5))
    effort = controller.update((2e-4, -1e-4), (1e-3, -2e-3), (5e-5, -5e-5), (-1e-3, 1e-3))
    np.testing.assert_allclose(effort, [0.016, -0.0065], atol=1e-15)
    assert controller.state.position_error_m == pytest.approx((1.5e-4, -5e-5))
    assert controller.state.velocity_error_m_s == pytest.approx((2e-3, -3e-3))


def test_modal_pd_is_not_four_independent_channel_feedback():
    controller = BendingModalPD(BendingModalPDConfig(1.0, 0.0))
    effort = controller.update((1.0, 2.0), (0.0, 0.0), (0.25, -0.5), (0.0, 0.0))
    assert effort == pytest.approx((0.75, 2.5))
    assert len(effort) == 2


@pytest.mark.parametrize("field,value", (("kp_n_per_m", -1.0), ("kd_n_s_per_m", -1.0), ("kp_n_per_m", np.nan), ("kd_n_s_per_m", True)))
def test_modal_pd_rejects_invalid_gains(field, value):
    values = {"kp_n_per_m": 1.0, "kd_n_s_per_m": 1.0, field: value}
    with pytest.raises(ValueError):
        BendingModalPDConfig(**values)


def test_modal_pd_rejects_invalid_runtime_vectors():
    controller = BendingModalPD(BendingModalPDConfig(1.0, 1.0))
    with pytest.raises(ValueError):
        controller.update((1.0,), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        controller.update((1.0, np.inf), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
