"""Fixed-root static characterization for the scaled StableSoft arm."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_dynamics import (
    build_fixed_arm_fixture,
    full_mass_matrix,
    named_joint_addresses,
    run_static_characterization,
)


SIM_FREQ = 4800
CTRL_FREQ = 240
INTEGRATOR = "implicitfast"
PHYSICAL_DURATION = 5.0
STEADY_WINDOW = 0.5
JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3


def _b1_spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


@pytest.fixture(scope="module")
def static_results():
    return {
        plane: run_static_characterization(
            plane,
            physical_duration=PHYSICAL_DURATION,
            steady_window=STEADY_WINDOW,
            sim_freq=SIM_FREQ,
            ctrl_freq=CTRL_FREQ,
            integrator=INTEGRATOR,
        )
        for plane in ("x", "y")
    }


def _diagnostics(result):
    return (
        f"plane={result.plane}, tip_drift={result.tip_window_drift}, "
        f"max_qvel={result.max_arm_qvel}, "
        f"joint_angle_change={result.joint_angle_window_change}, "
        f"bend={result.total_bend}, twist={result.total_twist}, "
        f"energy=({result.initial_energy}, {result.maximum_energy}, "
        f"{result.final_energy}), warnings={result.warning_count}, "
        f"contacts={result.max_contacts}"
    )


def test_b1_fixture_compiles_explicit_physics_and_passive_parameters():
    spec = _b1_spec()
    fixture = build_fixed_arm_fixture(
        spec,
        "x",
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
    )
    model = fixture.model
    addresses = named_joint_addresses(model, fixture.manifest)

    assert model.opt.timestep == pytest.approx(1.0 / SIM_FREQ)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert fixture.sample_stride == 20
    assert model.nu == 0
    assert model.ntendon == 4
    for index, (joint_id, _, dof_address) in enumerate(addresses):
        assert model.jnt_stiffness[joint_id] == pytest.approx(
            spec.resolved_joint_stiffness[index]
        )
        np.testing.assert_allclose(
            model.dof_damping[dof_address:dof_address + 3],
            spec.resolved_joint_damping[index],
            rtol=1e-12,
            atol=0.0,
        )


def test_b1_analytic_bending_inertia_matches_compiled_mass_matrix():
    spec = _b1_spec()
    fixture = build_fixed_arm_fixture(
        spec,
        "x",
        sim_freq=SIM_FREQ,
        ctrl_freq=CTRL_FREQ,
        integrator=INTEGRATOR,
    )
    mass_matrix = full_mass_matrix(fixture.model, fixture.data)
    addresses = named_joint_addresses(fixture.model, fixture.manifest)
    twist_inertias = []

    for expected, (_, _, dof_address) in zip(
        spec.bending_effective_inertias, addresses
    ):
        diagonal = np.diag(mass_matrix)[dof_address:dof_address + 3]
        np.testing.assert_allclose(
            diagonal[:2], expected, rtol=1e-12, atol=1e-15
        )
        twist_inertias.append(float(diagonal[2]))

    assert all(value > 0 for value in twist_inertias)
    assert all(
        twist < bending
        for twist, bending in zip(
            twist_inertias, spec.bending_effective_inertias
        )
    )
    expected_damping = tuple(
        2 * TARGET_DAMPING_RATIO * math.sqrt(JOINT_STIFFNESS * inertia)
        for inertia in spec.bending_effective_inertias
    )
    assert spec.resolved_joint_damping == pytest.approx(
        expected_damping, rel=1e-12
    )


@pytest.mark.parametrize("plane", ["x", "y"])
def test_b1_static_response_reaches_approved_steady_state(
    static_results, plane
):
    result = static_results[plane]
    diagnostics = _diagnostics(result)

    assert result.all_finite, diagnostics
    assert result.warning_count == 0, diagnostics
    assert result.max_contacts == 0, diagnostics
    assert result.tip_window_drift <= 1e-4, diagnostics
    assert result.max_arm_qvel <= 1e-3, diagnostics
    assert result.joint_angle_window_change <= 1e-3, diagnostics
    assert result.tip_gravity_displacement < -1e-6, diagnostics
    assert np.all(np.isfinite(result.tendon_lengths)), diagnostics
    assert result.model_counts == (11, 8, 32, 24, 10, 75, 4, 0, 4)
    assert result.maximum_energy <= result.initial_energy + 1e-9, diagnostics
    assert result.final_energy <= result.initial_energy + 1e-9, diagnostics


@pytest.mark.parametrize("plane", ["x", "y"])
def test_b1_static_bending_distribution_and_twist(static_results, plane):
    result = static_results[plane]
    diagnostics = _diagnostics(result)
    bends = result.joint_main_bends
    bend_sign = np.sign(np.sum(bends))
    significant = np.abs(bends) > 1e-5

    assert result.total_bend > 1e-3, diagnostics
    assert bend_sign != 0, diagnostics
    assert np.all(np.sign(bends[significant]) == bend_sign), diagnostics
    assert np.mean(np.abs(bends[:3])) > np.mean(np.abs(bends[-3:])), diagnostics
    assert result.total_twist <= max(
        1e-4, 0.01 * result.total_bend
    ), diagnostics


def test_b1_static_x_y_responses_are_mirror_symmetric(static_results):
    x_result = static_results["x"]
    y_result = static_results["y"]
    tip_x = abs(x_result.tip_gravity_displacement)
    tip_y = abs(y_result.tip_gravity_displacement)
    bend_x = abs(x_result.total_bend)
    bend_y = abs(y_result.total_bend)

    assert abs(tip_x - tip_y) <= max(1e-6, 0.005 * max(tip_x, tip_y))
    assert abs(bend_x - bend_y) <= max(
        1e-5, 0.005 * max(bend_x, bend_y)
    )
    np.testing.assert_allclose(
        np.abs(x_result.joint_main_bends),
        np.abs(y_result.joint_main_bends),
        rtol=0.005,
        atol=1e-5,
    )
