"""Free-UAV two-way coupling for the final external modal controller."""

import itertools
from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.control.ideal_tendon_tension import (
    resolve_ideal_tendon_actuator_bindings,
)
from multi_drone_mujoco.control.scaled_stablesoft_bending_control import (
    make_scaled_stablesoft_bending_controller,
)
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.tests._scaled_stablesoft_active_hover import (
    C1B3RunConfig,
    ModalControllerHoverRunConfig,
    early_energy_ledger_limit,
    energy_window_metrics,
    full_duration_energy_guard,
    run_c1_b3,
    run_modal_controller_hover,
)
from multi_drone_mujoco.tests._scaled_stablesoft_modal_hover import (
    ModalAccelerationHoverCommandSource,
    pulse_hold_commands,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


FREQUENCIES = (2400, 4800)
DIRECTIONS = MappingProxyType(
    {
        "plus_x": (1.0, 0.0),
        "minus_x": (-1.0, 0.0),
        "plus_y": (0.0, 1.0),
        "minus_y": (0.0, -1.0),
        "diagonal_pm": (2 ** -0.5, -(2 ** -0.5)),
    }
)


@pytest.fixture(scope="module")
def hover_results():
    baselines = {
        frequency: run_c1_b3(C1B3RunConfig(frequency, "baseline"))
        for frequency in FREQUENCIES
    }
    traces = {}
    for frequency, (name, direction) in itertools.product(
        FREQUENCIES, DIRECTIONS.items()
    ):
        source = ModalAccelerationHoverCommandSource(
            pulse_hold_commands(direction, 0.002, 12),
            disable_at_tick=180,
        )
        traces[frequency, name] = run_modal_controller_hover(
            ModalControllerHoverRunConfig(frequency),
            source,
            baseline=baselines[frequency],
        )
    return MappingProxyType({"baseline": baselines, "trace": traces})


def test_free_uav_tracks_all_signed_axes_and_diagonal_in_active_window(hover_results):
    for (frequency, name), trace in hover_results["trace"].items():
        control = trace.control
        direction = np.asarray(DIRECTIONS[name])
        orthogonal = np.asarray((-direction[1], direction[0]))
        tail = control["modal_actual_m"][132:180]
        along = tail @ direction
        cross = control["modal_actual_m"][:181] @ orthogonal
        assert control["modal_reference_m"][24] @ direction == pytest.approx(
            5e-6, abs=1e-18
        )
        assert 5.0e-6 <= float(np.mean(along)) <= 5.05e-6
        assert np.max(np.abs(cross)) <= 1e-12
        assert np.max(control["command"][:181]) <= 0.221e-3
        assert not np.any(control["allocator_saturated"][:181])
        assert not np.any(control["controller_fault_latched"])
        assert not np.any(control["feedforward_out_of_envelope"])


def test_uav_reaction_direction_matches_bend_moment_and_tilted_rotor_thrust(hover_results):
    traces = hover_results["trace"]
    for frequency in FREQUENCIES:
        plus_x = traces[frequency, "plus_x"].relative
        minus_x = traces[frequency, "minus_x"].relative
        plus_y = traces[frequency, "plus_y"].relative
        minus_y = traces[frequency, "minus_y"].relative
        tick = 180
        assert plus_x["uav_rotation_vector_rad"][tick, 1] > 0.0
        assert minus_x["uav_rotation_vector_rad"][tick, 1] < 0.0
        assert plus_y["uav_rotation_vector_rad"][tick, 0] < 0.0
        assert minus_y["uav_rotation_vector_rad"][tick, 0] > 0.0
        assert plus_x["uav_position_world_m"][tick, 0] > 0.0
        assert minus_x["uav_position_world_m"][tick, 0] < 0.0
        assert plus_y["uav_position_world_m"][tick, 1] > 0.0
        assert minus_y["uav_position_world_m"][tick, 1] < 0.0
        np.testing.assert_allclose(
            plus_x["uav_position_world_m"][:tick + 1, 0],
            -minus_x["uav_position_world_m"][:tick + 1, 0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plus_y["uav_position_world_m"][:tick + 1, 1],
            -minus_y["uav_position_world_m"][:tick + 1, 1],
            atol=1e-12,
        )


def test_free_uav_motion_changes_controller_measurement_and_tension(hover_results):
    # The free-base feedback trace must not collapse to a prescribed open-loop
    # command: both the measured q and tension vary throughout the hold.
    for trace in hover_results["trace"].values():
        control = trace.control
        hold = slice(24, 180)
        assert np.ptp(control["modal_actual_m"][hold, 0]) > 1e-9 or np.ptp(
            control["modal_actual_m"][hold, 1]
        ) > 1e-9
        assert np.ptp(control["command"][hold]) > 1e-6


def test_hover_error_and_open_loop_post_release_drift_are_quantified(hover_results):
    for frequency, traces in (
        (frequency, hover_results["trace"]) for frequency in FREQUENCIES
    ):
        baseline = hover_results["baseline"][frequency]
        for name in DIRECTIONS:
            trace = traces[frequency, name]
            relative = trace.relative
            active_position = np.linalg.norm(
                relative["uav_position_world_m"][:181], axis=1
            )
            active_attitude = relative["uav_quaternion_geodesic_rad"][:181]
            final_position = np.linalg.norm(
                trace.control["uav_position_world_m"][-1]
                - baseline.control["uav_position_world_m"][-1]
            )
            final_attitude = relative["uav_quaternion_geodesic_rad"][-1]
            assert 1e-4 <= np.max(active_position) <= 2.4e-3
            assert 1e-4 <= np.max(active_attitude) <= 5.2e-3
            # Equal open-loop hover RPM does not cancel the acquired lateral
            # velocity/attitude after tendon release; this is plant response.
            assert 0.40 <= final_position <= 0.45
            assert 0.04 <= final_attitude <= 0.051


def test_free_uav_force_power_energy_and_safety_contracts_hold(hover_results):
    for trace in hover_results["trace"].values():
        safety = trace.safety
        assert trace.status == "completed"
        assert safety.first_failure is None
        assert safety.maximum_contacts == 0
        assert np.all(trace.physics["warnings"] == 0)
        assert safety.minimum_tendon_margin_m >= -1e-9
        assert safety.maximum_force_command_error_n <= 1e-15
        assert safety.maximum_actuator_moment_residual <= 1e-12
        assert safety.maximum_tendon_power_identity_residual_w <= 1e-12
        assert safety.maximum_rotor_power_residual_w <= 1e-12
        energy = energy_window_metrics(trace)
        assert energy["early"]["maximum_abs_ledger_residual_j"] <= (
            early_energy_ledger_limit(
                energy["early"]["peak_abs_combined_work_j"]
            )
        )
        assert energy["full"]["maximum_abs_ledger_residual_j"] <= (
            full_duration_energy_guard(
                energy["full"]["peak_abs_combined_work_j"]
            )
        )
        assert trace.control["rotor_rpm"].shape[1:] == (4,)
        assert trace.control["command"].shape[1:] == (4,)


def test_free_uav_control_sequence_and_trajectories_converge_with_rate(hover_results):
    for name in DIRECTIONS:
        low = hover_results["trace"][2400, name].control
        high = hover_results["trace"][4800, name].control
        np.testing.assert_array_equal(
            low["modal_acceleration_command_m_s2"],
            high["modal_acceleration_command_m_s2"],
        )
        assert np.max(np.abs(low["modal_actual_m"] - high["modal_actual_m"])) <= 0.022e-6
        assert np.max(np.abs(low["command"] - high["command"])) <= 0.5e-6
        assert np.max(
            np.abs(low["uav_position_world_m"] - high["uav_position_world_m"])
        ) <= 18e-6


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_disabled_external_controller_is_exact_hover_baseline(frequency):
    baseline = run_c1_b3(C1B3RunConfig(frequency, "baseline"))
    source = ModalAccelerationHoverCommandSource(
        pulse_hold_commands((1.0, 0.0), 0.002, 12),
        enabled=False,
    )
    disabled = run_modal_controller_hover(
        ModalControllerHoverRunConfig(frequency), source, baseline=baseline
    )
    np.testing.assert_array_equal(disabled.control["command"], 0.0)
    for name in (
        "uav_position_world_m",
        "uav_quaternion",
        "tendon_length",
        "tendon_velocity",
        "qvel",
        "qacc",
    ):
        np.testing.assert_array_equal(disabled.control[name], baseline.control[name])


def test_fault_latch_clear_and_disable_are_exercised_on_free_uav():
    baseline = run_c1_b3(C1B3RunConfig(2400, "baseline"))
    source = ModalAccelerationHoverCommandSource(
        pulse_hold_commands((1.0, 0.0), 0.002, 12),
        fault_at_tick=30,
        clear_fault_at_tick=45,
        disable_at_tick=180,
    )
    trace = run_modal_controller_hover(
        ModalControllerHoverRunConfig(2400), source, baseline=baseline
    )
    control = trace.control
    assert control["controller_fault_latched"][30]
    np.testing.assert_array_equal(control["command"][30:45], 0.0)
    assert not control["controller_fault_latched"][45]
    assert np.any(control["command"][45:180] > 0.0)
    np.testing.assert_array_equal(control["command"][180:], 0.0)
    assert trace.status == "completed"


def test_public_env_step_keeps_rpm_and_tension_as_separate_four_vectors():
    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.asarray([[0.0, 0.0, 1.0]]),
        initial_rpys=np.asarray([[0.01, -0.01, 0.0]]),
        physics=Physics.MJC,
        sim_freq=2400,
        ctrl_freq=240,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        scaled_stablesoft_arm=ScaledStableSoftArmSpec(
            joint_stiffness=0.003086,
            joint_damping=None,
            target_damping_ratio=0.3,
        ),
        scaled_stablesoft_tendon_drive=IdealTendonTensionSpec(
            max_tension=0.01
        ),
        integrator="implicitfast",
    )
    try:
        env.reset()
        bindings = resolve_ideal_tendon_actuator_bindings(
            env.model, env.SCALED_STABLESOFT_ARM_MANIFEST
        )
        tendon_ids = np.asarray([binding.tendon_id for binding in bindings])
        controller = make_scaled_stablesoft_bending_controller(
            [binding.tendon_name for binding in bindings]
        )
        controller.reset(
            env.data.ten_length[tendon_ids], env.data.ten_velocity[tendon_ids]
        )
        controller.enable()
        accelerations = pulse_hold_commands(
            (1.0, 0.0), 0.002, 12, total_ticks=31
        )
        rotor_rpm = np.full(4, env.HOVER_RPM)
        tension = None
        for acceleration in accelerations:
            tension = controller.update_bending_acceleration(
                acceleration,
                env.data.ten_length[tendon_ids],
                env.data.ten_velocity[tendon_ids],
            )
            env.set_scaled_stablesoft_tension_command(tension)
            env.step(rotor_rpm)
        assert rotor_rpm.shape == (4,)
        assert np.asarray(tension).shape == (4,)
        assert env.data.ncon == 0
        assert np.all(np.isfinite(env.data.qpos))
        assert np.all(np.isfinite(env.data.qvel))
    finally:
        env.close()
