"""Analytic Gate C2-A tests for pure tendon acceleration control."""

from dataclasses import FrozenInstanceError, fields
import math

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_acceleration_controller import (
    TendonAccelerationControllerConfig,
    TendonAccelerationControllerState,
    TendonContractionAccelerationController,
)


TEST_LENGTHS_M = (1.0, 1.1, 1.2, 1.3)
TEST_ZERO_VECTOR = (0.0, 0.0, 0.0, 0.0)


def _config(**overrides):
    """Return explicit synthetic test parameters, not production tuning."""
    values = {
        "control_timestep_s": 0.1,
        "kp_n_per_m": 2.0,
        "kd_n_s_per_m": 0.5,
        "max_acceleration_m_s2": 10.0,
        "max_velocity_m_s": 10.0,
        "min_contraction_m": -10.0,
        "max_contraction_m": 10.0,
        "max_tension_n": 100.0,
        "preload_tension_n": 1.0,
    }
    values.update(overrides)
    return TendonAccelerationControllerConfig(**values)


def _controller(**overrides):
    return TendonContractionAccelerationController(_config(**overrides))


def _reset(controller, *, lengths=TEST_LENGTHS_M, velocities=TEST_ZERO_VECTOR):
    return controller.reset(lengths, velocities)


def test_config_normalizes_valid_real_scalars_and_is_frozen():
    config = TendonAccelerationControllerConfig(
        control_timestep_s=np.float32(0.1),
        kp_n_per_m=2,
        kd_n_s_per_m=0,
        max_acceleration_m_s2=3,
        max_velocity_m_s=4,
        min_contraction_m=-5,
        max_contraction_m=6,
        max_tension_n=7,
    )

    assert all(
        type(getattr(config, field.name)) is float for field in fields(config)
    )
    assert config.preload_tension_n == 0.0
    with pytest.raises(FrozenInstanceError):
        config.kp_n_per_m = 3.0


def test_config_requires_explicit_max_tension_and_tuning_parameters():
    with pytest.raises(TypeError):
        TendonAccelerationControllerConfig(control_timestep_s=0.1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_timestep_s", 0.0),
        ("control_timestep_s", -0.1),
        ("kp_n_per_m", -1.0),
        ("kd_n_s_per_m", -1.0),
        ("max_acceleration_m_s2", 0.0),
        ("max_acceleration_m_s2", -1.0),
        ("max_velocity_m_s", 0.0),
        ("max_velocity_m_s", -1.0),
        ("max_tension_n", 0.0),
        ("max_tension_n", -1.0),
        ("preload_tension_n", -1e-6),
        ("preload_tension_n", 101.0),
    ],
)
def test_config_rejects_invalid_ranges(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_timestep_s", True),
        ("kp_n_per_m", np.bool_(False)),
        ("kd_n_s_per_m", np.nan),
        ("max_acceleration_m_s2", np.inf),
        ("max_velocity_m_s", -np.inf),
        ("min_contraction_m", 1j),
        ("max_contraction_m", "1.0"),
        ("max_tension_n", object()),
        ("preload_tension_n", np.nan),
    ],
)
def test_config_rejects_non_real_or_non_finite_scalars(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize(
    "minimum,maximum,message",
    [
        (0.0, 0.0, "less than"),
        (1.0, -1.0, "less than"),
        (0.1, 1.0, "reset reference"),
        (-1.0, -0.1, "reset reference"),
    ],
)
def test_config_rejects_invalid_contraction_interval(
    minimum, maximum, message
):
    with pytest.raises(ValueError, match=message):
        _config(
            min_contraction_m=minimum,
            max_contraction_m=maximum,
        )


def test_controller_constructor_requires_its_config_type():
    with pytest.raises(TypeError, match="config"):
        TendonContractionAccelerationController(object())


@pytest.mark.parametrize(
    "value",
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        np.zeros((2, 2)),
        0.0,
        "abcd",
        True,
    ],
)
@pytest.mark.parametrize(
    "field",
    (
        "tendon_contraction_acceleration_cmd_m_s2",
        "tendon_length_m",
        "tendon_velocity_m_s",
    ),
)
def test_update_rejects_inputs_that_are_not_exact_shape_four(field, value):
    controller = _controller()
    _reset(controller)
    arguments = {
        "tendon_contraction_acceleration_cmd_m_s2": TEST_ZERO_VECTOR,
        "tendon_length_m": TEST_LENGTHS_M,
        "tendon_velocity_m_s": TEST_ZERO_VECTOR,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        controller.update(**arguments)


@pytest.mark.parametrize(
    "value,index",
    [
        ((0.0, True, 0.0, 0.0), 1),
        ((0.0, np.nan, 0.0, 0.0), 1),
        ((0.0, 0.0, np.inf, 0.0), 2),
        ((0.0, 0.0, 0.0, 1j), 3),
        ((object(), 0.0, 0.0, 0.0), 0),
        (("0", 0.0, 0.0, 0.0), 0),
    ],
)
@pytest.mark.parametrize(
    "field",
    (
        "tendon_contraction_acceleration_cmd_m_s2",
        "tendon_length_m",
        "tendon_velocity_m_s",
    ),
)
def test_update_rejects_invalid_channel_values_with_index(field, value, index):
    controller = _controller()
    _reset(controller)
    arguments = {
        "tendon_contraction_acceleration_cmd_m_s2": TEST_ZERO_VECTOR,
        "tendon_length_m": TEST_LENGTHS_M,
        "tendon_velocity_m_s": TEST_ZERO_VECTOR,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=rf"{field}\[{index}\]"):
        controller.update(**arguments)


@pytest.mark.parametrize(
    "field",
    ("tendon_length_m", "tendon_velocity_m_s"),
)
def test_reset_rejects_invalid_vectors(field):
    controller = _controller()
    arguments = {
        "tendon_length_m": TEST_LENGTHS_M,
        "tendon_velocity_m_s": TEST_ZERO_VECTOR,
    }
    arguments[field] = (0.0, np.nan, 0.0, 0.0)

    with pytest.raises(ValueError, match=rf"{field}\[1\]"):
        controller.reset(**arguments)


def test_reset_and_update_reject_negative_tendon_length_with_index():
    controller = _controller()
    invalid = (1.0, -0.1, 1.0, 1.0)
    with pytest.raises(ValueError, match=r"tendon_length_m\[1\]"):
        controller.reset(invalid, TEST_ZERO_VECTOR)

    _reset(controller)
    with pytest.raises(ValueError, match=r"tendon_length_m\[1\]"):
        controller.update(TEST_ZERO_VECTOR, invalid, TEST_ZERO_VECTOR)


def test_update_and_state_before_reset_raise_clear_errors():
    controller = _controller()
    assert not controller.is_reset
    with pytest.raises(RuntimeError, match="reset before reading state"):
        _ = controller.state
    with pytest.raises(RuntimeError, match="reset before update"):
        controller.update(
            TEST_ZERO_VECTOR,
            TEST_LENGTHS_M,
            TEST_ZERO_VECTOR,
        )


def test_reset_records_neutral_length_and_matches_velocity_reference():
    controller = _controller(preload_tension_n=0.0)
    velocities = (0.1, -0.2, 0.3, -0.4)
    state = controller.reset(TEST_LENGTHS_M, velocities)

    assert controller.is_reset
    assert state.neutral_tendon_length_m == TEST_LENGTHS_M
    assert state.tendon_length_m == TEST_LENGTHS_M
    assert state.contraction_actual_m == TEST_ZERO_VECTOR
    assert state.contraction_reference_m == TEST_ZERO_VECTOR
    assert state.contraction_velocity_actual_m_s == (-0.1, 0.2, -0.3, 0.4)
    assert state.contraction_velocity_reference_m_s == (-0.1, 0.2, -0.3, 0.4)
    assert state.position_error_m == TEST_ZERO_VECTOR
    assert state.velocity_error_m_s == TEST_ZERO_VECTOR
    assert state.tension_command_n == TEST_ZERO_VECTOR
    assert not any(state.acceleration_was_clipped)
    assert not any(state.reference_velocity_was_projected)


def test_reset_defensively_copies_measurement_inputs():
    controller = _controller()
    lengths = np.asarray(TEST_LENGTHS_M)
    velocities = np.array([0.1, -0.2, 0.3, -0.4])
    state = controller.reset(lengths, velocities)
    lengths[:] = 9.0
    velocities[:] = 9.0

    assert state.neutral_tendon_length_m == TEST_LENGTHS_M
    assert state.tendon_velocity_m_s == (0.1, -0.2, 0.3, -0.4)
    assert controller.state is state


def test_update_defensively_copies_runtime_inputs():
    controller = _controller()
    _reset(controller)
    acceleration = np.array([1.0, 2.0, 3.0, 4.0])
    lengths = np.asarray(TEST_LENGTHS_M)
    velocities = np.array([0.1, 0.2, 0.3, 0.4])
    controller.update(acceleration, lengths, velocities)
    state = controller.state
    acceleration[:] = 9.0
    lengths[:] = 9.0
    velocities[:] = 9.0

    assert state.acceleration_command_m_s2 == (1.0, 2.0, 3.0, 4.0)
    assert state.tendon_length_m == TEST_LENGTHS_M
    assert state.tendon_velocity_m_s == (0.1, 0.2, 0.3, 0.4)


def test_invalid_update_leaves_previous_state_unchanged():
    controller = _controller()
    initial = _reset(controller)
    with pytest.raises(ValueError):
        controller.update(
            (0.0, np.nan, 0.0, 0.0),
            TEST_LENGTHS_M,
            TEST_ZERO_VECTOR,
        )
    assert controller.state is initial


def test_repeated_reset_fully_replaces_episode_state_and_saturation_flags():
    controller = _controller(
        max_acceleration_m_s2=10.0,
        max_velocity_m_s=0.2,
        min_contraction_m=-0.01,
        max_contraction_m=0.01,
        max_tension_n=0.1,
        preload_tension_n=0.0,
    )
    _reset(controller)
    controller.update((10.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)
    assert any(controller.state.reference_velocity_was_projected)

    new_lengths = (2.0, 2.1, 2.2, 2.3)
    new_velocities = (-0.4, -0.3, -0.2, -0.1)
    state = controller.reset(new_lengths, new_velocities)

    assert state.neutral_tendon_length_m == new_lengths
    assert state.contraction_reference_m == TEST_ZERO_VECTOR
    assert state.contraction_velocity_reference_m_s == (0.4, 0.3, 0.2, 0.1)
    for flag_name in (
        "acceleration_was_clipped",
        "velocity_was_clipped",
        "contraction_was_clipped",
        "reference_velocity_was_projected",
        "tension_was_clipped_low",
        "tension_was_clipped_high",
    ):
        assert not any(getattr(state, flag_name))


def test_one_step_uses_new_velocity_for_semi_implicit_position_update():
    controller = _controller(preload_tension_n=0.0)
    _reset(controller)
    controller.update((2.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)

    assert controller.state.contraction_velocity_reference_m_s == pytest.approx(
        (0.2,) * 4, rel=0.0, abs=1e-15
    )
    assert controller.state.contraction_reference_m == pytest.approx(
        (0.02,) * 4, rel=0.0, abs=1e-15
    )


def test_one_step_state_and_pd_tension_match_analytic_values():
    controller = _controller()
    _reset(controller, velocities=(-0.2,) * 4)
    tension = controller.update(
        (1.0, -1.0, 0.0, 2.0),
        TEST_LENGTHS_M,
        (-0.2,) * 4,
    )

    np.testing.assert_allclose(
        controller.state.contraction_velocity_reference_m_s,
        (0.3, 0.1, 0.2, 0.4),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        controller.state.contraction_reference_m,
        (0.03, 0.01, 0.02, 0.04),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tension,
        (1.11, 0.97, 1.04, 1.18),
        rtol=0.0,
        atol=1e-15,
    )


def test_n_step_constant_acceleration_matches_closed_form():
    dt = 0.05
    acceleration = 0.4
    initial_velocity = 0.3
    steps = 7
    controller = _controller(
        control_timestep_s=dt,
        kp_n_per_m=0.0,
        kd_n_s_per_m=0.0,
        max_acceleration_m_s2=1.0,
        max_velocity_m_s=100.0,
        min_contraction_m=-100.0,
        max_contraction_m=100.0,
        preload_tension_n=0.0,
    )
    _reset(controller, velocities=(-initial_velocity,) * 4)

    for _ in range(steps):
        controller.update(
            (acceleration,) * 4,
            TEST_LENGTHS_M,
            (-initial_velocity,) * 4,
        )

    expected_velocity = initial_velocity + steps * acceleration * dt
    expected_contraction = (
        steps * initial_velocity * dt
        + acceleration * dt ** 2 * steps * (steps + 1) / 2
    )
    np.testing.assert_allclose(
        controller.state.contraction_velocity_reference_m_s,
        (expected_velocity,) * 4,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        controller.state.contraction_reference_m,
        (expected_contraction,) * 4,
        rtol=0.0,
        atol=1e-15,
    )


def test_acceleration_command_is_symmetrically_clipped():
    controller = _controller(max_acceleration_m_s2=1.0)
    _reset(controller)
    controller.update(
        (2.0, -3.0, 1.0, -1.0),
        TEST_LENGTHS_M,
        TEST_ZERO_VECTOR,
    )

    assert controller.state.effective_acceleration_m_s2 == (1.0, -1.0, 1.0, -1.0)
    assert controller.state.acceleration_was_clipped == (True, True, False, False)


def test_reference_velocity_is_clipped_before_position_integration():
    controller = _controller(max_velocity_m_s=0.25)
    _reset(controller)
    controller.update((10.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)

    assert controller.state.contraction_velocity_reference_m_s == (0.25,) * 4
    assert controller.state.contraction_reference_m == pytest.approx(
        (0.025,) * 4, rel=0.0, abs=1e-15
    )
    assert controller.state.velocity_was_clipped == (True,) * 4


@pytest.mark.parametrize(
    "acceleration,bound,expected_contraction",
    [
        (10.0, "upper", 0.05),
        (-10.0, "lower", -0.04),
    ],
)
def test_contraction_limits_project_outward_reference_velocity(
    acceleration, bound, expected_contraction
):
    controller = _controller(
        min_contraction_m=-0.04,
        max_contraction_m=0.05,
    )
    _reset(controller)
    controller.update(
        (acceleration,) * 4,
        TEST_LENGTHS_M,
        TEST_ZERO_VECTOR,
    )

    assert controller.state.contraction_reference_m == (
        expected_contraction,
    ) * 4
    assert controller.state.contraction_velocity_reference_m_s == TEST_ZERO_VECTOR
    assert controller.state.contraction_was_clipped == (True,) * 4
    assert controller.state.reference_velocity_was_projected == (True,) * 4
    if bound == "upper":
        assert expected_contraction == controller.config.max_contraction_m
    else:
        assert expected_contraction == controller.config.min_contraction_m


def test_exact_arrival_at_travel_endpoint_also_projects_velocity():
    controller = _controller(max_contraction_m=0.05)
    _reset(controller)
    controller.update((5.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)

    assert controller.state.contraction_reference_m == (0.05,) * 4
    assert controller.state.contraction_velocity_reference_m_s == TEST_ZERO_VECTOR
    assert controller.state.contraction_was_clipped == (False,) * 4
    assert controller.state.reference_velocity_was_projected == (True,) * 4


def test_zero_acceleration_preserves_nonzero_reference_velocity():
    controller = _controller(preload_tension_n=0.0)
    _reset(controller)
    controller.update((1.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)
    first_velocity = controller.state.contraction_velocity_reference_m_s
    first_contraction = controller.state.contraction_reference_m
    controller.update(TEST_ZERO_VECTOR, TEST_LENGTHS_M, TEST_ZERO_VECTOR)

    assert first_velocity == pytest.approx((0.1,) * 4)
    assert controller.state.contraction_velocity_reference_m_s == first_velocity
    np.testing.assert_allclose(
        controller.state.contraction_reference_m,
        np.asarray(first_contraction) + np.asarray(first_velocity) * 0.1,
        rtol=0.0,
        atol=1e-15,
    )


def test_negative_acceleration_brakes_without_producing_negative_tension():
    controller = _controller(preload_tension_n=0.0)
    _reset(controller)
    controller.update((1.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)
    controller.update((-1.0,) * 4, TEST_LENGTHS_M, TEST_ZERO_VECTOR)

    assert controller.state.contraction_velocity_reference_m_s == TEST_ZERO_VECTOR
    assert controller.state.contraction_reference_m == pytest.approx(
        (0.01,) * 4, rel=0.0, abs=1e-15
    )
    assert all(value >= 0.0 for value in controller.state.tension_command_n)


def test_position_error_pd_term_uses_contraction_sign_convention():
    controller = _controller(
        kd_n_s_per_m=0.0,
        preload_tension_n=0.1,
        max_tension_n=10.0,
    )
    _reset(controller, lengths=(1.0,) * 4)
    tension = controller.update(
        TEST_ZERO_VECTOR,
        (1.1, 0.9, 1.0, 1.0),
        TEST_ZERO_VECTOR,
    )

    np.testing.assert_allclose(
        controller.state.contraction_actual_m,
        (-0.1, 0.1, 0.0, 0.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        controller.state.position_error_m,
        (0.1, -0.1, 0.0, 0.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tension, (0.3, 0.0, 0.1, 0.1), rtol=0.0, atol=1e-15
    )


def test_velocity_error_pd_term_uses_negative_tendon_velocity():
    controller = _controller(
        kp_n_per_m=0.0,
        kd_n_s_per_m=0.5,
        preload_tension_n=0.1,
        max_tension_n=10.0,
    )
    _reset(controller, lengths=(1.0,) * 4)
    tension = controller.update(
        TEST_ZERO_VECTOR,
        (1.0,) * 4,
        (0.2, -0.2, 0.0, 0.0),
    )

    assert controller.state.contraction_velocity_actual_m_s == (-0.2, 0.2, -0.0, -0.0)
    np.testing.assert_allclose(
        controller.state.velocity_error_m_s,
        (0.2, -0.2, 0.0, 0.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tension, (0.2, 0.0, 0.1, 0.1), rtol=0.0, atol=1e-15
    )


def test_preload_is_the_zero_error_tension():
    controller = _controller(preload_tension_n=0.25)
    reset_state = _reset(controller)
    tension = controller.update(
        TEST_ZERO_VECTOR,
        TEST_LENGTHS_M,
        TEST_ZERO_VECTOR,
    )

    assert reset_state.tension_command_n == (0.25,) * 4
    assert tension == (0.25,) * 4


def test_zero_preload_and_zero_initial_error_produce_exact_zero_tension():
    controller = _controller(preload_tension_n=0.0)
    _reset(controller)
    tension = controller.update(
        TEST_ZERO_VECTOR,
        TEST_LENGTHS_M,
        TEST_ZERO_VECTOR,
    )

    assert tension == TEST_ZERO_VECTOR


def test_tension_is_clipped_at_both_one_sided_limits():
    controller = _controller(
        kp_n_per_m=10.0,
        kd_n_s_per_m=0.0,
        max_tension_n=0.5,
        preload_tension_n=0.0,
    )
    _reset(controller, lengths=(1.0,) * 4)
    tension = controller.update(
        TEST_ZERO_VECTOR,
        (2.0, 0.0, 1.0, 1.0),
        TEST_ZERO_VECTOR,
    )

    assert controller.state.unclipped_tension_n == (10.0, -10.0, 0.0, 0.0)
    assert tension == (0.5, 0.0, 0.0, 0.0)
    assert controller.state.tension_was_clipped_high == (True, False, False, False)
    assert controller.state.tension_was_clipped_low == (False, True, False, False)


def test_four_channels_follow_independent_scalar_algebra():
    controller = _controller(
        kp_n_per_m=1.0,
        kd_n_s_per_m=1.0,
        max_tension_n=10.0,
        preload_tension_n=1.0,
    )
    _reset(controller, lengths=(1.0,) * 4)
    tension = controller.update(
        (1.0, -2.0, 3.0, -4.0),
        (1.0, 0.9, 1.1, 1.0),
        (0.1, -0.2, 0.3, -0.4),
    )

    np.testing.assert_allclose(
        controller.state.contraction_reference_m,
        (0.01, -0.02, 0.03, -0.04),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        tension,
        (1.21, 0.48, 1.73, 0.16),
        rtol=0.0,
        atol=1e-15,
    )


def test_returned_output_and_state_cannot_mutate_internal_state():
    controller = _controller()
    state = _reset(controller)
    output = controller.update(
        TEST_ZERO_VECTOR,
        TEST_LENGTHS_M,
        TEST_ZERO_VECTOR,
    )

    assert isinstance(output, tuple)
    assert isinstance(controller.state, TendonAccelerationControllerState)
    with pytest.raises(TypeError):
        output[0] = 5.0
    with pytest.raises(TypeError):
        controller.state.tension_command_n[0] = 5.0
    with pytest.raises(FrozenInstanceError):
        state.tension_command_n = (5.0,) * 4
    assert controller.state.tension_command_n == (1.0,) * 4


def test_all_public_numeric_state_values_are_finite():
    controller = _controller()
    _reset(controller)
    controller.update(
        (1.0, -2.0, 3.0, -4.0),
        TEST_LENGTHS_M,
        (0.1, -0.2, 0.3, -0.4),
    )

    for field in fields(controller.state):
        values = getattr(controller.state, field.name)
        if values and isinstance(values[0], bool):
            continue
        assert all(math.isfinite(value) for value in values), field.name


def test_same_config_reset_and_inputs_are_deterministic():
    left = _controller()
    right = _controller()
    for controller in (left, right):
        _reset(controller, velocities=(0.1, -0.2, 0.3, -0.4))
        for acceleration in (
            (1.0, 2.0, 3.0, 4.0),
            TEST_ZERO_VECTOR,
            (-1.0, -2.0, -3.0, -4.0),
        ):
            controller.update(
                acceleration,
                TEST_LENGTHS_M,
                (0.1, -0.2, 0.3, -0.4),
            )

    assert left.state == right.state
    assert left.state.tension_command_n == right.state.tension_command_n
