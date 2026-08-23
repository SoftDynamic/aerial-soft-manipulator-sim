"""C1-A3 fixed-root ideal multi-tendon dynamics acceptance tests.

Equal diagonal or four-tendon commands can produce zero motion and zero
mechanical power in this exactly symmetric fixture.  That does not mean the
tendons are untensioned, does not rule out internal preload, and does not
demonstrate either the presence or absence of a stiffness change.
"""

from types import MappingProxyType

import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_actuation import (
    CORNERS,
    MAX_TENSION_N,
    ActiveTendonRunConfig,
    TendonCommandRunConfig,
    half_cosine_ramp_scale_at_tick,
    half_cosine_tension_at_tick,
    mixed_limit,
    run_single_tendon_characterization,
    run_tendon_command_characterization,
)


TENSION_LEVELS_N = (0.002, 0.005, 0.01)
SMALL_SIGNAL_TENSION_N = 0.002
COMBINATION_VECTORS = {
    "+X": (1.0, 1.0, 0.0, 0.0),
    "-X": (0.0, 0.0, 1.0, 1.0),
    "+Y": (1.0, 0.0, 1.0, 0.0),
    "-Y": (0.0, 1.0, 0.0, 1.0),
    "D1": (1.0, 0.0, 0.0, 1.0),
    "D2": (0.0, 1.0, 1.0, 0.0),
    "C4": (1.0, 1.0, 1.0, 1.0),
}
ADJACENT_NAMES = ("+X", "-X", "+Y", "-Y")
CANCELLATION_NAMES = ("D1", "D2", "C4")
AXIS_CONTRACT = {
    "+X": (0, 1.0, 1),
    "-X": (0, -1.0, 1),
    "+Y": (1, 1.0, 0),
    "-Y": (1, -1.0, 0),
}
ROTATE_Z_PI = np.diag((-1.0, -1.0, 1.0))
ROTATE_Z_POSITIVE_HALF_PI = np.asarray(
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
)


def _command(name, tension_n):
    return tuple(
        tension_n * coefficient
        for coefficient in COMBINATION_VECTORS[name]
    )


def _corner_from_tendon_name(tendon_name):
    matches = [
        corner
        for corner in CORNERS
        if tendon_name.endswith(f"_{corner}_tendon")
    ]
    assert len(matches) == 1, tendon_name
    return matches[0]


def _manifest_command(trace, command):
    by_corner = dict(zip(CORNERS, command))
    return np.asarray(
        [
            by_corner[_corner_from_tendon_name(tendon_name)]
            for tendon_name in trace.tendon_names
        ]
    )


def _manifest_permutation(source, target, corner_transform):
    target_by_corner = {
        _corner_from_tendon_name(name): index
        for index, name in enumerate(target.tendon_names)
    }
    return np.asarray(
        [
            target_by_corner[corner_transform[_corner_from_tendon_name(name)]]
            for name in source.tendon_names
        ],
        dtype=int,
    )


def _maximum_vector_error(candidate, reference):
    return float(np.max(np.linalg.norm(candidate - reference, axis=-1)))


def _trajectory_scale(*arrays):
    return max(
        float(np.max(np.linalg.norm(values, axis=-1)))
        for values in arrays
    )


def _rotation_vectors_to_quaternions(rotation_vectors):
    rotations = np.asarray(rotation_vectors, dtype=float)
    angles = np.linalg.norm(rotations, axis=1)
    quaternions = np.zeros((len(rotations), 4), dtype=float)
    quaternions[:, 0] = 1.0
    nonzero = angles > 0.0
    half_angles = 0.5 * angles[nonzero]
    quaternions[nonzero, 0] = np.cos(half_angles)
    quaternions[nonzero, 1:] = (
        rotations[nonzero]
        / angles[nonzero, np.newaxis]
        * np.sin(half_angles)[:, np.newaxis]
    )
    return quaternions


def _quaternion_geodesic_residual(actual_rotvec, predicted_rotvec):
    actual = _rotation_vectors_to_quaternions(actual_rotvec)
    predicted = _rotation_vectors_to_quaternions(predicted_rotvec)
    absolute_dot = np.abs(np.sum(actual * predicted, axis=1))
    return 2.0 * np.arccos(np.clip(absolute_dot, -1.0, 1.0))


def _diagnostics(name, tension_n, trace):
    return {
        "name": name,
        "tension_n": tension_n,
        "command": trace.config.final_tension_command_n,
        "active_indices": trace.active_indices,
        "steady_tip_m": trace.steady_tip_displacement_m,
        "steady_bend_rad": trace.steady_total_bend_rad,
        "steady_tendon_excursion_m": trace.steady_tendon_excursion_m,
        "safety": trace.safety,
    }


@pytest.fixture(scope="module")
def multi_traces_2400():
    return MappingProxyType(
        {
            (name, tension_n): run_tendon_command_characterization(
                TendonCommandRunConfig(
                    _command(name, tension_n),
                    sim_freq=2400,
                )
            )
            for tension_n in TENSION_LEVELS_N
            for name in COMBINATION_VECTORS
        }
    )


@pytest.fixture(scope="module")
def small_signal_single_traces_2400():
    return MappingProxyType(
        {
            corner: run_single_tendon_characterization(
                ActiveTendonRunConfig(
                    corner,
                    SMALL_SIGNAL_TENSION_N,
                    sim_freq=2400,
                )
            )
            for corner in CORNERS
        }
    )


@pytest.fixture(scope="module")
def representative_traces_4800():
    commands = {
        "+X": _command("+X", MAX_TENSION_N),
        "D1": _command("D1", MAX_TENSION_N),
        "C4": _command("C4", MAX_TENSION_N),
    }
    return MappingProxyType(
        {
            name: run_tendon_command_characterization(
                TendonCommandRunConfig(command, sim_freq=4800)
            )
            for name, command in commands.items()
        }
    )


def test_dimensionless_half_cosine_scale_preserves_tension_api():
    ramp_ticks = 60
    scales = np.asarray(
        [
            half_cosine_ramp_scale_at_tick(tick, ramp_ticks)
            for tick in range(2 * ramp_ticks + 1)
        ]
    )

    assert scales[0] == 0.0
    assert scales[ramp_ticks] == 1.0
    assert np.all(scales[ramp_ticks:] == 1.0)
    assert np.all((0.0 <= scales) & (scales <= 1.0))
    assert np.all(np.diff(scales) >= 0.0)
    np.testing.assert_array_equal(
        [
            half_cosine_tension_at_tick(0.005, tick, ramp_ticks)
            for tick in range(2 * ramp_ticks + 1)
        ],
        0.005 * scales,
    )


def test_tendon_command_config_defensively_normalizes_to_float_tuple():
    source = [0, np.float32(0.002), 0.005, MAX_TENSION_N]
    config = TendonCommandRunConfig(source)
    source[:] = [MAX_TENSION_N] * 4

    assert config.final_tension_command_n == pytest.approx(
        (0.0, 0.002, 0.005, MAX_TENSION_N)
    )
    assert isinstance(config.final_tension_command_n, tuple)
    assert all(type(value) is float for value in config.final_tension_command_n)
    assert config.active_indices == (1, 2, 3)


@pytest.mark.parametrize(
    "command",
    [
        0.002,
        (0.002, 0.0, 0.0),
        (0.002, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0),
        (0.0, True, 0.0, 0.0),
        (0.0, 0.0, False, 0.002),
        (0.0, 0.0, 0.0, True),
        (np.nan, 0.0, 0.0, 0.0),
        (np.inf, 0.0, 0.0, 0.0),
        (-1e-6, 0.0, 0.0, 0.0),
        (MAX_TENSION_N + 1e-6, 0.0, 0.0, 0.0),
    ],
)
def test_tendon_command_config_rejects_invalid_commands(command):
    with pytest.raises(ValueError):
        TendonCommandRunConfig(command)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tension_n": MAX_TENSION_N + 1e-6},
        {"sim_freq": True},
        {"sim_freq": 2500},
        {"integrator": "RK4"},
        {"ramp_duration_s": 0.251},
        {"steady_window_start_s": 3.001},
    ],
)
def test_tendon_command_config_rejects_invalid_run_grid(overrides):
    with pytest.raises(ValueError):
        TendonCommandRunConfig((0.002, 0.0, 0.0, 0.0), **overrides)


def test_legacy_config_cannot_raise_c1_diagnostic_tension_limit():
    with pytest.raises(ValueError, match="C1 active-dynamics diagnostic limit"):
        ActiveTendonRunConfig(
            "xp_yp",
            0.002,
            max_tension_n=MAX_TENSION_N + 1e-6,
        )


@pytest.mark.parametrize("name", tuple(COMBINATION_VECTORS))
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_multi_trace_command_force_binding_grid_and_read_only_contract(
    multi_traces_2400,
    name,
    tension_n,
):
    trace = multi_traces_2400[(name, tension_n)]
    config = trace.config
    final_command = _manifest_command(trace, _command(name, tension_n))
    expected_active = tuple(np.flatnonzero(final_command > 0.0).tolist())
    expected_mask = final_command > 0.0
    expected_scale = np.asarray(
        [
            half_cosine_ramp_scale_at_tick(
                int(tick),
                config.ramp_ticks,
            )
            for tick in trace.control_tick
        ]
    )
    expected_command = expected_scale[:, np.newaxis] * final_command
    diagnostics = _diagnostics(name, tension_n, trace)

    assert isinstance(config, TendonCommandRunConfig)
    assert trace.target_index is None
    assert trace.active_indices == expected_active
    np.testing.assert_array_equal(trace.active_mask, expected_mask)
    assert not trace.active_mask.flags.writeable
    assert all(final_command[index] > 0.0 for index in expected_active)
    assert trace.safety.completed_steps == config.requested_steps == 8400
    np.testing.assert_array_equal(trace.control_tick, np.arange(841))
    np.testing.assert_array_equal(
        trace.physics_step,
        np.arange(841) * config.sample_stride,
    )
    np.testing.assert_allclose(
        trace.time_s,
        np.arange(841) / config.ctrl_freq,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.tension_command_n,
        expected_command,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n,
        trace.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n[:, ~expected_mask],
        0.0,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        trace.actuator_force_n[-1, expected_mask],
        final_command[expected_mask],
        rtol=0.0,
        atol=1e-15,
    )
    for index, binding in enumerate(trace.bindings):
        assert binding.tendon_name == trace.tendon_names[index]
        assert binding.actuator_name == trace.actuator_names[index]
        assert expected_mask[index] == (index in trace.active_indices)

    np.testing.assert_allclose(
        trace.tendon_excursion_m,
        trace.tendon_length_m - trace.tendon_length_m[0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        trace.active_tendon_excursion_m,
        trace.tendon_excursion_m[:, expected_mask],
        rtol=0.0,
        atol=0.0,
    )
    for values in (
        trace.time_s,
        trace.tension_command_n,
        trace.actuator_force_n,
        trace.tip_displacement_m,
        trace.composite_rotation_vectors_rad,
        trace.tendon_length_m,
        trace.tendon_excursion_m,
        trace.steady_tendon_excursion_m,
        trace.active_tendon_excursion_m,
        trace.steady_active_tendon_excursion_m,
        trace.qvel_rad_s,
        trace.qacc_rad_s2,
        trace.command_tendon_power_w,
    ):
        assert not values.flags.writeable
        assert np.all(np.isfinite(values)), diagnostics
    with pytest.raises(ValueError, match="single-tendon"):
        _ = trace.target_tendon_excursion_m
    with pytest.raises(ValueError, match="single-tendon"):
        _ = trace.steady_target_tendon_excursion_m


def test_single_tendon_wrapper_preserves_legacy_trace_contract(
    small_signal_single_traces_2400,
):
    trace = small_signal_single_traces_2400["xp_yp"]
    config = trace.config

    assert isinstance(config, ActiveTendonRunConfig)
    assert config.corner == "xp_yp"
    assert config.final_tension_n == SMALL_SIGNAL_TENSION_N
    assert trace.target_index == CORNERS.index("xp_yp")
    assert trace.active_indices == (trace.target_index,)
    assert trace.steady_target_tendon_excursion_m == pytest.approx(
        trace.steady_tendon_excursion_m[trace.target_index]
    )
    np.testing.assert_array_equal(
        trace.target_tendon_excursion_m,
        trace.tendon_excursion_m[:, trace.target_index],
    )


@pytest.mark.parametrize("name", ADJACENT_NAMES)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_adjacent_equal_tension_synthesizes_pure_world_axis_bending(
    multi_traces_2400,
    name,
    tension_n,
):
    trace = multi_traces_2400[(name, tension_n)]
    main_axis, expected_sign, cross_axis = AXIS_CONTRACT[name]
    steady_tip = trace.steady_tip_displacement_m
    main = float(steady_tip[main_axis])
    cross = abs(float(steady_tip[cross_axis]))
    diagnostics = _diagnostics(name, tension_n, trace)

    assert expected_sign * main >= 5e-4, diagnostics
    assert cross <= mixed_limit(1e-8, 0.005, abs(main)), diagnostics
    assert np.all(trace.steady_active_tendon_excursion_m <= -1e-6), (
        diagnostics
    )
    assert trace.steady_total_bend_rad >= 0.01, diagnostics


@pytest.mark.parametrize("name", ADJACENT_NAMES)
def test_adjacent_response_is_strictly_ordered_by_tension(
    multi_traces_2400,
    name,
):
    main_axis = AXIS_CONTRACT[name][0]
    traces = [multi_traces_2400[(name, value)] for value in TENSION_LEVELS_N]
    main_displacement = [
        abs(float(trace.steady_tip_displacement_m[main_axis]))
        for trace in traces
    ]
    bends = [trace.steady_total_bend_rad for trace in traces]
    active_shortening = [
        float(-np.mean(trace.steady_active_tendon_excursion_m))
        for trace in traces
    ]

    assert main_displacement[0] < main_displacement[1] < main_displacement[2]
    assert bends[0] < bends[1] < bends[2]
    assert active_shortening[0] < active_shortening[1] < active_shortening[2]


@pytest.mark.parametrize("positive,negative", (("+X", "-X"), ("+Y", "-Y")))
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_adjacent_positive_negative_pairs_are_full_trajectory_mirrors(
    multi_traces_2400,
    positive,
    negative,
    tension_n,
):
    left = multi_traces_2400[(positive, tension_n)]
    right = multi_traces_2400[(negative, tension_n)]
    corner_transform = {
        "xp_yp": "xn_yn",
        "xp_yn": "xn_yp",
        "xn_yp": "xp_yn",
        "xn_yn": "xp_yp",
    }
    permutation = _manifest_permutation(left, right, corner_transform)
    transformed_tip = left.tip_displacement_m @ ROTATE_Z_PI.T
    transformed_composite = (
        left.composite_rotation_vectors_rad @ ROTATE_Z_PI.T
    )
    tip_scale = _trajectory_scale(transformed_tip, right.tip_displacement_m)
    bend_scale = _trajectory_scale(
        transformed_composite,
        right.composite_rotation_vectors_rad,
    )
    length_left = left.tendon_excursion_m
    length_right = right.tendon_excursion_m[:, permutation]
    length_scale = max(
        float(np.max(np.abs(length_left))),
        float(np.max(np.abs(length_right))),
    )

    assert _maximum_vector_error(
        transformed_tip,
        right.tip_displacement_m,
    ) <= mixed_limit(1e-8, 0.005, tip_scale)
    assert _maximum_vector_error(
        transformed_composite,
        right.composite_rotation_vectors_rad,
    ) <= mixed_limit(1e-7, 0.005, bend_scale)
    assert float(np.max(np.abs(length_left - length_right))) <= mixed_limit(
        1e-10,
        0.005,
        length_scale,
    )


def _constituent_single_traces(name, single_traces):
    return tuple(
        single_traces[corner]
        for corner, coefficient in zip(CORNERS, COMBINATION_VECTORS[name])
        if coefficient > 0.0
    )


@pytest.mark.parametrize("name", CANCELLATION_NAMES)
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_diagonal_and_equal_four_tendon_commands_cancel_bending(
    multi_traces_2400,
    small_signal_single_traces_2400,
    name,
    tension_n,
):
    trace = multi_traces_2400[(name, tension_n)]
    constituents = _constituent_single_traces(
        name,
        small_signal_single_traces_2400,
    )
    scale_factor = tension_n / SMALL_SIGNAL_TENSION_N
    constituent_tip_scale = scale_factor * max(
        float(np.linalg.norm(item.steady_tip_displacement_m[:2]))
        for item in constituents
    )
    constituent_bend_scale = scale_factor * max(
        item.steady_total_bend_rad for item in constituents
    )
    steady_xy = float(np.linalg.norm(trace.steady_tip_displacement_m[:2]))
    diagnostics = _diagnostics(name, tension_n, trace)

    assert steady_xy <= mixed_limit(
        1e-8,
        0.005,
        constituent_tip_scale,
    ), diagnostics
    assert trace.steady_total_bend_rad <= mixed_limit(
        1e-7,
        0.005,
        constituent_bend_scale,
    ), diagnostics
    assert np.all(np.isfinite(trace.tip_displacement_m[:, 2])), diagnostics

    active_excursion = trace.steady_active_tendon_excursion_m
    active_scale = float(np.max(np.abs(active_excursion)))
    assert float(np.ptp(active_excursion)) <= mixed_limit(
        1e-10,
        0.005,
        active_scale,
    ), diagnostics
    if name == "C4":
        all_scale = float(np.max(np.abs(trace.steady_tendon_excursion_m)))
        assert float(np.ptp(trace.steady_tendon_excursion_m)) <= mixed_limit(
            1e-10,
            0.005,
            all_scale,
        ), diagnostics


@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_diagonal_pairs_are_full_trajectory_quarter_turn_mirrors(
    multi_traces_2400,
    tension_n,
):
    d1 = multi_traces_2400[("D1", tension_n)]
    d2 = multi_traces_2400[("D2", tension_n)]
    corner_transform = {
        "xp_yp": "xn_yp",
        "xp_yn": "xp_yp",
        "xn_yp": "xn_yn",
        "xn_yn": "xp_yn",
    }
    permutation = _manifest_permutation(d1, d2, corner_transform)
    transformed_tip = d1.tip_displacement_m @ ROTATE_Z_POSITIVE_HALF_PI.T
    transformed_composite = (
        d1.composite_rotation_vectors_rad
        @ ROTATE_Z_POSITIVE_HALF_PI.T
    )

    np.testing.assert_allclose(
        d1.tension_command_n,
        d2.tension_command_n[:, permutation],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        d1.actuator_force_n,
        d2.actuator_force_n[:, permutation],
        rtol=0.0,
        atol=1e-15,
    )
    assert _maximum_vector_error(
        transformed_tip,
        d2.tip_displacement_m,
    ) <= mixed_limit(
        1e-8,
        0.005,
        _trajectory_scale(transformed_tip, d2.tip_displacement_m),
    )
    assert _maximum_vector_error(
        transformed_composite,
        d2.composite_rotation_vectors_rad,
    ) <= mixed_limit(
        1e-7,
        0.005,
        _trajectory_scale(
            transformed_composite,
            d2.composite_rotation_vectors_rad,
        ),
    )
    np.testing.assert_allclose(
        d1.tendon_excursion_m,
        d2.tendon_excursion_m[:, permutation],
        rtol=0.0,
        atol=1e-10,
    )


def _local_action_matrices(single_traces):
    exemplar = single_traces[CORNERS[0]]
    manifest_corners = tuple(
        _corner_from_tendon_name(name) for name in exemplar.tendon_names
    )
    tip_columns = []
    bend_columns = []
    for corner in manifest_corners:
        trace = single_traces[corner]
        assert trace.tendon_names == exemplar.tendon_names
        tip_columns.append(
            trace.steady_tip_displacement_m[:2]
            / SMALL_SIGNAL_TENSION_N
        )
        bend_columns.append(
            np.mean(
                trace.composite_rotation_vectors_rad[trace.steady_mask, :2],
                axis=0,
            )
            / SMALL_SIGNAL_TENSION_N
        )
    return (
        manifest_corners,
        np.column_stack(tip_columns),
        np.column_stack(bend_columns),
    )


def test_runtime_small_signal_action_matrices_have_geometric_symmetry(
    small_signal_single_traces_2400,
):
    manifest_corners, a_tip, a_bend = _local_action_matrices(
        small_signal_single_traces_2400
    )
    index = {corner: column for column, corner in enumerate(manifest_corners)}
    opposite_pairs = (("xp_yp", "xn_yn"), ("xp_yn", "xn_yp"))

    for matrix, absolute_floor in ((a_tip, 1e-8), (a_bend, 1e-7)):
        norms = np.linalg.norm(matrix, axis=0)
        scale = float(np.max(norms))
        assert float(np.ptp(norms)) <= mixed_limit(
            absolute_floor,
            0.005,
            scale,
        )
        assert np.linalg.norm(np.sum(matrix, axis=1)) <= mixed_limit(
            absolute_floor,
            0.005,
            scale,
        )
        for left, right in opposite_pairs:
            assert np.linalg.norm(
                matrix[:, index[left]] + matrix[:, index[right]]
            ) <= mixed_limit(absolute_floor, 0.005, scale)

    expected_tip_signs = {
        "xp_yp": (1, 1),
        "xp_yn": (1, -1),
        "xn_yp": (-1, 1),
        "xn_yn": (-1, -1),
    }
    for corner, tip_signs in expected_tip_signs.items():
        tip_column = a_tip[:, index[corner]]
        bend_column = a_bend[:, index[corner]]
        assert tuple(np.sign(tip_column).astype(int)) == tip_signs
        assert tuple(np.sign(bend_column).astype(int)) == (
            tip_signs[1],
            -tip_signs[0],
        )
        assert abs(abs(tip_column[0]) - abs(tip_column[1])) <= mixed_limit(
            1e-8,
            0.005,
            float(np.max(np.abs(tip_column))),
        )
        assert abs(abs(bend_column[0]) - abs(bend_column[1])) <= mixed_limit(
            1e-7,
            0.005,
            float(np.max(np.abs(bend_column))),
        )


@pytest.mark.parametrize("name", tuple(COMBINATION_VECTORS))
def test_runtime_small_signal_matrices_predict_all_combinations(
    multi_traces_2400,
    small_signal_single_traces_2400,
    name,
):
    trace = multi_traces_2400[(name, SMALL_SIGNAL_TENSION_N)]
    manifest_corners, a_tip, a_bend = _local_action_matrices(
        small_signal_single_traces_2400
    )
    command_by_corner = dict(
        zip(CORNERS, _command(name, SMALL_SIGNAL_TENSION_N))
    )
    command = np.asarray(
        [command_by_corner[corner] for corner in manifest_corners]
    )
    predicted_tip = a_tip @ command
    actual_tip = trace.steady_tip_displacement_m[:2]
    predicted_bend = a_bend @ command
    actual_bend = np.mean(
        trace.composite_rotation_vectors_rad[trace.steady_mask, :2],
        axis=0,
    )
    tip_scale = max(np.linalg.norm(actual_tip), np.linalg.norm(predicted_tip))
    bend_scale = max(
        np.linalg.norm(actual_bend),
        np.linalg.norm(predicted_bend),
    )

    assert np.linalg.norm(actual_tip - predicted_tip) <= mixed_limit(
        1e-8,
        0.005,
        float(tip_scale),
    )
    assert np.linalg.norm(actual_bend - predicted_bend) <= mixed_limit(
        1e-7,
        0.005,
        float(bend_scale),
    )


@pytest.mark.parametrize("name", tuple(COMBINATION_VECTORS))
def test_small_signal_full_trajectory_superposition(
    multi_traces_2400,
    small_signal_single_traces_2400,
    name,
):
    trace = multi_traces_2400[(name, SMALL_SIGNAL_TENSION_N)]
    constituents = _constituent_single_traces(
        name,
        small_signal_single_traces_2400,
    )
    predicted_tip = sum(
        (item.tip_displacement_m for item in constituents),
        np.zeros_like(trace.tip_displacement_m),
    )
    predicted_length = sum(
        (item.tendon_excursion_m for item in constituents),
        np.zeros_like(trace.tendon_excursion_m),
    )
    predicted_qvel = sum(
        (item.qvel_rad_s for item in constituents),
        np.zeros_like(trace.qvel_rad_s),
    )
    predicted_rotation = sum(
        (item.composite_rotation_vectors_rad for item in constituents),
        np.zeros_like(trace.composite_rotation_vectors_rad),
    )

    actual_tip_xy = trace.tip_displacement_m[:, :2]
    predicted_tip_xy = predicted_tip[:, :2]
    tip_scale = _trajectory_scale(actual_tip_xy, predicted_tip_xy)
    tip_limit = mixed_limit(1e-8, 0.005, tip_scale)
    steady_actual = np.mean(actual_tip_xy[trace.steady_mask], axis=0)
    steady_predicted = np.mean(predicted_tip_xy[trace.steady_mask], axis=0)

    length_scale = max(
        float(np.max(np.abs(trace.tendon_excursion_m))),
        float(np.max(np.abs(predicted_length))),
    )
    length_limit = mixed_limit(5e-8, 0.005, length_scale)
    qvel_scale = _trajectory_scale(trace.qvel_rad_s, predicted_qvel)
    qvel_limit = mixed_limit(1e-6, 0.005, qvel_scale)
    angle_scale = _trajectory_scale(
        trace.composite_rotation_vectors_rad,
        predicted_rotation,
    )
    orientation_limit = mixed_limit(1e-7, 0.005, angle_scale)
    geodesic = _quaternion_geodesic_residual(
        trace.composite_rotation_vectors_rad,
        predicted_rotation,
    )

    # Power is bilinear in force and velocity, so this remains diagnostic;
    # the strict power checks are the four identities in the safety test.
    actual_peak_power = float(
        np.max(np.abs(trace.actuator_space_power_w))
    )
    predicted_sum_power = sum(
        item.actuator_space_power_w for item in constituents
    )
    power_characterization = (
        actual_peak_power,
        float(np.max(np.abs(predicted_sum_power))),
    )
    diagnostics = {
        "name": name,
        "tip_error": _maximum_vector_error(actual_tip_xy, predicted_tip_xy),
        "tip_limit": tip_limit,
        "steady_tip_error": np.linalg.norm(steady_actual - steady_predicted),
        "length_error": float(
            np.max(np.abs(trace.tendon_excursion_m - predicted_length))
        ),
        "length_limit": length_limit,
        "qvel_error": _maximum_vector_error(trace.qvel_rad_s, predicted_qvel),
        "qvel_limit": qvel_limit,
        "geodesic_error": float(np.max(geodesic)),
        "orientation_limit": orientation_limit,
        "peak_power_characterization": power_characterization,
    }

    assert diagnostics["tip_error"] <= tip_limit, diagnostics
    assert diagnostics["steady_tip_error"] <= tip_limit, diagnostics
    assert diagnostics["length_error"] <= length_limit, diagnostics
    assert diagnostics["qvel_error"] <= qvel_limit, diagnostics
    assert abs(
        float(np.max(np.linalg.norm(trace.qvel_rad_s, axis=1)))
        - float(np.max(np.linalg.norm(predicted_qvel, axis=1)))
    ) <= qvel_limit, diagnostics
    assert diagnostics["geodesic_error"] <= orientation_limit, diagnostics
    assert np.all(np.isfinite(power_characterization)), diagnostics


@pytest.mark.parametrize("name", tuple(COMBINATION_VECTORS))
@pytest.mark.parametrize("tension_n", TENSION_LEVELS_N)
def test_multi_tendon_physics_rate_safety_twist_moment_and_power(
    multi_traces_2400,
    name,
    tension_n,
):
    trace = multi_traces_2400[(name, tension_n)]
    safety = trace.safety
    diagnostics = _diagnostics(name, tension_n, trace)

    assert safety.first_failure is None, diagnostics
    assert safety.warning_events == (), diagnostics
    assert safety.maximum_contacts == 0, diagnostics
    assert safety.completed_steps == safety.requested_steps == 8400
    assert safety.minimum_tendon_range_margin_m >= -1e-9, diagnostics
    assert safety.maximum_force_command_error_n <= 1e-15, diagnostics
    assert safety.maximum_moment_residual <= 1e-12, diagnostics
    assert safety.maximum_generalized_actuator_power_residual_w <= 1e-12
    assert safety.maximum_actuator_tendon_power_residual_w <= 1e-12
    assert safety.maximum_command_tendon_power_residual_w <= 1e-12
    assert safety.maximum_actuator_length_sign_residual_m <= 1e-12
    assert safety.maximum_actuator_velocity_sign_residual_m_s <= 1e-12
    assert safety.maximum_abs_joint_twist_rad <= mixed_limit(
        1e-6,
        0.01,
        safety.maximum_joint_bend_rad,
    ), diagnostics
    assert safety.maximum_abs_composite_twist_rad <= mixed_limit(
        1e-6,
        0.01,
        safety.maximum_composite_bend_rad,
    ), diagnostics
    assert np.all(trace.warning_counts == 0), diagnostics
    assert np.all(trace.contacts == 0), diagnostics
    assert np.all(trace.tendon_range_margin_m >= -1e-9), diagnostics
    np.testing.assert_allclose(
        trace.generalized_actuator_power_w,
        trace.actuator_space_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.actuator_space_power_w,
        trace.tendon_coordinate_power_w,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.tendon_coordinate_power_w,
        trace.command_tendon_power_w,
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", ("+X", "D1", "C4"))
def test_representative_multi_tendon_2400_converges_to_4800(
    multi_traces_2400,
    representative_traces_4800,
    name,
):
    candidate = multi_traces_2400[(name, MAX_TENSION_N)]
    reference = representative_traces_4800[name]

    np.testing.assert_array_equal(candidate.control_tick, reference.control_tick)
    np.testing.assert_allclose(candidate.time_s, reference.time_s, atol=1e-12)
    np.testing.assert_allclose(
        candidate.tension_command_n,
        reference.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        candidate.actuator_force_n,
        candidate.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        reference.actuator_force_n,
        reference.tension_command_n,
        rtol=0.0,
        atol=1e-15,
    )
    assert candidate.safety.completed_steps == 8400
    assert reference.safety.completed_steps == 16800

    tip_error = _maximum_vector_error(
        candidate.tip_displacement_m,
        reference.tip_displacement_m,
    )
    tip_scale = _trajectory_scale(
        candidate.tip_displacement_m,
        reference.tip_displacement_m,
    )
    joint_error = _maximum_vector_error(
        candidate.joint_rotation_vectors_rad,
        reference.joint_rotation_vectors_rad,
    )
    joint_scale = _trajectory_scale(
        candidate.joint_rotation_vectors_rad,
        reference.joint_rotation_vectors_rad,
    )
    composite_error = _maximum_vector_error(
        candidate.composite_rotation_vectors_rad,
        reference.composite_rotation_vectors_rad,
    )
    composite_scale = _trajectory_scale(
        candidate.composite_rotation_vectors_rad,
        reference.composite_rotation_vectors_rad,
    )
    reference_excursion = reference.tendon_excursion_m
    tendon_length_error = float(
        np.max(np.abs(candidate.tendon_length_m - reference.tendon_length_m))
    )
    candidate_velocity_extrema = np.stack(
        (candidate.tendon_velocity_m_s.min(0), candidate.tendon_velocity_m_s.max(0))
    )
    reference_velocity_extrema = np.stack(
        (reference.tendon_velocity_m_s.min(0), reference.tendon_velocity_m_s.max(0))
    )
    tendon_velocity_error = float(
        np.max(np.abs(candidate_velocity_extrema - reference_velocity_extrema))
    )
    steady_tip_error = float(
        np.linalg.norm(
            candidate.steady_tip_displacement_m
            - reference.steady_tip_displacement_m
        )
    )
    steady_bend_error = abs(
        candidate.steady_total_bend_rad - reference.steady_total_bend_rad
    )
    candidate_peak_qvel = float(np.max(candidate.maximum_arm_qvel_rad_s))
    reference_peak_qvel = float(np.max(reference.maximum_arm_qvel_rad_s))
    candidate_peak_power = float(
        np.max(np.abs(candidate.actuator_space_power_w))
    )
    reference_peak_power = float(
        np.max(np.abs(reference.actuator_space_power_w))
    )
    diagnostics = {
        "name": name,
        "tip_error": tip_error,
        "joint_error": joint_error,
        "composite_error": composite_error,
        "tendon_length_error": tendon_length_error,
        "tendon_velocity_error": tendon_velocity_error,
        "steady_tip_error": steady_tip_error,
        "steady_bend_error": steady_bend_error,
        "peak_qvel_error": abs(candidate_peak_qvel - reference_peak_qvel),
        "peak_power_error": abs(candidate_peak_power - reference_peak_power),
    }

    assert tip_error <= mixed_limit(2e-5, 0.02, tip_scale), diagnostics
    assert joint_error <= mixed_limit(1e-4, 0.02, joint_scale), diagnostics
    assert composite_error <= mixed_limit(
        1e-4,
        0.02,
        composite_scale,
    ), diagnostics
    assert tendon_length_error <= mixed_limit(
        2e-8,
        0.02,
        float(np.max(np.abs(reference_excursion))),
    ), diagnostics
    assert tendon_velocity_error <= mixed_limit(
        1e-6,
        0.02,
        float(np.max(np.abs(reference_velocity_extrema))),
    ), diagnostics
    assert steady_tip_error <= mixed_limit(
        1e-8,
        0.02,
        float(np.linalg.norm(reference.steady_tip_displacement_m)),
    ), diagnostics
    assert steady_bend_error <= mixed_limit(
        1e-8,
        0.02,
        reference.steady_total_bend_rad,
    ), diagnostics
    assert abs(candidate_peak_qvel - reference_peak_qvel) <= mixed_limit(
        1e-8,
        0.02,
        reference_peak_qvel,
    ), diagnostics
    assert abs(candidate_peak_power - reference_peak_power) <= mixed_limit(
        1e-12,
        0.02,
        reference_peak_power,
    ), diagnostics
    final_qvel_error = np.abs(
        candidate.qvel_rad_s[-1] - reference.qvel_rad_s[-1]
    )
    reference_qvel_amplitude = np.max(
        np.abs(reference.qvel_rad_s),
        axis=0,
    )
    assert np.all(
        final_qvel_error
        <= np.maximum(1e-5, 0.02 * reference_qvel_amplitude)
    ), diagnostics
