"""B3 stability, convergence, and diagnostic-protocol tests."""

import math

import mujoco
import numpy as np
import pytest

from multi_drone_mujoco.tests._scaled_stablesoft_stability import (
    ANALYSIS_AVAILABLE,
    CONVERGED,
    ENERGY_CONSERVATIVE,
    ENERGY_DISSIPATIVE,
    ENERGY_JSON_PREFIX,
    ENERGY_NUMERICAL_INJECTION,
    MATRIX_CHILD_ENV,
    QACC_EVIDENCE_LIMIT,
    RUN_COMPLETED,
    RUN_RAW_NONFINITE,
    SCHEMA_VERSION,
    PersistenceTracker,
    StabilityRunConfig,
    _run_child,
    compare_to_reference,
    length_tolerance,
    main,
    run_stability_case,
)


@pytest.fixture(scope="module")
def stable_results():
    configs = {
        "h1_rk4_2400": StabilityRunConfig("H1", "RK4", 2400, "x"),
        "h1_rk4_4800": StabilityRunConfig("H1", "RK4", 4800, "x"),
        "h1_rk4_9600": StabilityRunConfig("H1", "RK4", 9600, "x"),
        "h1_implicit_9600": StabilityRunConfig(
            "H1", "implicitfast", 9600, "x"
        ),
        "h2_implicit_2400_x": StabilityRunConfig(
            "H2", "implicitfast", 2400, "x"
        ),
        "h2_implicit_4800_x": StabilityRunConfig(
            "H2", "implicitfast", 4800, "x"
        ),
        "h2_implicit_9600_x": StabilityRunConfig(
            "H2", "implicitfast", 9600, "x"
        ),
        "h2_implicit_2400_y": StabilityRunConfig(
            "H2", "implicitfast", 2400, "y"
        ),
    }
    return {name: run_stability_case(config) for name, config in configs.items()}


@pytest.mark.parametrize(
    "sim_freq,expected_steps",
    [(240, 1), (2400, 10), (4800, 20), (9600, 40)],
)
def test_b3_persistence_uses_one_control_interval(sim_freq, expected_steps):
    config = StabilityRunConfig(
        "H1",
        "RK4",
        sim_freq,
        physical_duration=1.0 / 240,
    )
    tracker = PersistenceTracker()
    for _ in range(expected_steps):
        tracker.update(True, 1.0 / sim_freq)

    assert config.persistence_steps == expected_steps
    assert tracker.reached(expected_steps, 1.0 / 240)
    assert tracker.current_steps == expected_steps
    assert tracker.current_duration == pytest.approx(1.0 / 240)


def test_b3_energy_scale_is_compiled_and_not_double_counted(stable_results):
    scale = stable_results["h1_rk4_2400"].energy_scale

    assert scale.delta_ug_scale == pytest.approx(1.1772e-3, rel=1e-12)
    assert scale.elastic_90 == pytest.approx(4.7589998722e-4, rel=1e-11)
    assert scale.total == pytest.approx(
        scale.delta_ug_scale + scale.elastic_90,
        rel=1e-12,
    )
    assert scale.tolerance == pytest.approx(1e-4 * scale.total)
    assert scale.hard_runaway == pytest.approx(5 * scale.total)
    assert scale.single_step_hard_injection == pytest.approx(scale.total)


def test_b3_metadata_records_solver_and_runtime_environment(stable_results):
    result = stable_results["h1_rk4_2400"]
    metadata = result.metadata

    assert metadata.schema_version == SCHEMA_VERSION
    assert metadata.python_version
    assert metadata.numpy_version == np.__version__
    assert metadata.mujoco_python_version == mujoco.__version__
    if hasattr(mujoco, "mj_versionString"):
        assert metadata.mujoco_runtime_version == mujoco.mj_versionString()
    assert metadata.integrator_name == "mjINT_RK4"
    assert metadata.integrator_enum == mujoco.mjtIntegrator.mjINT_RK4
    assert metadata.timestep == pytest.approx(1.0 / 2400)
    assert metadata.solver_name
    assert metadata.iterations > 0
    assert metadata.ls_iterations >= 0
    assert metadata.tolerance > 0
    assert metadata.ls_tolerance > 0
    assert metadata.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY


def test_b3_h1_rk4_energy_error_converges_with_timestep(stable_results):
    results = [
        stable_results["h1_rk4_2400"],
        stable_results["h1_rk4_4800"],
        stable_results["h1_rk4_9600"],
    ]

    assert all(result.run_status == RUN_COMPLETED for result in results)
    assert all(result.energy_status == ENERGY_CONSERVATIVE for result in results)
    assert all(result.analysis_status == ANALYSIS_AVAILABLE for result in results)
    assert all(result.warning_events == [] for result in results)
    drifts = [result.energy["maximum_absolute_drift"] for result in results]
    assert drifts[0] > drifts[1] > drifts[2]
    for result in results:
        assert result.energy["maximum_absolute_drift_time"] >= 0
        assert math.isfinite(result.energy["normalized_maximum_absolute_drift"])
        assert math.isfinite(result.energy["normalized_final_drift"])


def test_b3_h1_implicitfast_completes_but_is_not_conservative_reference(
    stable_results,
):
    result = stable_results["h1_implicit_9600"]

    assert result.run_status == RUN_COMPLETED
    assert result.energy_status == ENERGY_NUMERICAL_INJECTION
    assert (
        result.energy["maximum_absolute_drift"]
        > result.energy_scale.tolerance
    )
    assert result.energy["maximum_positive_drift"] > 0


@pytest.mark.parametrize(
    "name",
    [
        "h2_implicit_2400_x",
        "h2_implicit_4800_x",
        "h2_implicit_9600_x",
        "h2_implicit_2400_y",
    ],
)
def test_b3_h2_stable_candidates_complete_with_dissipative_energy(
    stable_results, name
):
    result = stable_results[name]

    assert result.run_status == RUN_COMPLETED
    assert result.energy_status == ENERGY_DISSIPATIVE
    assert result.analysis_status == ANALYSIS_AVAILABLE
    assert result.completed_steps == result.requested_steps
    assert result.warning_events == []
    assert result.metrics["maximum_contacts"] == 0
    assert np.all(np.isfinite(result.metrics["tendon_final"]))
    for length, (lower, upper) in zip(
        result.metrics["tendon_final"],
        result.metrics["tendon_ranges"],
    ):
        tolerance = length_tolerance(
            length,
            lower,
            upper,
            0.08,
        )
        assert length >= lower - tolerance
        assert length <= upper + tolerance
    geometry_tolerance = length_tolerance(
        result.metrics["maximum_root_tip_length"],
        0.08,
    )
    assert result.metrics["maximum_root_tip_length"] <= (
        0.08 + geometry_tolerance
    )


@pytest.mark.parametrize(
    "candidate_name",
    ["h2_implicit_2400_x", "h2_implicit_4800_x"],
)
def test_b3_h2_candidate_converges_to_high_frequency_comparison(
    stable_results, candidate_name
):
    status, details = compare_to_reference(
        stable_results[candidate_name],
        stable_results["h2_implicit_9600_x"],
    )

    assert status == CONVERGED, details


def test_b3_h2_x_y_candidate_is_mirror_symmetric(stable_results):
    x_result = stable_results["h2_implicit_2400_x"]
    y_result = stable_results["h2_implicit_2400_y"]

    assert x_result.metrics["final_total_bend"] == pytest.approx(
        y_result.metrics["final_total_bend"],
        rel=0.005,
        abs=1e-5,
    )
    assert x_result.energy["final_drift"] == pytest.approx(
        y_result.energy["final_drift"],
        rel=0.005,
        abs=x_result.energy_scale.tolerance,
    )
    assert x_result.analysis["frequency"] == pytest.approx(
        y_result.analysis["frequency"],
        rel=0.02,
    )
    assert x_result.analysis["settling_time"] == pytest.approx(
        y_result.analysis["settling_time"],
        abs=0.02,
    )


def test_b3_qacc_spike_is_evidence_not_an_independent_stop(monkeypatch):
    original_forward = mujoco.mj_forward

    def high_qacc_forward(model, data):
        original_forward(model, data)
        data.qacc[:] = 2 * QACC_EVIDENCE_LIMIT

    monkeypatch.setattr(mujoco, "mj_forward", high_qacc_forward)
    result = run_stability_case(
        StabilityRunConfig(
            "H1",
            "RK4",
            2400,
            "x",
            physical_duration=1.0 / 2400,
        )
    )

    assert result.run_status == RUN_COMPLETED
    assert result.metrics["qacc_evidence_limit_exceeded"]


def test_b3_raw_nonfinite_stops_before_post_step_forward(monkeypatch):
    original_forward = mujoco.mj_forward
    forward_times = []

    def tracking_forward(model, data):
        forward_times.append(float(data.time))
        return original_forward(model, data)

    def nonfinite_step(model, data):
        data.time += model.opt.timestep
        data.qpos[:] = np.nan

    monkeypatch.setattr(mujoco, "mj_forward", tracking_forward)
    monkeypatch.setattr(mujoco, "mj_step", nonfinite_step)
    result = run_stability_case(
        StabilityRunConfig(
            "H1",
            "RK4",
            2400,
            "x",
            physical_duration=1.0 / 2400,
        )
    )

    assert result.run_status == RUN_RAW_NONFINITE
    assert result.completed_steps == 0
    assert result.first_failure["stage"] == "after_step"
    assert all(time_value == 0.0 for time_value in forward_times)


def test_b3_child_protocol_returns_one_versioned_payload():
    result = _run_child(
        StabilityRunConfig(
            "H1",
            "RK4",
            2400,
            "x",
            physical_duration=0.01,
        ),
        timeout=30.0,
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["run_status"] == RUN_COMPLETED
    assert result["metadata"]["python_version"]
    assert result["metadata"]["numpy_version"]
    assert result["metadata"]["mujoco_runtime_version"]
    assert result["completed_steps"] == result["requested_steps"]


def test_b3_matrix_mode_rejects_child_recursion(monkeypatch, capsys):
    monkeypatch.setenv(MATRIX_CHILD_ENV, "1")

    assert main(["--matrix"]) == 2
    assert "recursion" in capsys.readouterr().err


def test_b3_json_prefix_is_stable_for_parent_protocol():
    assert ENERGY_JSON_PREFIX == "B3_RESULT_JSON="
