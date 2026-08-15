"""Headless open-loop benchmark for the single-CF2X 3 g rigid payload."""

import sys

import mujoco
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


def _make_env(enable_3g_payload):
    return BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics.MJC,
        sim_freq=240,
        ctrl_freq=240,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        enable_3g_payload=enable_3g_payload,
    )


def _body_subtree_mass(model, root_name):
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id < 0:
        raise AssertionError(f"Body {root_name!r} was not found")

    subtree_ids = []
    for body_id in range(model.nbody):
        ancestor_id = body_id
        while True:
            if ancestor_id == root_id:
                subtree_ids.append(body_id)
                break
            parent_id = int(model.body_parentid[ancestor_id])
            if parent_id == ancestor_id:
                break
            ancestor_id = parent_id
    return float(np.sum(model.body_mass[subtree_ids]))


def _run_case(env, rpm):
    observation, _ = env.reset(seed=0)
    initial_z = float(env.pos[0, 2])
    joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint"
    )
    z_dof_address = int(env.model.jnt_dofadr[joint_id]) + 2
    rpm_action = np.full(4, rpm, dtype=float)
    first_qacc_z = None
    delta_z_02 = None
    finite = True

    for step in range(240):
        observation, _, _, _, _ = env.step(rpm_action)
        if step == 0:
            first_qacc_z = float(env.data.qacc[z_dof_address])
        if step == 47:
            delta_z_02 = float(env.pos[0, 2] - initial_z)
        finite = finite and all(
            np.all(np.isfinite(values))
            for values in (
                observation,
                env.data.qpos,
                env.data.qvel,
                env.data.qacc,
                rpm_action,
                env.last_clipped_action,
            )
        )

    return {
        "mass": _body_subtree_mass(env.model, "drone0"),
        "rpm": float(rpm),
        "thrust_to_weight": float(env.MAX_THRUST / env.GRAVITY),
        "first_qacc_z": first_qacc_z,
        "delta_z_02": delta_z_02,
        "delta_z_1": float(env.pos[0, 2] - initial_z),
        "final_vz": float(env.vel[0, 2]),
        "finite": finite,
    }


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _print_results(results):
    header = (
        f"{'case':<24} {'mass (kg)':>10} {'rpm':>12} {'T/W':>8} "
        f"{'qacc_z':>12} {'dz@0.2s':>12} {'dz@1s':>12} {'vz@1s':>12} {'finite':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(
            f"{name:<24} {result['mass']:>10.6f} {result['rpm']:>12.6f} "
            f"{result['thrust_to_weight']:>8.6f} {result['first_qacc_z']:>12.9f} "
            f"{result['delta_z_02']:>12.9f} {result['delta_z_1']:>12.9f} "
            f"{result['final_vz']:>12.9f} {str(result['finite']):>8}"
        )


def main():
    environments = []
    try:
        unloaded = _make_env(enable_3g_payload=False)
        environments.append(unloaded)
        loaded_old_rpm = _make_env(enable_3g_payload=True)
        environments.append(loaded_old_rpm)
        loaded_new_rpm = _make_env(enable_3g_payload=True)
        environments.append(loaded_new_rpm)

        results = {
            "unloaded_original_rpm": _run_case(unloaded, unloaded.HOVER_RPM),
            "loaded_original_rpm": _run_case(loaded_old_rpm, unloaded.HOVER_RPM),
            "loaded_recomputed_rpm": _run_case(
                loaded_new_rpm, loaded_new_rpm.HOVER_RPM
            ),
        }
        _print_results(results)

        unloaded_result = results["unloaded_original_rpm"]
        loaded_old_result = results["loaded_original_rpm"]
        loaded_new_result = results["loaded_recomputed_rpm"]

        _require(abs(unloaded_result["mass"] - 0.027) <= 1e-12,
                 "Unloaded MuJoCo subtree mass is not 0.027 kg")
        _require(abs(loaded_old_result["mass"] - 0.030) <= 1e-12,
                 "Loaded MuJoCo subtree mass is not 0.030 kg")
        payload_id = mujoco.mj_name2id(
            loaded_old_rpm.model, mujoco.mjtObj.mjOBJ_BODY, "drone0_payload"
        )
        _require(payload_id >= 0, "The payload body cannot be queried by name")
        _require(abs(unloaded_result["thrust_to_weight"] - 2.25) <= 1e-12,
                 "Unloaded thrust-to-weight ratio is not 2.25")
        _require(abs(loaded_old_result["thrust_to_weight"] - 2.025) <= 1e-12,
                 "Loaded thrust-to-weight ratio is not 2.025")
        for attribute in (
            "MAX_RPM",
            "MAX_THRUST",
            "MAX_XY_TORQUE",
            "MAX_Z_TORQUE",
        ):
            _require(
                getattr(loaded_old_rpm, attribute) == getattr(unloaded, attribute),
                f"{attribute} changed when the payload was enabled",
            )
        _require(
            np.isclose(
                loaded_new_rpm.HOVER_RPM / unloaded.HOVER_RPM,
                np.sqrt(0.030 / 0.027),
                rtol=1e-12,
                atol=0.0,
            ),
            "Hover RPM mass scaling is incorrect",
        )

        _require(unloaded_result["finite"], "Unloaded case produced NaN or Inf")
        _require(abs(unloaded_result["delta_z_1"]) < 1e-4,
                 "Unloaded case did not maintain altitude")
        _require(abs(unloaded_result["final_vz"]) < 1e-4,
                 "Unloaded case did not maintain zero vertical velocity")

        expected_loaded_acceleration = loaded_old_rpm.G * (
            unloaded.M / loaded_old_rpm.M - 1
        )
        _require(loaded_old_result["finite"], "Loaded old-RPM case produced NaN or Inf")
        _require(
            abs(loaded_old_result["first_qacc_z"] - expected_loaded_acceleration) < 2e-3,
            "Loaded old-RPM vertical acceleration is outside tolerance",
        )
        _require(loaded_old_result["delta_z_02"] < -0.015,
                 "Loaded old-RPM case did not descend clearly within 0.2 s")

        _require(loaded_new_result["finite"], "Loaded new-RPM case produced NaN or Inf")
        _require(abs(loaded_new_result["delta_z_1"]) < 1e-4,
                 "Loaded new-RPM case did not maintain altitude")
        _require(abs(loaded_new_result["final_vz"]) < 1e-4,
                 "Loaded new-RPM case did not maintain zero vertical velocity")

        print("\nPASS: all rigid-payload benchmark acceptance conditions were satisfied")
        return 0
    except AssertionError as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 1
    finally:
        for env in reversed(environments):
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
