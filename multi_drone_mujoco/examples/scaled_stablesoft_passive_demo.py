"""Interactive B4 passive-arm examples; never used for acceptance tests.

The viewer runs in this process.  Ordinary Python exceptions are reported,
but a native GLFW/OpenGL segmentation fault can terminate the process.  If
that happens, run the independent headless diagnostic CLI documented by
``--help``; pytest results do not depend on this viewer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ScaledStableSoftArmSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.utils.enums import ActionType, DroneModel, Physics


JOINT_STIFFNESS = 0.003086
TARGET_DAMPING_RATIO = 0.3
MOUNT_Z = -0.015
_VIEWER_SHUTDOWN_GRACE_SECONDS = 2.0


def _build_viewer_overlay(scenario, sim_time, contacts):
    """Build one MuJoCo viewer overlay entry."""
    return (
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "scenario\ntime\ncontacts",
        f"{scenario}\n{sim_time:.3f} s\n{contacts}",
    )


def _try_update_viewer_overlay(
    viewer,
    scenario,
    sim_time,
    contacts,
    *,
    enabled=True,
):
    """Update the optional overlay, disabling it after API incompatibility."""
    if not enabled:
        return False
    try:
        viewer.set_texts(
            _build_viewer_overlay(scenario, sim_time, contacts)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        print(
            "viewer overlay disabled after set_texts compatibility "
            f"error: {exc}",
            file=sys.stderr,
        )
        return False
    return True


@contextmanager
def _closing_viewer(
    viewer,
    *,
    shutdown_grace_s=_VIEWER_SHUTDOWN_GRACE_SECONDS,
    sleep_fn=time.sleep,
):
    """Close a viewer once, then let its native teardown finish."""
    try:
        yield viewer
    finally:
        close = getattr(viewer, "close", None)
        if callable(close):
            try:
                close()
            finally:
                # MuJoCo's passive viewer closes on a native background
                # thread.  Keep model/data alive until GLFW/X11 teardown has
                # completed before run_viewer reaches env.close().
                sleep_fn(shutdown_grace_s)


def _spec():
    return ScaledStableSoftArmSpec(
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=None,
        target_damping_ratio=TARGET_DAMPING_RATIO,
    )


def _initialize_bend(model, data, direction):
    manifest = build_scaled_stablesoft_arm_mjcf(
        "drone0", _spec(), MOUNT_Z
    ).manifest
    addresses = []
    for name in manifest.ball_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise AssertionError(name)
        addresses.append(
            (
                joint_id,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
        )
    axis = np.zeros(3)
    axis[0 if direction.endswith("x") else 1] = 1.0
    angle = 0.025 * (-1.0 if direction.startswith("-") else 1.0)
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
    for _, qpos_address, _ in addresses:
        data.qpos[qpos_address:qpos_address + 4] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)


def run_viewer(scenario, sim_freq, direction):
    env = BaseAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.MJC,
        sim_freq=sim_freq,
        ctrl_freq=240,
        gui=False,
        record=False,
        obstacles=False,
        vision_attributes=False,
        act_type=ActionType.RPM,
        render_mode=None,
        scaled_stablesoft_arm=_spec(),
        integrator="implicitfast",
    )
    try:
        env.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        if scenario == "open-loop-release":
            _initialize_bend(env.model, env.data, direction)
            duration = 5.0
        else:
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            duration = 1.0
        rpm = np.full(4, env.HOVER_RPM)
        warning_counts = tuple(int(item.number) for item in env.data.warning)
        warning_triggered = False
        launched_viewer = mujoco.viewer.launch_passive(env.model, env.data)
        with _closing_viewer(launched_viewer) as viewer:
            can_sync = callable(getattr(viewer, "sync", None))
            can_check_running = callable(
                getattr(viewer, "is_running", None)
            )
            overlay_enabled = callable(getattr(viewer, "set_texts", None))
            if not overlay_enabled:
                print(
                    "viewer overlay disabled: set_texts is unavailable",
                    file=sys.stderr,
                )
            start_wall = time.perf_counter()
            requested_steps = int(round(duration * sim_freq))
            for step_index in range(1, requested_steps + 1):
                if can_check_running and not viewer.is_running():
                    break
                env._physics(rpm, 0)
                mujoco.mj_step(env.model, env.data)
                current_counts = tuple(
                    int(item.number) for item in env.data.warning
                )
                warning_triggered = warning_triggered or any(
                    current > previous
                    for current, previous in zip(
                        current_counts, warning_counts
                    )
                )
                warning_counts = current_counts
                if not (
                    math.isfinite(float(env.data.time))
                    and np.all(np.isfinite(env.data.qpos))
                    and np.all(np.isfinite(env.data.qvel))
                ):
                    print(
                        "raw state became non-finite; stopping",
                        file=sys.stderr,
                    )
                    break
                mujoco.mj_forward(env.model, env.data)
                current_counts = tuple(
                    int(item.number) for item in env.data.warning
                )
                warning_triggered = warning_triggered or any(
                    current > previous
                    for current, previous in zip(
                        current_counts, warning_counts
                    )
                )
                warning_counts = current_counts
                if warning_triggered:
                    print(
                        "MuJoCo warning latched; stopping",
                        file=sys.stderr,
                    )
                    break
                if (
                    overlay_enabled
                    and step_index % max(1, sim_freq // 10) == 0
                ):
                    overlay_enabled = _try_update_viewer_overlay(
                        viewer,
                        scenario,
                        env.data.time,
                        env.data.ncon,
                        enabled=overlay_enabled,
                    )
                if can_sync:
                    viewer.sync()
                target_wall = start_wall + step_index / sim_freq
                remaining = target_wall - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                if scenario == "open-loop-release" and env.data.ncon:
                    print(f"first contact at t={env.data.time:.7f} s")
                    break
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Independent headless diagnostic: python -m "
            "multi_drone_mujoco.tests._scaled_stablesoft_coupled "
            "--diagnostic-matrix"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("hover", "open-loop-release"),
        default="hover",
    )
    parser.add_argument("--sim-freq", type=int, choices=(2400, 4800), default=2400)
    parser.add_argument("--direction", choices=("+x", "-x", "+y", "-y"), default="+x")
    args = parser.parse_args(argv)
    run_viewer(args.scenario, args.sim_freq, args.direction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
