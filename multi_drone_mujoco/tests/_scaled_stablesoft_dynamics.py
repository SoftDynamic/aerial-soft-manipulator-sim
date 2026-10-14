"""Test-only dynamics helpers for the scaled StableSoft arm."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import mujoco
import numpy as np

from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ArmMjcfManifest,
    ScaledStableSoftArmSpec,
    ScaledStableSoftVisualSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)


MOUNT_Z = -0.015
_SQRT_HALF = math.sqrt(0.5)
_FIXTURE_QUATERNIONS = {
    # The arm's nominal -Z centerline is rotated into the horizontal X/Y axis.
    "x": (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
    "y": (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
    # B2 releases the nominal arm vertically downward under gravity.
    "vertical": (1.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class FixedArmFixture:
    model: mujoco.MjModel
    data: mujoco.MjData
    manifest: ArmMjcfManifest
    sample_stride: int


def _quaternion_text(quaternion) -> str:
    return " ".join(format(float(value), ".17g") for value in quaternion)


def build_fixed_arm_fixture(
    spec: ScaledStableSoftArmSpec,
    plane: str,
    *,
    sim_freq: int,
    ctrl_freq: int,
    integrator: str,
    tendon_drive: Optional[IdealTendonTensionSpec] = None,
    visual: Optional[ScaledStableSoftVisualSpec] = None,
) -> FixedArmFixture:
    """Compile the production arm fragment under a welded test fixture."""
    if plane not in _FIXTURE_QUATERNIONS:
        raise ValueError("plane must be 'x', 'y', or 'vertical'")
    if sim_freq <= 0 or ctrl_freq <= 0 or sim_freq % ctrl_freq != 0:
        raise ValueError("sim_freq must be positive and divisible by ctrl_freq")
    if integrator not in ("RK4", "implicitfast"):
        raise ValueError("integrator must be RK4 or implicitfast")

    fragments = build_scaled_stablesoft_arm_mjcf(
        "fixture",
        spec,
        MOUNT_Z,
        tendon_drive=tendon_drive,
        visual=visual,
    )
    fixture_quaternion = _quaternion_text(_FIXTURE_QUATERNIONS[plane])
    actuator_section = ""
    if fragments.actuator_xml:
        actuator_section = f'''  <actuator>
{fragments.actuator_xml}
  </actuator>
'''
    asset_section = ""
    if fragments.asset_xml:
        asset_section = f'''  <asset>
{fragments.asset_xml}
  </asset>
'''
    xml = f'''<mujoco model="scaled_stablesoft_fixed_{plane}">
  <option gravity="0 0 -9.81" timestep="{1.0 / sim_freq}" integrator="{integrator}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>
{asset_section}  <worldbody>
    <body name="fixture_mount" quat="{fixture_quaternion}">
{fragments.worldbody_xml}    </body>
  </worldbody>
  <tendon>
{fragments.tendon_xml}
  </tendon>
{actuator_section}  <sensor>
{fragments.sensor_xml}
  </sensor>
</mujoco>
'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(model, data)
    return FixedArmFixture(
        model=model,
        data=data,
        manifest=fragments.manifest,
        sample_stride=sim_freq // ctrl_freq,
    )


def named_joint_addresses(model, manifest):
    """Return joint IDs and qpos/qvel addresses without global index literals."""
    result = []
    for name in manifest.ball_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise AssertionError(name)
        result.append(
            (
                joint_id,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
        )
    return tuple(result)


def _rotation_vector(quaternion):
    result = np.zeros(3, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float).copy()
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    mujoco.mju_quat2Vel(result, quaternion, 1.0)
    return result


def joint_rotation_vectors(model, data, addresses):
    return np.asarray(
        [
            _rotation_vector(data.qpos[qpos_address:qpos_address + 4])
            for _, qpos_address, _ in addresses
        ]
    )


def initialize_constant_curvature(
    fixture: FixedArmFixture,
    bend_axis: str,
    *,
    total_bend: float = 0.2,
):
    """Set equal local ball-joint bends without changing model.qpos0."""
    if bend_axis not in ("x", "y"):
        raise ValueError("bend_axis must be 'x' or 'y'")
    if not math.isfinite(total_bend) or total_bend <= 0:
        raise ValueError("total_bend must be finite and positive")

    addresses = named_joint_addresses(fixture.model, fixture.manifest)
    angle = total_bend / len(addresses)
    axis_index = 0 if bend_axis == "x" else 1
    axis = np.zeros(3, dtype=float)
    axis[axis_index] = 1.0
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
    for _, qpos_address, dof_address in addresses:
        fixture.data.qpos[qpos_address:qpos_address + 4] = quaternion
        fixture.data.qvel[dof_address:dof_address + 3] = 0.0

    mujoco.mj_normalizeQuat(fixture.model, fixture.data.qpos)
    mujoco.mj_forward(fixture.model, fixture.data)
    return addresses
