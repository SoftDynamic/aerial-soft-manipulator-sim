"""MJCF construction for drone aviary models."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from multi_drone_mujoco.models.components.passive_arm import (
    PassiveArmSpec,
    build_passive_arm_xml,
)
from multi_drone_mujoco.models.components.scaled_stablesoft_arm import (
    ArmMjcfManifest,
    ScaledStableSoftArmSpec,
    ScaledStableSoftVisualSpec,
    build_scaled_stablesoft_arm_mjcf,
)
from multi_drone_mujoco.models.components.tendon_drive import (
    IdealTendonTensionSpec,
)
from multi_drone_mujoco.models.parameters import (
    CF2X_3G_PAYLOAD_PARAMS,
    ResolvedUavParameters,
)
from multi_drone_mujoco.utils.enums import DroneModel


# CF2 mesh files (.obj) are vendored under multi_drone_mujoco/assets/cf2.
# They were copied from google-deepmind/mujoco_menagerie (bitcraze_crazyflie_2)
# so users no longer need to clone the whole menagerie repo. To use a different
# checkout, set the MJ_DRONES_CF2_ASSETS env var to its assets dir.
ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"
CF2_MESH_DIR = Path(os.environ.get("MJ_DRONES_CF2_ASSETS",
                                   str(ASSETS_PATH / "cf2")))


@dataclass(frozen=True)
class AviaryMjcfBuild:
    """XML and manifests emitted by one production MJCF build."""

    xml: str
    scaled_stablesoft_arm_manifest: Optional[ArmMjcfManifest]
    rotor_actuator_names: tuple[tuple[str, ...], ...]


def build_aviary_mjcf(
    num_drones: int,
    drone_model: DroneModel,
    init_xyzs: np.ndarray,
    init_rpys: np.ndarray,
    obstacles: bool = False,
    vision: bool = False,
    timestep: float = 1 / 240,
    integrator: str = "RK4",
    enable_3g_payload: bool = False,
    passive_arm: Optional[PassiveArmSpec] = None,
    scaled_stablesoft_arm: Optional[ScaledStableSoftArmSpec] = None,
    scaled_stablesoft_visual: Optional[ScaledStableSoftVisualSpec] = None,
    scaled_stablesoft_tendon_drive: Optional[
        IdealTendonTensionSpec
    ] = None,
    uav_parameters: Optional[ResolvedUavParameters] = None,
) -> AviaryMjcfBuild:
    """Generate XML and manifests for an aviary in one build pass."""
    if not isinstance(integrator, str):
        raise ValueError("integrator must be a string")
    if not integrator:
        raise ValueError("integrator must not be empty")
    if integrator not in ("RK4", "implicitfast"):
        raise ValueError(
            "integrator must be one of: RK4, implicitfast"
        )
    if passive_arm is not None and scaled_stablesoft_arm is not None:
        raise ValueError(
            "The scaled StableSoft arm cannot be combined with the passive arm"
        )
    if (
        scaled_stablesoft_visual is not None
        and scaled_stablesoft_arm is None
    ):
        raise ValueError(
            "A scaled StableSoft visual requires a scaled StableSoft arm"
        )
    if (
        scaled_stablesoft_tendon_drive is not None
        and scaled_stablesoft_arm is None
    ):
        raise ValueError(
            "A scaled StableSoft tendon drive requires a scaled "
            "StableSoft arm"
        )
    if scaled_stablesoft_tendon_drive is not None and not isinstance(
        scaled_stablesoft_tendon_drive, IdealTendonTensionSpec
    ):
        raise TypeError("scaled_stablesoft_tendon_drive must be an IdealTendonTensionSpec")
    if enable_3g_payload:
        if drone_model != DroneModel.CF2X:
            raise ValueError("The 3 g rigid payload is only supported for DroneModel.CF2X")
        if num_drones != 1:
            raise ValueError("The 3 g rigid payload benchmark requires num_drones == 1")
    if passive_arm is not None:
        if drone_model != DroneModel.CF2X:
            raise ValueError("The passive arm is only supported for DroneModel.CF2X")
        if num_drones != 1:
            raise ValueError("The passive arm requires num_drones == 1")
        if enable_3g_payload:
            raise ValueError("The passive arm cannot be combined with the 3 g rigid payload")
    if scaled_stablesoft_arm is not None:
        if drone_model != DroneModel.CF2X:
            raise ValueError(
                "The scaled StableSoft arm is only supported for DroneModel.CF2X"
            )
        if num_drones != 1:
            raise ValueError(
                "The scaled StableSoft arm requires num_drones == 1"
            )
        if enable_3g_payload:
            raise ValueError(
                "The scaled StableSoft arm cannot be combined with the 3 g rigid payload"
            )

    # Validate the model combination before interpreting the initial pose arrays.
    init_xyzs = np.asarray(init_xyzs).reshape(num_drones, 3)
    init_rpys = np.asarray(init_rpys).reshape(num_drones, 3)
    meshdir = str(CF2_MESH_DIR)
    params = uav_parameters or ResolvedUavParameters.resolve(drone_model)

    # Visual and collision meshes
    visual_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_{i}.obj" name="cf2_vis_{i}"/>'
        for i in range(7)
    )
    collision_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_collision_{i}.obj" name="cf2_col_{i}"/>'
        for i in range(32)
    )

    # Drone bodies with 4 propeller sites for force application
    mass = params["mass"]
    ixx, iyy, izz = params["ixx"], params["iyy"], params["izz"]
    L = params["arm_length"]

    drone_bodies = ""
    actuators = ""
    sensors = ""
    tendons = ""
    arm_assets = ""
    scaled_stablesoft_arm_manifest = None
    rotor_actuator_names = []
    max_rotor_thrust = params["thrust2weight_ratio"] * mass * 9.81 / 4
    reaction_ratio = params["km"] / params["kf"]
    reaction_signs = (-1, 1, -1, 1)
    if drone_model == DroneModel.RACE:
        reaction_signs = tuple(-sign for sign in reaction_signs)

    for d in range(num_drones):
        x, y, z = init_xyzs[d]
        # Convert RPY to quaternion for initial orientation
        r, p_angle, yaw = init_rpys[d]
        # Simple RPY to quat (small angles)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p_angle / 2), np.sin(p_angle / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        prefix = f"drone{d}"

        # Propeller positions (X-configuration for CF2X)
        if drone_model == DroneModel.CF2X:
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),    # front-left
                (-L / np.sqrt(2), L / np.sqrt(2), 0),   # front-right (corrected)
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),  # rear-right
                (L / np.sqrt(2), -L / np.sqrt(2), 0),   # rear-left
            ]
        elif drone_model == DroneModel.CF2P:
            prop_offsets = [
                (L, 0, 0),     # front
                (0, L, 0),     # left
                (-L, 0, 0),    # rear
                (0, -L, 0),    # right
            ]
        else:  # RACE (same as X)
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),
                (L / np.sqrt(2), -L / np.sqrt(2), 0),
            ]

        prop_sites = ""
        for pi, (px, py, pz) in enumerate(prop_offsets):
            prop_sites += f'      <site name="{prefix}_prop{pi}" pos="{px} {py} {pz}" group="5"/>\n'

        rotor_names = tuple(f"r{4 * d + pi}_thrust_N" for pi in range(4))
        rotor_actuator_names.append(rotor_names)
        for pi, actuator_name in enumerate(rotor_names):
            actuators += (
                f'    <general name="{actuator_name}" site="{prefix}_prop{pi}" '
                f'gear="0 0 1 0 0 {reaction_signs[pi] * reaction_ratio:.17g}" '
                f'ctrllimited="true" ctrlrange="0 {max_rotor_thrust:.17g}"/>\n'
            )

        drone_bodies += f"""
    <body name="{prefix}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{prefix}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{prefix}_collision" type="cylinder" pos="0 0 {params['collision_z_offset']}" size="{params['collision_r']} {params['collision_h'] / 2}" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom mesh="cf2_vis_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_vis_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_vis_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_vis_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_vis_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_vis_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_vis_6" material="white" class="visual"/>
      <site name="{prefix}_center" pos="0 0 0" group="5"/>
{prop_sites}"""

        if enable_3g_payload:
            payload_mass = CF2X_3G_PAYLOAD_PARAMS["mass"]
            payload_x, payload_y, payload_z = CF2X_3G_PAYLOAD_PARAMS["position"]
            payload_radius = CF2X_3G_PAYLOAD_PARAMS["radius"]
            payload_half_height = CF2X_3G_PAYLOAD_PARAMS["half_height"]
            payload_full_height = 2 * payload_half_height
            payload_ixx = payload_mass * (
                3 * payload_radius ** 2 + payload_full_height ** 2
            ) / 12
            payload_izz = 0.5 * payload_mass * payload_radius ** 2
            drone_bodies += f"""      <body name="{prefix}_payload" pos="{payload_x} {payload_y} {payload_z}">
        <inertial pos="0 0 0" mass="{payload_mass}" diaginertia="{payload_ixx} {payload_ixx} {payload_izz}"/>
        <geom name="{prefix}_payload_geom" type="cylinder" size="{payload_radius} {payload_half_height}" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/>
      </body>
"""

        if passive_arm is not None:
            mount_z = params["collision_z_offset"] - params["collision_h"] / 2
            drone_bodies += build_passive_arm_xml(prefix, passive_arm, mount_z)

        if scaled_stablesoft_arm is not None:
            mount_z = params["collision_z_offset"] - params["collision_h"] / 2
            arm_fragments = build_scaled_stablesoft_arm_mjcf(
                prefix,
                scaled_stablesoft_arm,
                mount_z,
                tendon_drive=scaled_stablesoft_tendon_drive,
                visual=scaled_stablesoft_visual,
            )
            scaled_stablesoft_arm_manifest = arm_fragments.manifest
            if arm_fragments.asset_xml:
                arm_assets += "\n" + arm_fragments.asset_xml
            drone_bodies += arm_fragments.worldbody_xml
            tendons += arm_fragments.tendon_xml
            actuators += arm_fragments.actuator_xml
            sensors += "\n" + arm_fragments.sensor_xml

        # Add camera for vision
        if vision:
            drone_bodies += f'      <camera name="{prefix}_cam" pos="0.02 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>\n'

        drone_bodies += "    </body>\n"

        # Sensors
        sensors += f"""
    <gyro name="{prefix}_gyro" site="{prefix}_center"/>
    <accelerometer name="{prefix}_acc" site="{prefix}_center"/>
    <framequat name="{prefix}_quat" objtype="site" objname="{prefix}_center"/>
    <framepos name="{prefix}_pos" objtype="site" objname="{prefix}_center"/>
    <framelinvel name="{prefix}_vel" objtype="site" objname="{prefix}_center"/>
    <frameangvel name="{prefix}_angvel" objtype="site" objname="{prefix}_center"/>"""

    # Obstacles
    obstacle_bodies = ""
    if obstacles:
        obstacle_bodies = """
    <body name="obstacle_box" pos="0.5 0.5 0.3">
      <geom type="box" size="0.1 0.1 0.3" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="obstacle_sphere" pos="-0.5 0.5 0.5">
      <geom type="sphere" size="0.1" rgba="0.2 0.8 0.2 1"/>
    </body>
    <body name="obstacle_cylinder" pos="0 -0.5 0.4">
      <geom type="cylinder" size="0.05 0.3" rgba="0.2 0.2 0.8 1"/>
    </body>"""

    tendon_section = ""
    if tendons:
        tendon_section = f"""
  <tendon>
{tendons}
  </tendon>
"""

    actuator_section = ""
    if actuators:
        actuator_section = f"""
  <actuator>
{actuators}
  </actuator>
"""

    xml = f"""<mujoco model="aviary_{num_drones}x_{drone_model.value}">
  <option integrator="{integrator}" density="1.225" viscosity="1.8e-5" timestep="{timestep}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{visual_meshes}
{collision_meshes}{arm_assets}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="10 10 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_bodies}{obstacle_bodies}
  </worldbody>
{tendon_section}{actuator_section}

  <sensor>
{sensors}
  </sensor>
</mujoco>
"""
    return AviaryMjcfBuild(
        xml=xml,
        scaled_stablesoft_arm_manifest=scaled_stablesoft_arm_manifest,
        rotor_actuator_names=tuple(rotor_actuator_names),
    )
