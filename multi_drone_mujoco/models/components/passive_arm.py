"""Single-link passive arm MJCF component."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PassiveArmSpec:
    """Physical parameters for a single rigid passive arm."""

    length: float
    radius: float
    mass: float
    initial_angle: float
    joint_damping: float

    def __post_init__(self):
        values = (
            self.length,
            self.radius,
            self.mass,
            self.initial_angle,
            self.joint_damping,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Passive arm parameters must be finite")
        if self.length <= 0:
            raise ValueError("Passive arm length must be positive")
        if self.radius <= 0:
            raise ValueError("Passive arm radius must be positive")
        if self.mass <= 0:
            raise ValueError("Passive arm mass must be positive")
        if self.joint_damping < 0:
            raise ValueError("Passive arm joint damping must be non-negative")


def build_passive_arm_xml(
    drone_prefix: str,
    spec: PassiveArmSpec,
    mount_z: float,
) -> str:
    """Build the MJCF body for a single-link passive arm."""
    half_length = spec.length / 2
    transverse_inertia = spec.mass * (
        3 * spec.radius ** 2 + spec.length ** 2
    ) / 12
    axial_inertia = 0.5 * spec.mass * spec.radius ** 2
    half_angle = spec.initial_angle / 2
    quat_w = math.cos(half_angle)
    quat_y = math.sin(half_angle)
    joint_ref_degrees = math.degrees(spec.initial_angle)

    return f"""      <body name="{drone_prefix}_passive_arm" pos="0 0 {mount_z}" quat="{quat_w} 0 {quat_y} 0">
        <joint name="{drone_prefix}_passive_arm_joint" type="hinge" axis="0 1 0" ref="{joint_ref_degrees}" damping="{spec.joint_damping}" stiffness="0" armature="0" frictionloss="0" limited="false"/>
        <inertial pos="0 0 {-half_length}" mass="{spec.mass}" diaginertia="{transverse_inertia} {transverse_inertia} {axial_inertia}"/>
        <geom name="{drone_prefix}_passive_arm_geom" type="cylinder" pos="0 0 {-half_length}" size="{spec.radius} {half_length}" rgba="0.2 0.4 0.9 1" contype="1" conaffinity="1"/>
      </body>
"""
