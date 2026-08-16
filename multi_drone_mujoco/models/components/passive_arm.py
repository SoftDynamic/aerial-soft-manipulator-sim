"""Multi-segment two-direction passive flexible arm MJCF component."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PassiveArmSpec:
    """Whole-arm parameters for a discretized passive flexible arm.

    Bending angles, stiffnesses, and damping coefficients describe the whole
    arm. The builder distributes them over ``num_segments`` identical rigid
    segments while preserving the equivalent uniform-bending properties.
    """

    total_length: float
    radius: float
    total_mass: float
    num_segments: int
    initial_bend_x: float
    initial_bend_y: float
    bend_stiffness_x: float
    bend_stiffness_y: float
    bend_damping_x: float
    bend_damping_y: float
    max_bend_x: float
    max_bend_y: float

    def __post_init__(self):
        if isinstance(self.num_segments, bool) or not isinstance(
            self.num_segments, int
        ):
            raise ValueError("Passive arm num_segments must be an integer")
        if self.num_segments < 1:
            raise ValueError("Passive arm num_segments must be at least 1")

        values = (
            self.total_length,
            self.radius,
            self.total_mass,
            self.initial_bend_x,
            self.initial_bend_y,
            self.bend_stiffness_x,
            self.bend_stiffness_y,
            self.bend_damping_x,
            self.bend_damping_y,
            self.max_bend_x,
            self.max_bend_y,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Passive arm parameters must be finite")
        if self.total_length <= 0:
            raise ValueError("Passive arm total_length must be positive")
        if self.radius <= 0:
            raise ValueError("Passive arm radius must be positive")
        if self.total_mass <= 0:
            raise ValueError("Passive arm total_mass must be positive")
        if self.bend_stiffness_x < 0 or self.bend_stiffness_y < 0:
            raise ValueError("Passive arm bend stiffness must be non-negative")
        if self.bend_damping_x < 0 or self.bend_damping_y < 0:
            raise ValueError("Passive arm bend damping must be non-negative")
        if self.max_bend_x <= 0 or self.max_bend_y <= 0:
            raise ValueError("Passive arm max bend must be positive")
        if abs(self.initial_bend_x) > self.max_bend_x:
            raise ValueError("Passive arm initial_bend_x must be within max_bend_x")
        if abs(self.initial_bend_y) > self.max_bend_y:
            raise ValueError("Passive arm initial_bend_y must be within max_bend_y")


def _multiply_quaternions(left, right):
    """Return the Hamilton product of two ``(w, x, y, z)`` quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def build_passive_arm_xml(
    drone_prefix: str,
    spec: PassiveArmSpec,
    mount_z: float,
) -> str:
    """Build a serial rigid-body approximation of a passive flexible arm."""
    segment_length = spec.total_length / spec.num_segments
    segment_mass = spec.total_mass / spec.num_segments
    half_length = segment_length / 2
    transverse_inertia = segment_mass * (
        3 * spec.radius ** 2 + segment_length ** 2
    ) / 12
    axial_inertia = 0.5 * segment_mass * spec.radius ** 2

    bend_x = spec.initial_bend_x / spec.num_segments
    bend_y = spec.initial_bend_y / spec.num_segments
    bend_x_degrees = math.degrees(bend_x)
    bend_y_degrees = math.degrees(bend_y)
    max_bend_x_degrees = math.degrees(
        spec.max_bend_x / spec.num_segments
    )
    max_bend_y_degrees = math.degrees(
        spec.max_bend_y / spec.num_segments
    )
    joint_stiffness_x = spec.num_segments * spec.bend_stiffness_x
    joint_stiffness_y = spec.num_segments * spec.bend_stiffness_y
    joint_damping_x = spec.num_segments * spec.bend_damping_x
    joint_damping_y = spec.num_segments * spec.bend_damping_y

    # Joint order is bend-X (axis -Y), then bend-Y (axis +X). The reference
    # body orientation uses the reverse product so setting both qpos values to
    # their zero spring references cancels the preloaded reference pose exactly.
    bend_x_quaternion = (
        math.cos(bend_x / 2),
        0.0,
        -math.sin(bend_x / 2),
        0.0,
    )
    bend_y_quaternion = (
        math.cos(bend_y / 2),
        math.sin(bend_y / 2),
        0.0,
        0.0,
    )
    body_quaternion = _multiply_quaternions(
        bend_y_quaternion,
        bend_x_quaternion,
    )
    quaternion_text = " ".join(str(value) for value in body_quaternion)

    child_xml = ""
    for segment_index in reversed(range(spec.num_segments)):
        body_name = (
            f"{drone_prefix}_passive_arm_segment_{segment_index:02d}"
        )
        body_z = mount_z if segment_index == 0 else -segment_length
        tip_site = ""
        if segment_index == spec.num_segments - 1:
            tip_site = (
                f'        <site name="{drone_prefix}_passive_arm_tip_site" '
                f'pos="0 0 {-segment_length}" group="5"/>\n'
            )

        child_xml = f'''      <body name="{body_name}" pos="0 0 {body_z}" quat="{quaternion_text}">
        <joint name="{body_name}_bend_x_joint" type="hinge" axis="0 -1 0" ref="{bend_x_degrees}" springref="0" damping="{joint_damping_x}" stiffness="{joint_stiffness_x}" armature="0" frictionloss="0" limited="true" range="{-max_bend_x_degrees} {max_bend_x_degrees}"/>
        <joint name="{body_name}_bend_y_joint" type="hinge" axis="1 0 0" ref="{bend_y_degrees}" springref="0" damping="{joint_damping_y}" stiffness="{joint_stiffness_y}" armature="0" frictionloss="0" limited="true" range="{-max_bend_y_degrees} {max_bend_y_degrees}"/>
        <inertial pos="0 0 {-half_length}" mass="{segment_mass}" diaginertia="{transverse_inertia} {transverse_inertia} {axial_inertia}"/>
        <geom name="{body_name}_geom" type="cylinder" pos="0 0 {-half_length}" size="{spec.radius} {half_length}" rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>
{tip_site}{child_xml}      </body>
'''

    return child_xml
