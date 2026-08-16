"""Scaled StableSoft passive arm topology and tendon routing."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple


_STABLESOFT_REFERENCE_LENGTH = 9 * 0.096
_STABLESOFT_TENDON_RANGE_UPPER = 2.0
_CORNER_SIGNS = (
    ("xp_yp", 1.0, 1.0),
    ("xp_yn", 1.0, -1.0),
    ("xn_yp", -1.0, 1.0),
    ("xn_yn", -1.0, -1.0),
)


def solid_cylinder_diagonal_inertia(
    mass: float,
    radius: float,
    length: float,
) -> Tuple[float, float, float]:
    """Return principal inertia for a solid cylinder aligned with local Z."""
    transverse = mass * (3 * radius ** 2 + length ** 2) / 12
    axial = 0.5 * mass * radius ** 2
    return transverse, transverse, axial


def _normalized_finite_float(name: str, value: float) -> float:
    """Return a finite float while rejecting booleans and non-real values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"Scaled StableSoft arm {name} must be a real number, not bool"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Scaled StableSoft arm {name} must be a finite number"
        ) from exc
    if not math.isfinite(normalized):
        raise ValueError(
            f"Scaled StableSoft arm {name} must be finite"
        )
    return normalized


@dataclass(frozen=True)
class ScaledStableSoftArmSpec:
    """Core parameters for the nine-segment scaled StableSoft surrogate."""

    num_segments: int = 9
    total_length: float = 0.08
    total_mass: float = 0.003
    body_radius: float = 0.003
    tendon_range: Optional[Tuple[float, float]] = None
    tendon_frictionloss: float = 0.0

    def __post_init__(self):
        if isinstance(self.num_segments, bool) or not isinstance(
            self.num_segments, int
        ):
            raise ValueError(
                "Scaled StableSoft arm num_segments must be an integer"
            )
        if self.num_segments != 9:
            raise ValueError(
                "Scaled StableSoft arm num_segments must equal 9"
            )

        for field_name in (
            "total_length",
            "total_mass",
            "body_radius",
            "tendon_frictionloss",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_finite_float(field_name, getattr(self, field_name)),
            )
        if self.total_length <= 0:
            raise ValueError(
                "Scaled StableSoft arm total_length must be positive"
            )
        if self.total_mass <= 0:
            raise ValueError(
                "Scaled StableSoft arm total_mass must be positive"
            )
        if self.body_radius <= 0:
            raise ValueError(
                "Scaled StableSoft arm body_radius must be positive"
            )
        if self.tendon_frictionloss < 0:
            raise ValueError(
                "Scaled StableSoft arm tendon_frictionloss must be non-negative"
            )

        if self.tendon_range is not None:
            if not isinstance(self.tendon_range, (tuple, list)) or len(
                self.tendon_range
            ) != 2:
                raise ValueError(
                    "Scaled StableSoft arm tendon_range must contain two values"
                )
            lower = _normalized_finite_float(
                "tendon_range lower endpoint", self.tendon_range[0]
            )
            upper = _normalized_finite_float(
                "tendon_range upper endpoint", self.tendon_range[1]
            )
            object.__setattr__(self, "tendon_range", (lower, upper))
            if lower < 0 or lower >= upper:
                raise ValueError(
                    "Scaled StableSoft arm tendon_range must satisfy "
                    "0 <= lower < upper"
                )

        self._validate_derived_quantities()

    def _validate_derived_quantities(self):
        """Reject numerically or physically invalid derived model values."""
        try:
            derived_positive = (
                ("length_scale", self.length_scale),
                ("segment_pitch", self.segment_pitch),
                ("segment_mass", self.segment_mass),
                ("tendon_site_half_width", self.tendon_site_half_width),
                ("proximal_site_offset", self.proximal_site_offset),
                ("distal_site_offset", self.distal_site_offset),
                ("initial_tendon_length", self.initial_tendon_length),
                ("tendon_width", self.tendon_width),
                ("routing_site_size", self.routing_site_size),
            )
            tendon_lower, tendon_upper = self.effective_tendon_range
            inertia = solid_cylinder_diagonal_inertia(
                self.segment_mass,
                self.body_radius,
                self.segment_pitch,
            )
        except OverflowError as exc:
            raise ValueError(
                "Scaled StableSoft arm derived quantities overflowed "
                "during validation"
            ) from exc

        for name, value in derived_positive:
            if not math.isfinite(value):
                raise ValueError(
                    f"Scaled StableSoft arm derived {name} must be finite"
                )
            if value <= 0:
                raise ValueError(
                    f"Scaled StableSoft arm derived {name} must be positive"
                )

        if not math.isfinite(tendon_lower) or not math.isfinite(tendon_upper):
            raise ValueError(
                "Scaled StableSoft arm effective tendon_range must be finite"
            )
        if tendon_lower < 0 or tendon_lower >= tendon_upper:
            raise ValueError(
                "Scaled StableSoft arm tendon_range must satisfy "
                "0 <= lower < upper"
            )
        if not tendon_lower <= self.initial_tendon_length <= tendon_upper:
            raise ValueError(
                "Scaled StableSoft arm tendon_range must contain the "
                "initial tendon length"
            )

        if not all(math.isfinite(value) for value in inertia):
            raise ValueError(
                "Scaled StableSoft arm solid-cylinder inertia must be finite"
            )
        if not all(value > 0 for value in inertia):
            raise ValueError(
                "Scaled StableSoft arm solid-cylinder inertia must be positive"
            )
        try:
            triangle_is_strict = all(
                inertia[axis]
                < math.fsum(
                    inertia[other_axis]
                    for other_axis in range(3)
                    if other_axis != axis
                )
                for axis in range(3)
            )
        except OverflowError as exc:
            raise ValueError(
                "Scaled StableSoft arm inertia triangle validation overflowed"
            ) from exc
        if not triangle_is_strict:
            raise ValueError(
                "Scaled StableSoft arm solid-cylinder inertia must satisfy "
                "strict triangle inequalities"
            )

    @property
    def num_ball_joints(self) -> int:
        return self.num_segments - 1

    @property
    def length_scale(self) -> float:
        return self.total_length / _STABLESOFT_REFERENCE_LENGTH

    @property
    def segment_pitch(self) -> float:
        return self.total_length / self.num_segments

    @property
    def segment_mass(self) -> float:
        return self.total_mass / self.num_segments

    @property
    def tendon_site_half_width(self) -> float:
        return 0.035 * self.length_scale

    @property
    def proximal_site_offset(self) -> float:
        return 0.008 * self.length_scale

    @property
    def distal_site_offset(self) -> float:
        return 0.088 * self.length_scale

    @property
    def initial_tendon_length(self) -> float:
        within_segments = self.num_segments * (
            self.distal_site_offset - self.proximal_site_offset
        )
        between_segments = self.num_ball_joints * (
            self.segment_pitch
            + self.proximal_site_offset
            - self.distal_site_offset
        )
        return within_segments + between_segments

    @property
    def effective_tendon_range(self) -> Tuple[float, float]:
        if self.tendon_range is None:
            # This is the legacy model's 2.0 m range scaled geometrically. It
            # is deliberately not a spool travel or hardware length limit.
            return (
                0.0,
                _STABLESOFT_TENDON_RANGE_UPPER * self.length_scale,
            )
        return float(self.tendon_range[0]), float(self.tendon_range[1])

    @property
    def tendon_width(self) -> float:
        # Display-only radius: this does not represent a physical tendon
        # diameter and has no force or routing effect.
        return max(0.0004 * self.length_scale, 0.05 * self.body_radius)

    @property
    def routing_site_size(self) -> float:
        # Display-only marker size: this does not represent tendon hardware.
        return max(0.001 * self.length_scale, 0.08 * self.body_radius)

    @property
    def collision_enabled(self) -> bool:
        return False

    @property
    def tip_payload_mass(self) -> float:
        return 0.0


@dataclass(frozen=True)
class ArmMjcfManifest:
    """Stable names and paths emitted for one scaled StableSoft arm."""

    segment_body_names: Tuple[str, ...]
    ball_joint_names: Tuple[str, ...]
    routing_site_names: Tuple[str, ...]
    tendon_names: Tuple[str, ...]
    tendon_sensor_names: Tuple[str, ...]
    tendon_site_routes: Tuple[Tuple[str, ...], ...]
    arm_tip_site_name: str


@dataclass(frozen=True)
class ArmMjcfFragments:
    """MJCF children contributed to separate top-level model sections."""

    worldbody_xml: str
    tendon_xml: str
    actuator_xml: str
    sensor_xml: str
    manifest: ArmMjcfManifest


def _float_text(value: float) -> str:
    return format(value, ".17g")


def _segment_body_name(drone_prefix: str, segment_index: int) -> str:
    return (
        f"{drone_prefix}_scaled_stablesoft_arm_segment_{segment_index:02d}"
    )


def _routing_site_name(
    body_name: str,
    corner_name: str,
    position_name: str,
) -> str:
    return f"{body_name}_{corner_name}_{position_name}_site"


def build_scaled_stablesoft_arm_mjcf(
    drone_prefix: str,
    spec: ScaledStableSoftArmSpec,
    mount_z: float,
) -> ArmMjcfFragments:
    """Build the passive nine-segment arm and its four tendon routes."""
    segment_names = tuple(
        _segment_body_name(drone_prefix, segment_index)
        for segment_index in range(spec.num_segments)
    )
    joint_names = tuple(
        f"{segment_names[segment_index]}_ball_joint"
        for segment_index in range(1, spec.num_segments)
    )
    tip_site_name = f"{drone_prefix}_scaled_stablesoft_arm_tip_site"

    route_names = []
    tendon_names = []
    sensor_names = []
    all_routing_site_names = []
    for corner_name, _, _ in _CORNER_SIGNS:
        route = []
        for body_name in segment_names:
            proximal_name = _routing_site_name(
                body_name, corner_name, "proximal"
            )
            distal_name = _routing_site_name(
                body_name, corner_name, "distal"
            )
            route.extend((proximal_name, distal_name))
            all_routing_site_names.extend((proximal_name, distal_name))
        route_names.append(tuple(route))
        tendon_name = f"{drone_prefix}_scaled_stablesoft_arm_{corner_name}_tendon"
        tendon_names.append(tendon_name)
        sensor_names.append(f"{tendon_name}_length_sensor")

    transverse, _, axial = solid_cylinder_diagonal_inertia(
        spec.segment_mass,
        spec.body_radius,
        spec.segment_pitch,
    )
    half_pitch = spec.segment_pitch / 2
    child_xml = ""
    for segment_index in reversed(range(spec.num_segments)):
        body_name = segment_names[segment_index]
        body_z = mount_z if segment_index == 0 else -spec.segment_pitch
        joint_xml = ""
        if segment_index > 0:
            joint_xml = (
                f'        <joint name="{body_name}_ball_joint" type="ball" '
                'stiffness="0" damping="0" frictionloss="0" armature="0"/>'
                "\n"
            )

        site_lines = []
        for corner_name, x_sign, y_sign in _CORNER_SIGNS:
            x = x_sign * spec.tendon_site_half_width
            y = y_sign * spec.tendon_site_half_width
            for position_name, offset in (
                ("proximal", spec.proximal_site_offset),
                ("distal", spec.distal_site_offset),
            ):
                site_name = _routing_site_name(
                    body_name, corner_name, position_name
                )
                site_lines.append(
                    f'        <site name="{site_name}" pos="{_float_text(x)} '
                    f'{_float_text(y)} {_float_text(-offset)}" '
                    f'size="{_float_text(spec.routing_site_size)}" group="5"/>'
                )
        sites_xml = "\n".join(site_lines) + "\n"

        tip_xml = ""
        if segment_index == spec.num_segments - 1:
            tip_xml = (
                f'        <site name="{tip_site_name}" pos="0 0 '
                f'{_float_text(-spec.segment_pitch)}" group="5"/>\n'
            )

        child_xml = f'''      <body name="{body_name}" pos="0 0 {_float_text(body_z)}">
{joint_xml}        <inertial pos="0 0 {_float_text(-half_pitch)}" mass="{_float_text(spec.segment_mass)}" diaginertia="{_float_text(transverse)} {_float_text(transverse)} {_float_text(axial)}"/>
        <geom name="{body_name}_geom" type="cylinder" pos="0 0 {_float_text(-half_pitch)}" size="{_float_text(spec.body_radius)} {_float_text(half_pitch)}" rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0"/>
{sites_xml}{tip_xml}{child_xml}      </body>
'''

    tendon_lower, tendon_upper = spec.effective_tendon_range
    tendon_lines = []
    sensor_lines = []
    for tendon_name, sensor_name, route in zip(
        tendon_names, sensor_names, route_names
    ):
        route_xml = "\n".join(
            f'      <site site="{site_name}"/>' for site_name in route
        )
        tendon_lines.append(
            f'    <spatial name="{tendon_name}" '
            f'width="{_float_text(spec.tendon_width)}" '
            f'frictionloss="{_float_text(spec.tendon_frictionloss)}" '
            f'limited="true" range="{_float_text(tendon_lower)} '
            f'{_float_text(tendon_upper)}" rgba="0.95 0.3 0.3 1">\n'
            f"{route_xml}\n"
            "    </spatial>"
        )
        sensor_lines.append(
            f'    <tendonpos name="{sensor_name}" tendon="{tendon_name}"/>'
        )

    manifest = ArmMjcfManifest(
        segment_body_names=segment_names,
        ball_joint_names=joint_names,
        routing_site_names=tuple(all_routing_site_names),
        tendon_names=tuple(tendon_names),
        tendon_sensor_names=tuple(sensor_names),
        tendon_site_routes=tuple(route_names),
        arm_tip_site_name=tip_site_name,
    )
    return ArmMjcfFragments(
        worldbody_xml=child_xml,
        tendon_xml="\n".join(tendon_lines),
        actuator_xml="",
        sensor_xml="\n".join(sensor_lines),
        manifest=manifest,
    )
