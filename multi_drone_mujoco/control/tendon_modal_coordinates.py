"""Geometric modal coordinates for the four StableSoft tendons.

The manifest corner order is resolved by name; no numeric actuator or tendon
IDs participate in the transform.  With tendon contraction ``s`` in metres,

``q = (1/4) G s`` and ``s = G.T q``.

The two public bending coordinates ``q_x`` and ``q_y`` are tendon modal
contractions in metres.  Their derivatives and accelerations therefore have
units m/s and m/s^2.  They are not tip/UAV Cartesian coordinates, curvature,
or motor/spool coordinates.  Positive q_x/q_y bend the tip toward +X/+Y.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import numpy as np


MODE_NAMES = ("common", "bend_x", "bend_y", "twist_candidate")
_EXPECTED_CORNERS = frozenset(("xp_yp", "xp_yn", "xn_yp", "xn_yn"))


def _readonly(values):
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _finite_vector(name, values, size):
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of {size} real values") from exc
    if len(raw) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    normalized = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a real number")
        item = float(value)
        if not math.isfinite(item):
            raise ValueError(f"{name}[{index}] must be finite")
        normalized.append(item)
    return np.asarray(normalized, dtype=float)


def _corner_from_tendon_name(name):
    if not isinstance(name, str) or not name:
        raise ValueError("tendon names must be non-empty strings")
    matches = [corner for corner in _EXPECTED_CORNERS if name.endswith(f"_{corner}_tendon")]
    if len(matches) != 1:
        raise ValueError(f"tendon name must identify exactly one StableSoft corner: {name!r}")
    return matches[0]


@dataclass(frozen=True)
class BendingProjection:
    requested: np.ndarray
    projected: np.ndarray
    residual: np.ndarray
    bending_coordinates: np.ndarray
    residual_norm: float
    relative_residual: float
    was_projected: bool

    def __post_init__(self):
        for name in ("requested", "projected", "residual", "bending_coordinates"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class TendonModalCoordinates:
    """Manifest-ordered four-tendon modal transform."""

    tendon_names: Tuple[str, ...]
    corners: Tuple[str, ...]
    g: np.ndarray
    bending_g: np.ndarray
    bending_projection: np.ndarray

    def __post_init__(self):
        for name in ("g", "bending_g", "bending_projection"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @classmethod
    def from_tendon_names(cls, tendon_names):
        names = tuple(tendon_names)
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError("exactly four unique tendon names are required")
        corners = tuple(_corner_from_tendon_name(name) for name in names)
        if frozenset(corners) != _EXPECTED_CORNERS:
            raise ValueError("tendon names must contain all four unique corners")
        x = np.asarray([1.0 if corner.startswith("xp_") else -1.0 for corner in corners])
        y = np.asarray([1.0 if corner.endswith("_yp") else -1.0 for corner in corners])
        g = np.vstack((np.ones(4), x, y, x * y))
        bending_g = g[1:3]
        projection = 0.25 * bending_g.T @ bending_g
        return cls(names, corners, g, bending_g, projection)

    def tendon_to_modal(self, tendon_values):
        values = _finite_vector("tendon_values", tendon_values, 4)
        return _readonly(0.25 * self.g @ values)

    def modal_to_tendon(self, modal_values):
        values = _finite_vector("modal_values", modal_values, 4)
        return _readonly(self.g.T @ values)

    def tendon_to_bending(self, tendon_values):
        values = _finite_vector("tendon_values", tendon_values, 4)
        return _readonly(0.25 * self.bending_g @ values)

    def bending_to_tendon(self, bending_values):
        values = _finite_vector("bending_values", bending_values, 2)
        return _readonly(self.bending_g.T @ values)

    def project_to_bending(self, tendon_values):
        requested = _finite_vector("tendon_values", tendon_values, 4)
        bending = 0.25 * self.bending_g @ requested
        projected = self.bending_g.T @ bending
        residual = requested - projected
        residual_norm = float(np.linalg.norm(residual))
        requested_norm = float(np.linalg.norm(requested))
        scale = max(1.0, requested_norm)
        tolerance = 64 * np.finfo(float).eps * scale
        return BendingProjection(
            requested=requested,
            projected=projected,
            residual=residual,
            bending_coordinates=bending,
            residual_norm=residual_norm,
            relative_residual=(residual_norm / requested_norm if requested_norm else 0.0),
            was_projected=residual_norm > tolerance,
        )
