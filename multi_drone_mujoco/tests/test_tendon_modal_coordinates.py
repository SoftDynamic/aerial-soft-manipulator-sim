"""Gate C2-C2-A geometric tendon/modal coordinate contracts."""

import itertools

import numpy as np
import pytest

from multi_drone_mujoco.control.tendon_modal_coordinates import (
    MODE_NAMES,
    TendonModalCoordinates,
)


NAMES = tuple(f"arm_{corner}_tendon" for corner in ("xp_yp", "xp_yn", "xn_yp", "xn_yn"))


def test_documented_transform_scaling_signs_and_units():
    transform = TendonModalCoordinates.from_tendon_names(NAMES)
    expected_g = np.array(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=float,
    )
    assert MODE_NAMES == ("common", "bend_x", "bend_y", "twist_candidate")
    np.testing.assert_array_equal(transform.g, expected_g)
    np.testing.assert_allclose(
        transform.bending_to_tendon((2e-4, -1e-4)),
        [1e-4, 3e-4, -3e-4, -1e-4],
        rtol=0.0,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        transform.tendon_to_bending([1e-4, 3e-4, -3e-4, -1e-4]),
        [2e-4, -1e-4],
        rtol=0.0,
        atol=1e-18,
    )


def test_transform_round_trip_and_projection_identities():
    transform = TendonModalCoordinates.from_tendon_names(NAMES)
    modal = np.array([1e-5, 2e-4, -3e-4, 4e-5])
    np.testing.assert_allclose(
        transform.tendon_to_modal(transform.modal_to_tendon(modal)),
        modal,
        rtol=0.0,
        atol=1e-18,
    )
    p = transform.bending_projection
    np.testing.assert_allclose(p @ p, p, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(p.T, p)
    np.testing.assert_allclose(p @ transform.g[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(p @ transform.g[3], 0.0, atol=1e-15)


def test_single_tendon_reference_is_explicitly_incompatible():
    transform = TendonModalCoordinates.from_tendon_names(NAMES)
    result = transform.project_to_bending([100e-6, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.projected, [50e-6, 0.0, 0.0, -50e-6], atol=1e-20)
    np.testing.assert_allclose(result.residual, [50e-6, 0.0, 0.0, 50e-6], atol=1e-20)
    np.testing.assert_allclose(result.bending_coordinates, [25e-6, 25e-6], atol=1e-20)
    assert result.was_projected
    assert result.relative_residual == pytest.approx(2 ** -0.5)


def test_compatible_input_remains_exact_and_arrays_are_read_only():
    transform = TendonModalCoordinates.from_tendon_names(NAMES)
    compatible = transform.bending_to_tendon((0.4, -0.2))
    result = transform.project_to_bending(compatible)
    np.testing.assert_allclose(result.projected, compatible, atol=1e-15)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-15)
    assert not result.was_projected
    assert all(not value.flags.writeable for value in (transform.g, transform.bending_g, transform.bending_projection, result.projected))


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(4))))
def test_manifest_permutation_preserves_physical_corner_transform(permutation):
    reference = TendonModalCoordinates.from_tendon_names(NAMES)
    names = tuple(NAMES[index] for index in permutation)
    candidate = TendonModalCoordinates.from_tendon_names(names)
    physical = np.array([0.1, 0.2, -0.3, -0.4])
    permuted = physical[list(permutation)]
    np.testing.assert_allclose(
        candidate.tendon_to_modal(permuted),
        reference.tendon_to_modal(physical),
        rtol=0.0,
        atol=1e-16,
    )
    restored = candidate.modal_to_tendon(reference.tendon_to_modal(physical))
    np.testing.assert_allclose(restored, physical[list(permutation)], rtol=0.0, atol=1e-16)


@pytest.mark.parametrize("values", ([1, 2, 3], [1, 2, 3, 4, 5], [1, 2, np.nan, 4], [1, 2, True, 4]))
def test_invalid_vectors_are_rejected(values):
    transform = TendonModalCoordinates.from_tendon_names(NAMES)
    with pytest.raises(ValueError):
        transform.project_to_bending(values)


def test_invalid_manifest_names_are_rejected():
    with pytest.raises(ValueError):
        TendonModalCoordinates.from_tendon_names(NAMES[:3])
    with pytest.raises(ValueError):
        TendonModalCoordinates.from_tendon_names((NAMES[0],) * 4)
    with pytest.raises(ValueError):
        TendonModalCoordinates.from_tendon_names((*NAMES[:3], "arm_missing_tendon"))
