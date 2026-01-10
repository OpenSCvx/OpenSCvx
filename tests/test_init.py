"""Tests for trajectory initialization utilities (openscvx.init)."""

import numpy as np
import pytest

from openscvx.init import linspace, nlerp, slerp

# =============================================================================
# linspace
# =============================================================================


def test_linspace_two_keyframes():
    """Test basic linspace with start and end keyframes."""
    result = linspace(
        keyframes=[[0.0, 0.0], [10.0, 20.0]],
        nodes=[0, 10],
    )

    assert result.shape == (11, 2)
    np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(result[10], [10.0, 20.0])
    np.testing.assert_array_almost_equal(result[5], [5.0, 10.0])  # midpoint


def test_linspace_multiple_keyframes():
    """Test linspace with intermediate keyframes."""
    result = linspace(
        keyframes=[[0.0], [10.0], [0.0]],
        nodes=[0, 5, 10],
    )

    assert result.shape == (11, 1)
    np.testing.assert_array_almost_equal(result[0], [0.0])
    np.testing.assert_array_almost_equal(result[5], [10.0])
    np.testing.assert_array_almost_equal(result[10], [0.0])
    np.testing.assert_array_almost_equal(result[2], [4.0])  # 0 -> 10, at 2/5


def test_linspace_adjacent_nodes():
    """Test linspace with adjacent nodes (spacing of 1)."""
    result = linspace(
        keyframes=[[0.0], [10.0], [5.0]],
        nodes=[0, 1, 2],
    )

    assert result.shape == (3, 1)
    np.testing.assert_array_almost_equal(result[0], [0.0])
    np.testing.assert_array_almost_equal(result[1], [10.0])
    np.testing.assert_array_almost_equal(result[2], [5.0])


def test_linspace_validation():
    """Test linspace input validation."""
    with pytest.raises(ValueError, match="same length"):
        linspace(keyframes=[[0.0], [10.0], [20.0]], nodes=[0, 10])

    with pytest.raises(ValueError, match="strictly increasing"):
        linspace(keyframes=[[0.0], [10.0]], nodes=[5, 5])

    with pytest.raises(ValueError, match="strictly increasing"):
        linspace(keyframes=[[0.0], [10.0]], nodes=[5, 3])  # decreasing

    with pytest.raises(ValueError, match="At least 2 keyframes"):
        linspace(keyframes=[[0.0]], nodes=[0])


# =============================================================================
# nlerp
# =============================================================================


def test_nlerp_basic():
    """Test nlerp interpolation and normalization."""
    q_identity = [1.0, 0.0, 0.0, 0.0]
    q_90z = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]

    result = nlerp(keyframes=[q_identity, q_90z], nodes=[0, 10])

    assert result.shape == (11, 4)
    np.testing.assert_array_almost_equal(result[0], q_identity)
    np.testing.assert_array_almost_equal(result[10], q_90z, decimal=5)

    # All quaternions should be normalized
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(11))


def test_nlerp_multiple_keyframes():
    """Test nlerp with three keyframes."""
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    q2 = [0.0, 0.0, 0.0, 1.0]

    result = nlerp(keyframes=[q0, q1, q2], nodes=[0, 5, 10])

    np.testing.assert_array_almost_equal(result[0], q0)
    np.testing.assert_array_almost_equal(result[5], q1, decimal=5)
    np.testing.assert_array_almost_equal(result[10], q2)


def test_nlerp_validation():
    """Test nlerp input validation."""
    with pytest.raises(ValueError, match="shape.*expected.*4"):
        nlerp(keyframes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], nodes=[0, 10])


# =============================================================================
# slerp
# =============================================================================


def test_slerp_basic():
    """Test slerp interpolation."""
    q_identity = [1.0, 0.0, 0.0, 0.0]
    q_90z = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]

    result = slerp(keyframes=[q_identity, q_90z], nodes=[0, 10])

    assert result.shape == (11, 4)
    np.testing.assert_array_almost_equal(result[0], q_identity)
    np.testing.assert_array_almost_equal(result[10], q_90z, decimal=5)


def test_slerp_constant_angular_velocity():
    """Test that slerp produces constant angular velocity."""
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = [0.0, 0.0, 0.0, 1.0]  # 180 deg about z

    result = slerp(keyframes=[q0, q1], nodes=[0, 10])

    # Angular distance between consecutive quaternions should be constant
    angular_distances = []
    for i in range(10):
        dot = np.abs(np.dot(result[i], result[i + 1]))
        angle = 2 * np.arccos(np.clip(dot, -1.0, 1.0))
        angular_distances.append(angle)

    np.testing.assert_array_almost_equal(
        angular_distances, np.full(10, angular_distances[0]), decimal=5
    )


def test_slerp_vs_nlerp_similar_for_small_angles():
    """Test that slerp and nlerp are similar for small rotations."""
    q0 = [1.0, 0.0, 0.0, 0.0]
    angle = np.radians(5)
    q1 = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]

    result_slerp = slerp(keyframes=[q0, q1], nodes=[0, 10])
    result_nlerp = nlerp(keyframes=[q0, q1], nodes=[0, 10])

    np.testing.assert_array_almost_equal(result_slerp, result_nlerp, decimal=3)
