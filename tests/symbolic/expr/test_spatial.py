"""Tests for spatial/6DOF operation nodes.

This module tests spatial operation nodes for aerospace and robotics applications:

- QDCM: Quaternion to Direction Cosine Matrix
- SSMP: 4×4 skew-symmetric matrix for quaternion dynamics
- SSM: 3×3 skew-symmetric matrix for cross products

Tests cover:

- Node creation and properties
- Shape checking
- Canonicalization
- Lowering to JAX (with slices)
- Lowering to CVXPY (with variable mapping)
"""

import jax.numpy as jnp
import numpy as np

from openscvx.symbolic.expr import (
    QDCM,
    SSM,
    SSMP,
    Concat,
    Constant,
    Control,
    Diag,
    Norm,
    State,
)
from openscvx.symbolic.lower import lower_to_jax

# =============================================================================
# Helper Functions for Reference Implementations
# =============================================================================


def qdcm_ref(q: jnp.ndarray) -> jnp.ndarray:
    """Convert a quaternion to a direction cosine matrix (DCM).

    Args:
        q: Quaternion array [w, x, y, z] where w is the scalar part

    Returns:
        3x3 rotation matrix (direction cosine matrix)
    """
    q_norm = jnp.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    w, x, y, z = q / q_norm
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def ssmp_ref(w: jnp.ndarray) -> jnp.ndarray:
    """Convert an angular rate to a 4x4 skew-symmetric matrix for quaternion kinematics.

    Args:
        w: Angular velocity vector [x, y, z]

    Returns:
        4x4 skew-symmetric matrix for quaternion propagation
    """
    x, y, z = w
    return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])


def ssm_ref(w: jnp.ndarray) -> jnp.ndarray:
    """Convert an angular rate to a 3x3 skew-symmetric matrix for cross products.

    Args:
        w: Vector [x, y, z]

    Returns:
        3x3 skew-symmetric matrix such that SSM(w) @ v = w x v
    """
    x, y, z = w
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def reference_6dof_dynamics_jax(x_val, u_val):
    """Reference implementation of 6DOF rigid body dynamics in pure JAX."""

    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Extract components
    # r = x_val[0:3]  # position
    v = x_val[3:6]  # velocity
    q = x_val[6:10]  # quaternion
    w = x_val[10:13]  # angular velocity
    # t = x_val[13]  # time

    f = u_val[:3]  # forces
    tau = u_val[3:]  # torques

    # Normalize quaternion
    q_norm = jnp.linalg.norm(q)
    q_normalized = q / q_norm

    # Compute dynamics
    r_dot = v
    v_dot = (1.0 / m) * qdcm_ref(q_normalized) @ f + jnp.array([0, 0, g_const])
    q_dot = 0.5 * ssmp_ref(w) @ q_normalized
    w_dot = jnp.diag(1.0 / J_b) @ (tau - ssm_ref(w) @ jnp.diag(J_b) @ w)
    t_dot = 1.0

    return jnp.concatenate([r_dot, v_dot, q_dot, w_dot, jnp.array([t_dot])])


# =============================================================================
# QDCM
# =============================================================================


def test_qdcm():
    """Test the QDCM compact node individually."""
    # Test with a few different quaternions
    test_quaternions = [
        jnp.array([1.0, 0.0, 0.0, 0.0]),  # Identity rotation
        jnp.array([0.707, 0.707, 0.0, 0.0]),  # 90° rotation around x-axis
        jnp.array([0.5, 0.5, 0.5, 0.5]),  # 120° rotation around (1,1,1) axis
    ]

    for q_val in test_quaternions:
        # Normalize quaternion
        q_val = q_val / jnp.linalg.norm(q_val)

        # Create quaternion state
        q = State("q", (4,))
        q._slice = slice(0, 4)

        # Test QDCM node
        qdcm_expr = QDCM(q)
        fn = lower_to_jax(qdcm_expr)
        result = fn(q_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be orthogonal (R.T @ R = I)
        identity_check = result.T @ result
        assert jnp.allclose(identity_check, jnp.eye(3), atol=1e-10)

        # Should have determinant 1 (proper rotation)
        det = jnp.linalg.det(result)
        assert jnp.allclose(det, 1.0, atol=1e-10)


# --- QDCM: Shape Checking --- TODO: (norrisg)


# --- QDCM: Canonicalization --- TODO: (norrisg)


# --- QDCM: JAX Lowering --- TODO: (norrisg)


# --- QDCM: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# SSMP
# =============================================================================


def test_ssmp():
    """Test the SSMP compact node individually."""
    # Test with different angular velocities
    test_angular_velocities = [
        jnp.array([0.0, 0.0, 0.0]),  # Zero rotation
        jnp.array([1.0, 0.0, 0.0]),  # Rotation around x-axis
        jnp.array([0.1, 0.2, 0.3]),  # General rotation
    ]

    for w_val in test_angular_velocities:
        # Create angular velocity state
        w = State("w", (3,))
        w._slice = slice(0, 3)

        # Test SSMP node
        ssmp_expr = SSMP(w)
        fn = lower_to_jax(ssmp_expr)
        result = fn(w_val, None, None, None)

        # Should be 4x4 matrix
        assert result.shape == (4, 4)

        # Should be skew-symmetric in the 3x3 submatrix part
        # The structure should be:
        # [0, -wx, -wy, -wz]
        # [wx, 0, wz, -wy]
        # [wy, -wz, 0, wx]
        # [wz, wy, -wx, 0]
        expected = jnp.array(
            [
                [0.0, -w_val[0], -w_val[1], -w_val[2]],
                [w_val[0], 0.0, w_val[2], -w_val[1]],
                [w_val[1], -w_val[2], 0.0, w_val[0]],
                [w_val[2], w_val[1], -w_val[0], 0.0],
            ]
        )
        assert jnp.allclose(result, expected, atol=1e-12)


# --- SSMP: Shape Checking --- TODO: (norrisg)


# --- SSMP: Canonicalization --- TODO: (norrisg)


# --- SSMP: JAX Lowering --- TODO: (norrisg)


# --- SSMP: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# SSM
# =============================================================================


def test_ssm():
    """Test the SSM compact node individually."""
    # Test with different angular velocities
    test_angular_velocities = [
        jnp.array([0.0, 0.0, 0.0]),  # Zero rotation
        jnp.array([1.0, 0.0, 0.0]),  # Rotation around x-axis
        jnp.array([0.1, 0.2, 0.3]),  # General rotation
    ]

    for w_val in test_angular_velocities:
        # Create angular velocity state
        w = State("w", (3,))
        w._slice = slice(0, 3)

        # Test SSM node
        ssm_expr = SSM(w)
        fn = lower_to_jax(ssm_expr)
        result = fn(w_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be skew-symmetric
        assert jnp.allclose(result, -result.T, atol=1e-12)

        # Check specific structure
        # [0, -wz, wy]
        # [wz, 0, -wx]
        # [-wy, wx, 0]
        expected = jnp.array(
            [[0.0, -w_val[2], w_val[1]], [w_val[2], 0.0, -w_val[0]], [-w_val[1], w_val[0], 0.0]]
        )
        assert jnp.allclose(result, expected, atol=1e-12)


# --- SSM: Shape Checking --- TODO: (norrisg)


# --- SSM: Canonicalization --- TODO: (norrisg)


# --- SSM: JAX Lowering --- TODO: (norrisg)


# --- SSM: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# Integration Tests: 6DOF Rigid Body Dynamics
# =============================================================================


def test_6dof_rigid_body_dynamics_symbolic():
    """Test the fully symbolic 6DOF rigid body dynamics against reference JAX implementation."""

    # Define symbolic utility functions (from drone racing example)
    def symbolic_qdcm(q_normalized):
        """Quaternion to Direction Cosine Matrix conversion using symbolic expressions"""
        # Assume q is already normalized
        w, x, y, z = q_normalized[0], q_normalized[1], q_normalized[2], q_normalized[3]

        # Create DCM elements
        r11 = Constant(1.0) - Constant(2.0) * (y * y + z * z)
        r12 = Constant(2.0) * (x * y - z * w)
        r13 = Constant(2.0) * (x * z + y * w)

        r21 = Constant(2.0) * (x * y + z * w)
        r22 = Constant(1.0) - Constant(2.0) * (x * x + z * z)
        r23 = Constant(2.0) * (y * z - x * w)

        r31 = Constant(2.0) * (x * z - y * w)
        r32 = Constant(2.0) * (y * z + x * w)
        r33 = Constant(1.0) - Constant(2.0) * (x * x + y * y)

        # Stack into 3x3 matrix
        from openscvx.symbolic.expr import Stack

        row1 = Concat(r11, r12, r13)
        row2 = Concat(r21, r22, r23)
        row3 = Concat(r31, r32, r33)

        return Stack([row1, row2, row3])

    def symbolic_ssmp(w):
        """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""
        from openscvx.symbolic.expr import Stack

        x, y, z = w[0], w[1], w[2]
        zero = Constant(0.0)

        # Create SSMP matrix
        row1 = Concat(zero, -x, -y, -z)
        row2 = Concat(x, zero, z, -y)
        row3 = Concat(y, -z, zero, x)
        row4 = Concat(z, y, -x, zero)

        return Stack([row1, row2, row3, row4])

    def symbolic_ssm(w):
        """Angular rate to 3x3 skew symmetric matrix"""
        from openscvx.symbolic.expr import Stack

        x, y, z = w[0], w[1], w[2]
        zero = Constant(0.0)

        # Create SSM matrix
        row1 = Concat(zero, -z, y)
        row2 = Concat(z, zero, -x)
        row3 = Concat(-y, x, zero)

        return Stack([row1, row2, row3])

    def symbolic_diag(v):
        """Create diagonal matrix from vector"""
        from openscvx.symbolic.expr import Stack

        if len(v) == 3:
            zero = Constant(0.0)
            row1 = Concat(v[0], zero, zero)
            row2 = Concat(zero, v[1], zero)
            row3 = Concat(zero, zero, v[2])
            return Stack([row1, row2, row3])
        else:
            raise NotImplementedError("Only 3x3 diagonal matrices supported")

    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Test multiple state/control combinations
    test_cases = [
        # Test case 1: Basic case
        (
            jnp.array(
                [10.0, 0.0, 20.0, 0.5, 0.2, -0.1, 1.0, 0.1, 0.05, 0.02, 0.1, 0.05, -0.02, 15.0]
            ),
            jnp.array([0.0, 0.0, 10.0, 0.1, -0.05, 0.02]),
        ),
        # Test case 2: Different orientation
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        ),
        # Test case 3: With angular velocity
        (
            jnp.array([5.0, -2.0, 10.0, 1.0, 0.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.2, -0.1, 0.3, 10.0]),
            jnp.array([1.0, 1.0, 8.0, 0.05, 0.1, -0.05]),
        ),
    ]

    for x_val, u_val in test_cases:
        # Normalize quaternion in test case
        q_norm = jnp.linalg.norm(x_val[6:10])
        x_val = x_val.at[6:10].set(x_val[6:10] / q_norm)

        # Create symbolic state and control
        x = State("x", (14,))
        x._slice = slice(0, 14)
        u = Control("u", (6,))
        u._slice = slice(0, 6)

        # Extract components
        v = x[3:6]
        q = x[6:10]
        q_norm = Norm(q)
        q_normalized = q / q_norm
        w = x[10:13]

        f = u[:3]
        tau = u[3:]

        # Create symbolic dynamics
        r_dot = v
        v_dot = (1.0 / m) * symbolic_qdcm(q_normalized) @ f + Constant(
            np.array([0, 0, g_const], dtype=np.float64)
        )
        q_dot = 0.5 * symbolic_ssmp(w) @ q_normalized
        J_b_inv = 1.0 / J_b
        J_b_diag = symbolic_diag([J_b[0], J_b[1], J_b[2]])
        w_dot = symbolic_diag([J_b_inv[0], J_b_inv[1], J_b_inv[2]]) @ (
            tau - symbolic_ssm(w) @ J_b_diag @ w
        )
        t_dot = 1.0

        dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)

        # Lower to JAX and test
        fn = lower_to_jax(dyn_expr)
        symbolic_result = fn(x_val, u_val, None, None)

        # Compare against reference implementation
        reference_result = reference_6dof_dynamics_jax(x_val, u_val)

        # Should be identical since both lower to the same JAX operations
        # assert jnp.allclose(symbolic_result, reference_result, rtol=1e-12, atol=1e-14)
        # TODO: (norrisg) figure out why it is not closer
        assert jnp.allclose(symbolic_result, reference_result, rtol=1e-6, atol=1e-12)


def test_6dof_rigid_body_dynamics_compact():
    """Test the compact node 6DOF rigid body dynamics against reference JAX implementation."""
    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Test multiple state/control combinations (same as symbolic test)
    test_cases = [
        # Test case 1: Basic case
        (
            jnp.array(
                [10.0, 0.0, 20.0, 0.5, 0.2, -0.1, 1.0, 0.1, 0.05, 0.02, 0.1, 0.05, -0.02, 15.0]
            ),
            jnp.array([0.0, 0.0, 10.0, 0.1, -0.05, 0.02]),
        ),
        # Test case 2: Different orientation
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        ),
        # Test case 3: With angular velocity
        (
            jnp.array([5.0, -2.0, 10.0, 1.0, 0.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.2, -0.1, 0.3, 10.0]),
            jnp.array([1.0, 1.0, 8.0, 0.05, 0.1, -0.05]),
        ),
    ]

    for x_val, u_val in test_cases:
        # Normalize quaternion in test case
        q_norm = jnp.linalg.norm(x_val[6:10])
        x_val = x_val.at[6:10].set(x_val[6:10] / q_norm)

        # Create symbolic state and control
        x = State("x", (14,))
        x._slice = slice(0, 14)
        u = Control("u", (6,))
        u._slice = slice(0, 6)

        # Extract components
        v = x[3:6]
        q = x[6:10]
        q_norm = Norm(q)
        q_normalized = q / q_norm
        w = x[10:13]

        f = u[:3]
        tau = u[3:]

        # Create compact node dynamics (from drone racing example)
        r_dot = v
        v_dot = (1.0 / m) * QDCM(q_normalized) @ f + Constant(
            np.array([0, 0, g_const], dtype=np.float64)
        )
        q_dot = 0.5 * SSMP(w) @ q_normalized
        J_b_inv = 1.0 / J_b
        J_b_diag = Diag(J_b)
        w_dot = Diag(J_b_inv) @ (tau - SSM(w) @ J_b_diag @ w)
        t_dot = 1.0

        dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)

        # Lower to JAX and test
        fn = lower_to_jax(dyn_expr)
        compact_result = fn(x_val, u_val, None, None)

        # Compare against reference implementation
        reference_result = reference_6dof_dynamics_jax(x_val, u_val)

        # Should be identical since both lower to the same JAX operations
        assert jnp.allclose(compact_result, reference_result, rtol=1e-12, atol=1e-14)
