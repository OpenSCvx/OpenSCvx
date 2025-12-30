"""Tests for Lie algebra operation nodes.

This module tests Lie algebra operation nodes for rigid body dynamics:

- AdjointDual: Coadjoint operator ad* for Coriolis/centrifugal forces
- Adjoint: Lie bracket for twist-on-twist action

Tests cover:

- Node creation and properties
- Shape checking
- Canonicalization
- Lowering to JAX (with slices)
- Mathematical correctness against reference implementations
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Adjoint,
    AdjointDual,
    Constant,
    State,
)
from openscvx.symbolic.lower import lower_to_jax

# =============================================================================
# Helper Functions for Reference Implementations
# =============================================================================


def adjoint_dual_ref(twist: jnp.ndarray, momentum: jnp.ndarray) -> jnp.ndarray:
    """Reference implementation of coadjoint operator ad*.

    For se(3), given twist ξ = [v; ω] and momentum μ = [f; τ]:
        ad*_ξ(μ) = [ω × f + v × τ; ω × τ]

    Args:
        twist: 6D twist [v; ω] (linear velocity, angular velocity)
        momentum: 6D momentum [f; τ] (force, torque)

    Returns:
        6D coadjoint result
    """
    v = twist[:3]
    omega = twist[3:]
    f = momentum[:3]
    tau = momentum[3:]

    linear_part = jnp.cross(omega, f) + jnp.cross(v, tau)
    angular_part = jnp.cross(omega, tau)

    return jnp.concatenate([linear_part, angular_part])


def adjoint_ref(twist1: jnp.ndarray, twist2: jnp.ndarray) -> jnp.ndarray:
    """Reference implementation of adjoint operator (Lie bracket).

    For se(3), given twists ξ₁ = [v₁; ω₁] and ξ₂ = [v₂; ω₂]:
        [ξ₁, ξ₂] = [ω₁ × v₂ - ω₂ × v₁; ω₁ × ω₂]

    Args:
        twist1: First 6D twist [v; ω]
        twist2: Second 6D twist [v; ω]

    Returns:
        6D Lie bracket result
    """
    v1 = twist1[:3]
    omega1 = twist1[3:]
    v2 = twist2[:3]
    omega2 = twist2[3:]

    linear_part = jnp.cross(omega1, v2) - jnp.cross(omega2, v1)
    angular_part = jnp.cross(omega1, omega2)

    return jnp.concatenate([linear_part, angular_part])


# =============================================================================
# AdjointDual
# =============================================================================

# --- AdjointDual: Basic Usage ---


def test_adjoint_dual_creation_and_properties():
    """Test that AdjointDual can be created and has correct properties."""
    twist = State("twist", (6,))
    momentum = State("momentum", (6,))
    ad_dual = AdjointDual(twist, momentum)

    # Check that children() returns both inputs
    assert ad_dual.children() == [twist, momentum]

    # Check repr
    assert repr(ad_dual) == f"ad_dual({twist!r}, {momentum!r})"


def test_adjoint_dual_with_constant():
    """Test that AdjointDual can be created with constant inputs."""
    twist_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    momentum_val = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    ad_dual = AdjointDual(twist_val, momentum_val)

    # Should wrap constants in to_expr
    assert len(ad_dual.children()) == 2
    assert isinstance(ad_dual.children()[0], Constant)
    assert isinstance(ad_dual.children()[1], Constant)


# --- AdjointDual: Shape Checking ---


def test_adjoint_dual_shape_inference():
    """Test that AdjointDual infers shape (6,) from 6D inputs."""
    twist = State("twist", (6,))
    momentum = State("momentum", (6,))
    ad_dual = AdjointDual(twist, momentum)

    assert ad_dual.check_shape() == (6,)


def test_adjoint_dual_shape_validation_wrong_twist_shape():
    """Test that AdjointDual raises error for non-6D twist."""
    twist = State("twist", (3,))
    momentum = State("momentum", (6,))
    ad_dual = AdjointDual(twist, momentum)

    with pytest.raises(
        ValueError, match=r"AdjointDual expects twist with shape \(6,\), got \(3,\)"
    ):
        ad_dual.check_shape()


def test_adjoint_dual_shape_validation_wrong_momentum_shape():
    """Test that AdjointDual raises error for non-6D momentum."""
    twist = State("twist", (6,))
    momentum = State("momentum", (3,))
    ad_dual = AdjointDual(twist, momentum)

    with pytest.raises(
        ValueError, match=r"AdjointDual expects momentum with shape \(6,\), got \(3,\)"
    ):
        ad_dual.check_shape()


# --- AdjointDual: Canonicalization ---


def test_adjoint_dual_canonicalize_preserves_structure():
    """Test that AdjointDual canonicalizes its children."""
    from openscvx.symbolic.expr import Add

    twist = State("twist", (6,))
    momentum = State("momentum", (6,))

    # Create expressions that can be canonicalized
    twist_expr = Add(twist, Constant(np.zeros(6)))
    momentum_expr = Add(momentum, Constant(np.zeros(6)))
    ad_dual = AdjointDual(twist_expr, momentum_expr)

    canonical = ad_dual.canonicalize()

    # Should still be an AdjointDual node
    assert isinstance(canonical, AdjointDual)
    # Children should be canonicalized
    assert canonical.children()[0] == twist
    assert canonical.children()[1] == momentum


# --- AdjointDual: JAX Lowering ---


def test_adjoint_dual_jax_lowering():
    """Test AdjointDual lowering to JAX against reference implementation."""
    test_cases = [
        # Zero twist
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        ),
        # Zero momentum
        (
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        # General case
        (
            jnp.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.4]),
            jnp.array([5.0, 3.0, -2.0, 0.5, -0.3, 0.1]),
        ),
        # Pure angular velocity (v=0)
        (
            jnp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3]),
            jnp.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5]),
        ),
    ]

    for twist_val, momentum_val in test_cases:
        # Create symbolic state
        twist = State("twist", (6,))
        twist._slice = slice(0, 6)

        # Create AdjointDual expression
        ad_dual_expr = AdjointDual(twist, Constant(momentum_val))
        fn = lower_to_jax(ad_dual_expr)
        result = fn(twist_val, None, None, None)

        # Compare against reference
        expected = adjoint_dual_ref(twist_val, momentum_val)
        assert result.shape == (6,)
        assert jnp.allclose(result, expected, atol=1e-12)


def test_adjoint_dual_euler_equation():
    """Test that ad*_ξ(μ) reduces to ω × (J @ ω) for pure rotation.

    For a rigid body with pure angular motion (v=0) and diagonal inertia J:
    - twist ξ = [0; ω]
    - momentum μ = M @ ξ = [0; J @ ω]
    - ad*_ξ(μ) = [0; ω × (J @ ω)]

    This is the Coriolis/gyroscopic term in Euler's rotational equations.
    """
    omega = jnp.array([0.1, -0.2, 0.3])
    J = jnp.array([1.0, 2.0, 3.0])  # diagonal inertia

    twist = jnp.concatenate([jnp.zeros(3), omega])
    momentum = jnp.concatenate([jnp.zeros(3), J * omega])

    result = adjoint_dual_ref(twist, momentum)

    # Linear part should be zero (no translation)
    assert jnp.allclose(result[:3], jnp.zeros(3), atol=1e-12)

    # Angular part should be ω × (J @ ω)
    expected_angular = jnp.cross(omega, J * omega)
    assert jnp.allclose(result[3:], expected_angular, atol=1e-12)


def test_adjoint_dual_rigid_body_dynamics():
    """Test AdjointDual in the context of rigid body dynamics.

    For a rigid body with spatial inertia M and twist ξ, the bias force is:
        b = ad*_ξ(M @ ξ)

    For the special case of pure rotation with diagonal inertia J:
        b = [0; ω × (J @ ω)]
    """
    # Diagonal spatial inertia (m*I for linear, J for angular)
    m = 2.0
    J = jnp.array([0.5, 1.0, 1.5])
    M = jnp.diag(jnp.concatenate([m * jnp.ones(3), J]))

    # Pure rotation (v=0)
    omega = jnp.array([0.1, 0.2, 0.3])
    twist = jnp.concatenate([jnp.zeros(3), omega])

    # Compute momentum
    momentum = M @ twist  # = [0; J @ ω]

    # Create symbolic version
    twist_state = State("twist", (6,))
    twist_state._slice = slice(0, 6)

    ad_dual_expr = AdjointDual(twist_state, Constant(momentum))
    fn = lower_to_jax(ad_dual_expr)
    result = fn(twist, None, None, None)

    # For pure rotation, the linear part should be zero
    # Angular part should be ω × (J @ ω)
    expected_angular = jnp.cross(omega, J * omega)
    expected = jnp.concatenate([jnp.zeros(3), expected_angular])

    assert jnp.allclose(result, expected, atol=1e-12)


# =============================================================================
# Adjoint
# =============================================================================

# --- Adjoint: Basic Usage ---


def test_adjoint_creation_and_properties():
    """Test that Adjoint can be created and has correct properties."""
    twist1 = State("twist1", (6,))
    twist2 = State("twist2", (6,))
    ad = Adjoint(twist1, twist2)

    # Check that children() returns both inputs
    assert ad.children() == [twist1, twist2]

    # Check repr
    assert repr(ad) == f"ad({twist1!r}, {twist2!r})"


def test_adjoint_with_constant():
    """Test that Adjoint can be created with constant inputs."""
    twist1_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    twist2_val = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    ad = Adjoint(twist1_val, twist2_val)

    # Should wrap constants in to_expr
    assert len(ad.children()) == 2
    assert isinstance(ad.children()[0], Constant)
    assert isinstance(ad.children()[1], Constant)


# --- Adjoint: Shape Checking ---


def test_adjoint_shape_inference():
    """Test that Adjoint infers shape (6,) from 6D inputs."""
    twist1 = State("twist1", (6,))
    twist2 = State("twist2", (6,))
    ad = Adjoint(twist1, twist2)

    assert ad.check_shape() == (6,)


def test_adjoint_shape_validation_wrong_twist1_shape():
    """Test that Adjoint raises error for non-6D twist1."""
    twist1 = State("twist1", (3,))
    twist2 = State("twist2", (6,))
    ad = Adjoint(twist1, twist2)

    with pytest.raises(ValueError, match=r"Adjoint expects twist1 with shape \(6,\), got \(3,\)"):
        ad.check_shape()


def test_adjoint_shape_validation_wrong_twist2_shape():
    """Test that Adjoint raises error for non-6D twist2."""
    twist1 = State("twist1", (6,))
    twist2 = State("twist2", (4,))
    ad = Adjoint(twist1, twist2)

    with pytest.raises(ValueError, match=r"Adjoint expects twist2 with shape \(6,\), got \(4,\)"):
        ad.check_shape()


# --- Adjoint: Canonicalization ---


def test_adjoint_canonicalize_preserves_structure():
    """Test that Adjoint canonicalizes its children."""
    from openscvx.symbolic.expr import Add

    twist1 = State("twist1", (6,))
    twist2 = State("twist2", (6,))

    # Create expressions that can be canonicalized
    twist1_expr = Add(twist1, Constant(np.zeros(6)))
    twist2_expr = Add(twist2, Constant(np.zeros(6)))
    ad = Adjoint(twist1_expr, twist2_expr)

    canonical = ad.canonicalize()

    # Should still be an Adjoint node
    assert isinstance(canonical, Adjoint)
    # Children should be canonicalized
    assert canonical.children()[0] == twist1
    assert canonical.children()[1] == twist2


# --- Adjoint: JAX Lowering ---


def test_adjoint_jax_lowering():
    """Test Adjoint lowering to JAX against reference implementation."""
    test_cases = [
        # Zero twists
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        ),
        # Same twist (should give zero due to antisymmetry)
        (
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ),
        # General case
        (
            jnp.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.4]),
            jnp.array([0.5, 0.3, -2.0, 0.1, -0.3, 0.2]),
        ),
    ]

    for twist1_val, twist2_val in test_cases:
        # Create symbolic states
        twist1 = State("twist1", (6,))
        twist1._slice = slice(0, 6)
        twist2 = State("twist2", (6,))
        twist2._slice = slice(0, 6)

        # Create Adjoint expression (using twist1 from state, twist2 as constant)
        ad_expr = Adjoint(twist1, Constant(twist2_val))
        fn = lower_to_jax(ad_expr)
        result = fn(twist1_val, None, None, None)

        # Compare against reference
        expected = adjoint_ref(twist1_val, twist2_val)
        assert result.shape == (6,)
        assert jnp.allclose(result, expected, atol=1e-12)


def test_adjoint_antisymmetry():
    """Test that [ξ₁, ξ₂] = -[ξ₂, ξ₁]."""
    twist1 = jnp.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.2])
    twist2 = jnp.array([0.5, 0.3, -0.1, 0.2, 0.4, -0.3])

    result_12 = adjoint_ref(twist1, twist2)
    result_21 = adjoint_ref(twist2, twist1)

    assert jnp.allclose(result_12, -result_21, atol=1e-12)


def test_adjoint_jacobi_identity():
    """Test the Jacobi identity: [ξ₁, [ξ₂, ξ₃]] + [ξ₂, [ξ₃, ξ₁]] + [ξ₃, [ξ₁, ξ₂]] = 0."""
    twist1 = jnp.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.2])
    twist2 = jnp.array([0.5, 0.3, -0.1, 0.2, 0.4, -0.3])
    twist3 = jnp.array([-0.2, 0.1, 0.4, -0.3, 0.2, 0.1])

    # [ξ₁, [ξ₂, ξ₃]]
    term1 = adjoint_ref(twist1, adjoint_ref(twist2, twist3))
    # [ξ₂, [ξ₃, ξ₁]]
    term2 = adjoint_ref(twist2, adjoint_ref(twist3, twist1))
    # [ξ₃, [ξ₁, ξ₂]]
    term3 = adjoint_ref(twist3, adjoint_ref(twist1, twist2))

    jacobi_sum = term1 + term2 + term3
    # Use atol=1e-6 for float32 precision
    assert jnp.allclose(jacobi_sum, jnp.zeros(6), atol=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================


def test_newton_euler_dynamics_with_lie_algebra():
    """Test Newton-Euler dynamics formulation using Lie algebra operators.

    For a rigid body:
        M @ ξ_dot = F_ext - ad*_ξ(M @ ξ)

    Compare against the traditional formulation.

    For twist ξ = [v; ω] and momentum μ = M @ ξ = [m*v; J @ ω]:
        ad*_ξ(μ) = [ω × (m*v) + v × (J @ ω); ω × (J @ ω)]

    The linear part has an extra term v × (J @ ω) due to momentum coupling,
    but for body-fixed frames with the body at rest this often simplifies.
    """
    # Parameters
    m = 2.0  # mass
    J = jnp.array([0.5, 1.0, 1.5])  # diagonal inertia

    # State: twist = [v; ω]
    v = jnp.array([1.0, 0.5, -0.2])  # linear velocity
    omega = jnp.array([0.1, -0.2, 0.3])  # angular velocity
    twist_val = jnp.concatenate([v, omega])

    # Momentum: μ = [m*v; J*ω] for diagonal inertia
    p = m * v  # linear momentum
    L = J * omega  # angular momentum (element-wise for diagonal J)
    momentum_val = jnp.concatenate([p, L])

    # Lie algebra formulation: ad*_ξ(μ)
    twist_state = State("twist", (6,))
    twist_state._slice = slice(0, 6)

    ad_dual_expr = AdjointDual(twist_state, Constant(momentum_val))
    lie_fn = lower_to_jax(ad_dual_expr)
    lie_result = lie_fn(twist_val, None, None, None)

    # Manual computation using the formula:
    # ad*_ξ(μ) = [ω × p + v × L; ω × L]
    linear_bias = jnp.cross(omega, p) + jnp.cross(v, L)
    angular_bias = jnp.cross(omega, L)
    expected = jnp.concatenate([linear_bias, angular_bias])

    # Results should match
    assert jnp.allclose(lie_result, expected, atol=1e-12)


def test_adjoint_dual_combined_with_dynamics():
    """Test using AdjointDual in a complete dynamics expression."""
    from openscvx.symbolic.expr import Control

    # Create symbolic state and control
    twist = State("twist", (6,))
    twist._slice = slice(0, 6)

    wrench = Control("wrench", (6,))
    wrench._slice = slice(0, 6)

    # Parameters
    m = 1.0
    J = np.diag([0.1, 0.2, 0.3])
    M = np.block([[m * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), J]])
    M_inv = np.linalg.inv(M)

    # Dynamics: twist_dot = M_inv @ (wrench - ad*_twist(M @ twist))
    M_param = Constant(M)
    M_inv_param = Constant(M_inv)

    momentum = M_param @ twist
    bias_force = AdjointDual(twist, momentum)
    twist_dot = M_inv_param @ (wrench - bias_force)

    # Lower and evaluate
    fn = lower_to_jax(twist_dot)

    # Test values
    twist_val = jnp.array([1.0, 0.5, -0.2, 0.1, -0.2, 0.3])
    wrench_val = jnp.array([0.5, 0.0, 0.0, 0.01, -0.02, 0.01])

    # twist goes in x (state), wrench goes in u (control)
    result = fn(twist_val, wrench_val, None, None)

    # Result should be 6D
    assert result.shape == (6,)

    # Verify against manual computation
    momentum_val = M @ twist_val
    bias_val = adjoint_dual_ref(twist_val, momentum_val)
    expected = M_inv @ (wrench_val - bias_val)

    assert jnp.allclose(result, expected, atol=1e-12)
