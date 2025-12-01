"""Tests for NodeReference - inter-node constraint functionality.

This module tests the NodeReference expression class which enables users to define
constraints across different trajectory nodes, such as:
- Rate limits: (position.node(k) - position.node(k-1)) <= threshold
- Multi-step dependencies: state.node(k) == 2*state.node(k-1) - state.node(k-2)
- Consistency constraints across time steps
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Control,
    NodeReference,
    State,
    Variable,
)

# =============================================================================
# NodeReference Creation and Basic Properties
# =============================================================================


def test_node_reference_creation_from_state():
    """Test creating NodeReference from State variable."""
    position = State("pos", shape=(3,))

    # Create node reference using .node() method
    pos_at_k = position.node(5)

    assert isinstance(pos_at_k, NodeReference)
    assert pos_at_k.base is position
    assert pos_at_k.node_idx == 5


def test_node_reference_creation_from_control():
    """Test creating NodeReference from Control variable."""
    thrust = Control("thrust", shape=(2,))

    thrust_at_k = thrust.node(10)

    assert isinstance(thrust_at_k, NodeReference)
    assert thrust_at_k.base is thrust
    assert thrust_at_k.node_idx == 10


def test_node_reference_creation_from_variable():
    """Test creating NodeReference from generic Variable."""
    x = Variable("x", shape=(1,))

    x_at_k = x.node(3)

    assert isinstance(x_at_k, NodeReference)
    assert x_at_k.base is x
    assert x_at_k.node_idx == 3


def test_node_reference_accepts_integers_only():
    """Test that node index only accepts integers."""
    position = State("pos", shape=(3,))

    # Valid: integer indices
    ref_abs = position.node(5)
    assert ref_abs.node_idx == 5

    # Invalid: non-integer indices
    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.node(1.5)

    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.node([1, 2])

    with pytest.raises(TypeError, match="Node index must be an integer"):
        position.node("k")


def test_node_reference_allows_negative_indices():
    """Test that negative indices are allowed for relative referencing."""
    velocity = State("vel", shape=(3,))

    # Negative indices should work (useful for k-1, k-2, etc.)
    vel_prev = velocity.node(-1)
    vel_prev2 = velocity.node(-2)

    assert vel_prev.node_idx == -1
    assert vel_prev2.node_idx == -2


# =============================================================================
# NodeReference Shape Checking
# =============================================================================


def test_node_reference_preserves_base_shape():
    """Test that NodeReference has the same shape as its base expression."""
    # Scalar state
    scalar_state = State("x", shape=(1,))
    assert scalar_state.node(0).check_shape() == (1,)

    # Vector state
    vector_state = State("pos", shape=(3,))
    assert vector_state.node(5).check_shape() == (3,)

    # Matrix-shaped control
    matrix_control = Control("gain", shape=(3, 4))
    assert matrix_control.node(2).check_shape() == (3, 4)


def test_node_reference_with_indexed_base():
    """Test NodeReference on spatially-indexed expressions."""
    position = State("pos", shape=(3,))

    # First get spatial index, then node reference
    x_component = position[0]  # Shape (,) - scalar
    x_at_k = x_component.node(5)

    # Should preserve the scalar shape from indexing
    assert x_at_k.check_shape() == ()


# =============================================================================
# NodeReference Tree Structure
# =============================================================================


def test_node_reference_children():
    """Test that NodeReference correctly reports its base as a child."""
    state = State("x", shape=(2,))
    ref = state.node(3)

    children = ref.children()
    assert len(children) == 1
    assert children[0] is state


def test_node_reference_canonicalize():
    """Test canonicalization of NodeReference expressions."""
    state = State("x", shape=(2,))
    ref = state.node(5)

    # Canonicalize should preserve node reference structure
    canon_ref = ref.canonicalize()

    assert isinstance(canon_ref, NodeReference)
    assert canon_ref.node_idx == 5
    # Base should also be canonicalized (though Leaf nodes are already canonical)
    assert isinstance(canon_ref.base, State)


# =============================================================================
# NodeReference in Arithmetic Operations
# =============================================================================


def test_node_reference_arithmetic_operations():
    """Test that NodeReference works in arithmetic operations."""
    velocity = State("vel", shape=(3,))

    # Create references at different nodes
    vel_k = velocity.node(10)
    vel_k_minus_1 = velocity.node(9)

    # Compute velocity change (delta_v)
    delta_v = vel_k - vel_k_minus_1

    # Should create a subtraction expression
    from openscvx.symbolic.expr import Sub

    assert isinstance(delta_v, Sub)

    # Check that children are the node references
    assert isinstance(delta_v.left, NodeReference)
    assert isinstance(delta_v.right, NodeReference)
    assert delta_v.left.node_idx == 10
    assert delta_v.right.node_idx == 9


def test_node_reference_with_constants():
    """Test NodeReference in operations with constants."""
    position = State("pos", shape=(3,))

    pos_k = position.node(5)

    # Add constant
    expr1 = pos_k + 1.0
    assert expr1.check_shape() == (3,)

    # Multiply by constant
    expr2 = 2.0 * pos_k
    assert expr2.check_shape() == (3,)

    # Subtract constant array
    expr3 = pos_k - np.array([1.0, 2.0, 3.0])
    assert expr3.check_shape() == (3,)


def test_node_reference_complex_expression():
    """Test NodeReference in more complex multi-step expressions."""
    state = State("x", shape=(1,))

    # Create a "Fibonacci-like" expression: x[k] == x[k-1] + x[k-2]
    x_k = state.node(10)
    x_k_minus_1 = state.node(9)
    x_k_minus_2 = state.node(8)

    recurrence = x_k - x_k_minus_1 - x_k_minus_2

    # Should be able to check shape
    assert recurrence.check_shape() == (1,)


# =============================================================================
# NodeReference in Constraints
# =============================================================================


def test_node_reference_inequality_constraint():
    """Test creating inequality constraints with NodeReference."""
    velocity = State("vel", shape=(3,))

    vel_k = velocity.node(10)
    vel_k_minus_1 = velocity.node(9)

    # Rate limit constraint: change in velocity <= max_accel
    max_accel = 0.5
    constraint = (vel_k - vel_k_minus_1) <= max_accel

    from openscvx.symbolic.expr import Inequality

    assert isinstance(constraint, Inequality)

    # Should be able to check shape (constraints return scalar)
    assert constraint.check_shape() == ()


def test_node_reference_equality_constraint():
    """Test creating equality constraints with NodeReference."""
    position = State("pos", shape=(2,))

    # Periodicity constraint: position at start equals position at end
    pos_start = position.node(0)
    pos_end = position.node(100)

    constraint = pos_start == pos_end

    from openscvx.symbolic.expr import Equality

    assert isinstance(constraint, Equality)
    assert constraint.check_shape() == ()


def test_node_reference_with_nodal_constraint():
    """Test combining NodeReference with .at() for nodal constraints."""
    position = State("pos", shape=(3,))

    # Rate limit applied at nodes 1 through 99
    N = 100
    rate_limit = 0.1

    # Create constraint: pos[k] - pos[k-1] <= rate_limit for k in 1..99
    constraint = (position.node(10) - position.node(9) <= rate_limit).at(range(1, N))

    from openscvx.symbolic.expr import NodalConstraint

    assert isinstance(constraint, NodalConstraint)
    assert constraint.nodes == list(range(1, N))


# =============================================================================
# NodeReference String Representation
# =============================================================================


def test_node_reference_repr():
    """Test string representation of NodeReference."""
    state = State("x", shape=(2,))
    ref = state.node(5)

    repr_str = repr(ref)

    # Should show the base and node index
    assert ".node(5)" in repr_str
    assert "State" in repr_str or "x" in repr_str


def test_node_reference_repr_with_negative_index():
    """Test repr with negative node index."""
    velocity = State("vel", shape=(3,))
    ref = velocity.node(-1)

    repr_str = repr(ref)
    assert ".node(-1)" in repr_str


# =============================================================================
# Integration Tests - Real-World Use Cases
# =============================================================================


def test_position_rate_constraint():
    """Test realistic position rate limiting constraint."""
    position = State("pos", shape=(3,))
    max_step = 0.1

    # Position change between consecutive nodes must be small
    # This would typically be written as:
    # constraint = (position.node(k) - position.node(k-1) <= max_step).at(range(1, N))

    pos_k = position.node(10)
    pos_k_prev = position.node(9)

    constraint = (pos_k - pos_k_prev) <= max_step

    assert constraint.check_shape() == ()

    # Apply to specific nodes
    nodal_constraint = constraint.at([1, 2, 3, 4, 5])
    from openscvx.symbolic.expr import NodalConstraint

    assert isinstance(nodal_constraint, NodalConstraint)


def test_velocity_consistency_constraint():
    """Test velocity must be consistent with position change."""
    position = State("pos", shape=(3,))
    velocity = State("vel", shape=(3,))
    dt = 0.1  # Time step

    # Velocity should match position change: vel[k] ≈ (pos[k] - pos[k-1]) / dt
    pos_k = position.node(10)
    pos_k_prev = position.node(9)
    vel_k = velocity.node(10)

    # Constraint: vel[k] == (pos[k] - pos[k-1]) / dt
    constraint = vel_k == (pos_k - pos_k_prev) / dt

    assert constraint.check_shape() == ()


def test_control_rate_limiting():
    """Test control input rate limiting."""
    thrust = Control("thrust", shape=(3,))
    max_thrust_rate = 1.0

    # Thrust can't change too quickly between time steps
    thrust_k = thrust.node(5)
    thrust_k_prev = thrust.node(4)

    # Constraint: |thrust[k] - thrust[k-1]| <= max_rate
    # This is typically enforced as two inequalities or using norm
    delta_thrust = thrust_k - thrust_k_prev

    # Simple component-wise constraint
    constraint = delta_thrust <= max_thrust_rate

    assert constraint.check_shape() == ()


def test_spatial_indexing_with_node_reference():
    """Test combining spatial indexing with node references."""
    position = State("pos", shape=(3,))

    # Rate limit only on z-component (index 2)
    z_k = position[2].node(10)
    z_k_prev = position[2].node(9)

    max_z_rate = 0.05
    constraint = (z_k - z_k_prev) <= max_z_rate

    # z component is scalar after indexing
    assert z_k.check_shape() == ()
    assert constraint.check_shape() == ()


def test_multi_step_dependency():
    """Test multi-step dependencies like second-order differences."""
    state = State("x", shape=(1,))

    # Second-order finite difference (acceleration)
    # accel = (x[k+1] - 2*x[k] + x[k-1]) / dt^2
    x_next = state.node(11)
    x_curr = state.node(10)
    x_prev = state.node(9)

    dt = 0.1
    accel = (x_next - 2 * x_curr + x_prev) / (dt**2)

    # Should be able to constrain this
    max_accel = 5.0
    constraint = accel <= max_accel

    assert constraint.check_shape() == ()


def test_boundary_coupling_constraint():
    """Test coupling initial and final states (e.g., for periodic orbits)."""
    state = State("x", shape=(2,))

    # Periodic boundary condition: state at end equals state at start
    x_start = state.node(0)
    x_end = state.node(100)

    periodicity_constraint = x_start == x_end

    from openscvx.symbolic.expr import Equality

    assert isinstance(periodicity_constraint, Equality)

    # Apply only at the boundary nodes
    constraint_at_boundary = periodicity_constraint.at([0, 100])
    from openscvx.symbolic.expr import NodalConstraint

    assert isinstance(constraint_at_boundary, NodalConstraint)


# =============================================================================
# Relative Indexing Tests (New Feature)
# =============================================================================


# Tests for relative indexing have been removed - only absolute indexing is supported
