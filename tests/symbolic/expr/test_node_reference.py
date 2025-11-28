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


def test_node_reference_accepts_integer_or_string():
    """Test that node index accepts both integers and strings."""
    position = State("pos", shape=(3,))

    # Valid: integer indices (absolute)
    ref_abs = position.node(5)
    assert ref_abs.node_idx == 5
    assert not ref_abs.is_relative

    # Valid: string indices (relative)
    ref_rel = position.node("k")
    assert ref_rel.node_idx == "k"
    assert ref_rel.is_relative

    # Invalid: non-integer, non-string indices
    with pytest.raises(TypeError, match="Node index must be an integer or string"):
        position.node(1.5)

    with pytest.raises(TypeError, match="Node index must be an integer or string"):
        position.node([1, 2])


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


def test_relative_node_reference_parsing_basic():
    """Test parsing of relative node index strings."""
    position = State("pos", shape=(3,))

    # Test 'k' (offset 0)
    pos_k = position.node("k")
    assert pos_k.is_relative
    assert pos_k.offset == 0
    assert pos_k.node_idx == "k"

    # Test 'k-1' (offset -1)
    pos_k_minus_1 = position.node("k-1")
    assert pos_k_minus_1.is_relative
    assert pos_k_minus_1.offset == -1

    # Test 'k+2' (offset +2)
    pos_k_plus_2 = position.node("k+2")
    assert pos_k_plus_2.is_relative
    assert pos_k_plus_2.offset == 2


def test_relative_node_reference_parsing_various_offsets():
    """Test parsing various offset patterns."""
    state = State("x", shape=(1,))

    # Large offsets
    assert state.node("k-10").offset == -10
    assert state.node("k+25").offset == 25

    # Single digit
    assert state.node("k-5").offset == -5
    assert state.node("k+7").offset == 7


def test_relative_node_reference_parsing_with_whitespace():
    """Test that whitespace is handled correctly."""
    state = State("x", shape=(1,))

    # Whitespace should be stripped
    assert state.node("k - 1").offset == -1
    assert state.node("k + 2").offset == 2
    assert state.node(" k ").offset == 0


def test_relative_node_reference_invalid_format():
    """Test that invalid relative index formats raise errors."""
    state = State("x", shape=(1,))

    # Must start with 'k'
    with pytest.raises(ValueError, match="Relative node index must start with 'k'"):
        state.node("j")

    with pytest.raises(ValueError, match="Invalid relative node index format"):
        state.node("k1")  # Missing operator

    with pytest.raises(ValueError, match="Invalid relative node index format"):
        state.node("k-")  # Missing number

    with pytest.raises(ValueError, match="Invalid relative node index format"):
        state.node("k+")  # Missing number

    with pytest.raises(ValueError, match="Invalid relative node index format"):
        state.node("kk")


def test_relative_indexing_in_constraint():
    """Test using relative indexing in constraints."""
    position = State("pos", shape=(3,))

    # Create rate limit constraint using relative indexing
    pos_k = position.node("k")
    pos_k_prev = position.node("k-1")

    rate_constraint = (pos_k - pos_k_prev) <= 0.1

    # Should work with .at()
    nodal_constraint = rate_constraint.at(range(1, 10))

    from openscvx.symbolic.expr import NodalConstraint

    assert isinstance(nodal_constraint, NodalConstraint)
    assert nodal_constraint.nodes == list(range(1, 10))


def test_absolute_vs_relative_detection():
    """Test that absolute and relative modes are correctly detected."""
    state = State("x", shape=(1,))

    # Absolute
    abs_ref = state.node(5)
    assert abs_ref.is_absolute()
    assert not abs_ref.is_relative

    # Relative
    rel_ref = state.node("k")
    assert not rel_ref.is_absolute()
    assert rel_ref.is_relative


# =============================================================================
# Bounds Checking Tests
# =============================================================================


def test_bounds_checking_relative_valid():
    """Test bounds checking for valid relative indexing."""
    position = State("pos", shape=(3,))
    N = 10

    # Valid: k-1 at nodes 1..9 (accesses 0..8)
    constraint = (position.node("k") - position.node("k-1") <= 0.1).at(range(1, N))
    constraint.validate_bounds(N)  # Should not raise


def test_bounds_checking_relative_too_low():
    """Test bounds checking catches negative index access."""
    position = State("pos", shape=(3,))
    N = 10

    # Invalid: k-1 at node 0 would access node -1
    constraint = (position.node("k") - position.node("k-1") <= 0.1).at([0])

    with pytest.raises(ValueError, match="accesses invalid node index -1"):
        constraint.validate_bounds(N)


def test_bounds_checking_relative_too_high():
    """Test bounds checking catches out-of-bounds high access."""
    position = State("pos", shape=(3,))
    N = 10

    # Invalid: k+1 at node 9 would access node 10 (>= N)
    constraint = (position.node("k") - position.node("k+1") <= 0.1).at([9])

    with pytest.raises(ValueError, match="accesses invalid node index 10"):
        constraint.validate_bounds(N)


def test_bounds_checking_relative_multiple_offsets():
    """Test bounds checking with multiple offsets (like k, k-1, k-2)."""
    state = State("x", shape=(1,))
    N = 10

    # Valid: k, k-1, k-2 at nodes 2..9
    constraint = (state.node("k") - 2 * state.node("k-1") + state.node("k-2") <= 0.1).at(
        range(2, N)
    )
    constraint.validate_bounds(N)  # Should not raise

    # Invalid: same constraint at node 1 (would access k-2 = -1)
    constraint_invalid = (state.node("k") - 2 * state.node("k-1") + state.node("k-2") <= 0.1).at(
        [1]
    )

    with pytest.raises(ValueError, match="accesses invalid node index -1"):
        constraint_invalid.validate_bounds(N)


def test_bounds_checking_absolute_valid():
    """Test bounds checking for valid absolute indexing."""
    position = State("pos", shape=(3,))
    N = 10

    # Valid: references to nodes 0 and 9
    constraint = (position.node(0) == position.node(9)).at([0])
    constraint.validate_bounds(N)  # Should not raise


def test_bounds_checking_absolute_too_high():
    """Test bounds checking catches absolute index >= N."""
    position = State("pos", shape=(3,))
    N = 10

    # Invalid: reference to node 10 (>= N)
    constraint = (position.node(10) == position.node(0)).at([0])

    with pytest.raises(ValueError, match="invalid absolute node index 10"):
        constraint.validate_bounds(N)


def test_bounds_checking_absolute_negative():
    """Test bounds checking catches negative absolute indices."""
    position = State("pos", shape=(3,))
    N = 10

    # Invalid: negative absolute index
    constraint = (position.node(-1) == position.node(0)).at([0])

    with pytest.raises(ValueError, match="invalid absolute node index -1"):
        constraint.validate_bounds(N)


def test_bounds_checking_mixed_mode_error():
    """Test that mixing relative and absolute indexing raises error during collection."""
    from openscvx.symbolic.lower import collect_node_references

    position = State("pos", shape=(3,))

    # Mix absolute and relative - should raise during analysis
    constraint_expr = position.node(5) - position.node("k")

    with pytest.raises(ValueError, match="Cannot mix relative.*and absolute"):
        collect_node_references(constraint_expr)


# =============================================================================
# Cross-Node Constraint Detection
# =============================================================================


def test_contains_node_reference():
    """Test detection of NodeReference in expressions."""
    from openscvx.symbolic.lower import contains_node_reference

    position = State("pos", shape=(3,))

    # Regular expression - no NodeReference
    regular_expr = position + 1.0
    assert not contains_node_reference(regular_expr)

    # With NodeReference
    cross_node_expr = position.node("k") - position.node("k-1")
    assert contains_node_reference(cross_node_expr)

    # Deeply nested
    nested_expr = (position.node("k") - position.node("k-1")) * 2.0 + 1.0
    assert contains_node_reference(nested_expr)


def test_collect_node_references_relative():
    """Test collecting node references from relative indexing expressions."""
    from openscvx.symbolic.lower import collect_node_references

    position = State("pos", shape=(3,))

    # Single reference
    expr1 = position.node("k")
    refs, is_relative = collect_node_references(expr1)
    assert is_relative
    assert refs == [0]

    # Two references (k, k-1)
    expr2 = position.node("k") - position.node("k-1")
    refs, is_relative = collect_node_references(expr2)
    assert is_relative
    assert refs == [-1, 0]  # Sorted offsets

    # Three references (k, k-1, k-2)
    expr3 = position.node("k") - 2 * position.node("k-1") + position.node("k-2")
    refs, is_relative = collect_node_references(expr3)
    assert is_relative
    assert refs == [-2, -1, 0]


def test_collect_node_references_absolute():
    """Test collecting node references from absolute indexing expressions."""
    from openscvx.symbolic.lower import collect_node_references

    position = State("pos", shape=(3,))

    # Single reference
    expr1 = position.node(5)
    refs, is_relative = collect_node_references(expr1)
    assert not is_relative
    assert refs == [5]

    # Two references
    expr2 = position.node(0) - position.node(10)
    refs, is_relative = collect_node_references(expr2)
    assert not is_relative
    assert refs == [0, 10]


def test_collect_node_references_no_refs():
    """Test collecting from expression with no NodeReferences."""
    from openscvx.symbolic.lower import collect_node_references

    position = State("pos", shape=(3,))

    # No NodeReferences
    expr = position + 1.0
    refs, is_relative = collect_node_references(expr)
    assert not is_relative  # Defaults to False
    assert refs == []


# =============================================================================
# Absolute Mode Fixed Reference Semantics Tests
# =============================================================================


def test_absolute_mode_fixed_reference_semantics():
    """Test that absolute mode always references the same fixed nodes.

    This tests the semantic behavior: position.node(3) should always access
    node 3, regardless of which eval_node the constraint is evaluated at.
    """
    import jax.numpy as jnp

    from openscvx.symbolic.lower import create_cross_node_wrapper, lower_to_jax

    position = State("pos", shape=(2,))
    position._slice = slice(0, 2)  # Manually assign slice for testing

    # Absolute constraint: position[5] - position[3]
    expr = position.node(5) - position.node(3)

    # Lower to JAX
    constraint_fn = lower_to_jax(expr)

    # Create wrapper for evaluation at multiple nodes
    wrapped_fn = create_cross_node_wrapper(
        constraint_fn,
        references=[3, 5],  # Absolute indices
        is_relative=False,
        eval_nodes=[0, 1, 2, 10],  # Evaluate at these nodes
    )

    # Create fake trajectory
    X = jnp.arange(20).reshape(10, 2).astype(float)  # 10 nodes, 2-dim state
    U = jnp.zeros((10, 0))
    params = {}

    # Evaluate wrapped constraint
    results = wrapped_fn(X, U, params)

    # Expected: should return the same value (X[5] - X[3]) repeated 4 times
    expected_value = X[5] - X[3]  # [10, 11] - [6, 7] = [4, 4]
    expected = jnp.tile(expected_value, 4)  # Repeat for 4 eval_nodes

    assert results.shape == (8,)  # 4 eval_nodes * 2-dim state
    assert jnp.allclose(results, expected)


def test_absolute_vs_relative_semantics():
    """Compare absolute vs relative mode to show semantic difference."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import create_cross_node_wrapper, lower_to_jax

    position = State("pos", shape=(1,))
    position._slice = slice(0, 1)  # Manually assign slice for testing

    # Create trajectory
    X = jnp.arange(10).reshape(10, 1).astype(float)  # [0, 1, 2, ..., 9]
    U = jnp.zeros((10, 0))
    params = {}

    # Absolute mode: position.node(5) - position.node(3)
    abs_expr = position.node(5) - position.node(3)
    abs_fn = lower_to_jax(abs_expr)
    abs_wrapped = create_cross_node_wrapper(abs_fn, [3, 5], False, [6, 7, 8])
    abs_results = abs_wrapped(X, U, params)

    # Should always be X[5] - X[3] = 5 - 3 = 2
    assert jnp.allclose(abs_results, jnp.array([2.0, 2.0, 2.0]))

    # Relative mode: position.node('k') - position.node('k-2')
    rel_expr = position.node("k") - position.node("k-2")
    rel_fn = lower_to_jax(rel_expr)
    rel_wrapped = create_cross_node_wrapper(rel_fn, [-2, 0], True, [6, 7, 8])
    rel_results = rel_wrapped(X, U, params)

    # Should be X[6]-X[4]=2, X[7]-X[5]=2, X[8]-X[6]=2
    assert jnp.allclose(rel_results, jnp.array([2.0, 2.0, 2.0]))

    # Different eval nodes in relative mode give different results
    rel_wrapped2 = create_cross_node_wrapper(rel_fn, [-2, 0], True, [3, 4, 5])
    rel_results2 = rel_wrapped2(X, U, params)

    # Should be X[3]-X[1]=2, X[4]-X[2]=2, X[5]-X[3]=2
    assert jnp.allclose(rel_results2, jnp.array([2.0, 2.0, 2.0]))
