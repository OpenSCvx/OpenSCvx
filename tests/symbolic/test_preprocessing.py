import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import Add, Concat, Constant, Control, State
from openscvx.symbolic.preprocessing import (
    collect_and_assign_slices,
    convert_dynamics_dict_to_expr,
    validate_constraints_at_root,
    validate_dynamics_dict,
    validate_dynamics_dict_dimensions,
    validate_dynamics_dimension,
    validate_variable_names,
)


def test_unique_names_passes():
    a = State("a", (2,))
    b = State("b", (2,))
    c = Control("c", (1,))
    validate_variable_names([Add(a, b), c])  # no error


def test_duplicate_names_raises():
    a1 = State("x", (2,))
    a2 = State("x", (3,))
    with pytest.raises(ValueError):
        validate_variable_names([a1, a2])


def test_repeated_same_state_across_exprs_passes():
    # same State instance appears in two different expressions → no error
    x = State("x", (2,))
    expr1 = Add(x, Constant(np.zeros((2,))))
    expr2 = Constant(np.ones((2,))) - x
    # should not raise
    validate_variable_names([expr1, expr2])


def test_two_distinct_instances_same_name_raises():
    # two *different* State objects with the same .name → error
    x1 = State("x", (2,))
    x2 = State("x", (2,))
    with pytest.raises(ValueError) as exc:
        validate_variable_names([x1, x2])
    assert "Duplicate variable name" in str(exc.value)


def test_reserved_prefix_raises():
    bad = State("_hidden", (1,))
    with pytest.raises(ValueError):
        validate_variable_names([bad])


def test_reserved_names_collision():
    s = State("foo", (1,))
    with pytest.raises(ValueError):
        validate_variable_names([s], reserved_names={"foo", "bar"})


def test_collect_single_state():
    x = State("x", (4,))
    states, controls = collect_and_assign_slices([x], [])
    assert x._slice == slice(0, 4)
    assert len(states) == 1
    assert states[0] is x
    assert len(controls) == 0


def test_collect_multiple_states_preserves_order():
    a = State("a", (2,))
    b = State("b", (3,))
    states, controls = collect_and_assign_slices([a, b], [])
    assert slice(0, 2, None) == slice(0, 2)
    assert a._slice == slice(0, 2)
    assert b._slice == slice(2, 5)
    assert len(states) == 2
    assert states[0] is a
    assert states[1] is b
    assert len(controls) == 0


def test_collect_states_and_controls_separate_namespaces():
    s1 = State("s1", (2,))
    c1 = Control("c1", (3,))
    states, controls = collect_and_assign_slices([s1], [c1])
    # states live in x; controls live in u
    assert s1._slice == slice(0, 2)
    assert c1._slice == slice(0, 3)
    assert len(states) == 1
    assert states[0] is s1
    assert len(controls) == 1
    assert controls[0] is c1


def test_states_and_controls_independent_offsets():
    # two states but only one control
    s1 = State("s1", (2,))
    s2 = State("s2", (1,))
    c1 = Control("c1", (2,))
    states, controls = collect_and_assign_slices([s1, s2], [c1])
    # states: offsets 0→2, 2→3
    assert s1._slice == slice(0, 2)
    assert s2._slice == slice(2, 3)
    # controls: offset resets to zero
    assert c1._slice == slice(0, 2)
    # verify collected variables
    assert len(states) == 2
    assert s1 in states
    assert s2 in states
    assert len(controls) == 1
    assert controls[0] is c1


def test_manual_slice_shape_mismatch_raises():
    # Create a State of dimension 3, but give it a slice of length 2
    s = State("s", (3,))
    s._slice = slice(0, 2)

    with pytest.raises(ValueError) as excinfo:
        collect_and_assign_slices([s], [])

    msg = str(excinfo.value)
    assert "Manual slice for 's'" in msg
    assert "length 2" in msg
    assert "(3,)" in msg


def test_idempotent_on_repeat_calls():
    s = State("s", (3,))
    states1, controls1 = collect_and_assign_slices([s], [])
    first = s._slice
    states2, controls2 = collect_and_assign_slices([s], [])
    assert s._slice is first
    # Same state should be collected each time
    assert len(states1) == len(states2) == 1
    assert states1[0] is s
    assert states2[0] is s


def test_manual_slice_assignment():
    s = State("s", (2,))
    s._slice = slice(0, 2)
    t = State("t", (3,))  # left to auto-assign
    states, controls = collect_and_assign_slices([s, t], [])
    assert s._slice == slice(0, 2)
    assert t._slice == slice(2, 5)
    assert len(states) == 2
    assert s in states
    assert t in states


def test_invalid_manual_slice_assignment_nonzero_start():
    # starts at nonzero:
    s = State("s", (2,))
    s._slice = slice(1, 3)
    with pytest.raises(ValueError):
        collect_and_assign_slices([s], [])


def test_invalid_manual_slice_assignment_gaps():
    # gap/overlap:
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(3, 5)
    with pytest.raises(ValueError):
        collect_and_assign_slices([a, b], [])


def test_collect_empty_expressions():
    # Test collecting from empty lists
    states, controls = collect_and_assign_slices([], [])
    assert len(states) == 0
    assert len(controls) == 0


def test_collect_only_constants():
    # Test collecting with no variables (empty lists)
    states, controls = collect_and_assign_slices([], [])
    assert len(states) == 0
    assert len(controls) == 0


def test_root_constraint_passes():
    # a == 5  is a top‐level constraint → OK
    a = Constant(jnp.array([1.0, 2.0]))
    c1 = a == 5
    c2 = a <= jnp.array([3.0, 4.0])
    # should not raise
    validate_constraints_at_root(c1)
    validate_constraints_at_root(c2)


def test_nested_constraint_raises():
    # Add(a, (b == 3))  nests a constraint under Add → should error
    a = Constant(jnp.array([1.0, 2.0]))
    b = Constant(jnp.array([3.0, 4.0]))
    nested = Add(a, b == 3)

    with pytest.raises(ValueError) as exc:
        validate_constraints_at_root(nested)
    msg = str(exc.value)
    assert "Nested Constraint found at depth 1" in msg
    assert "constraints must only appear as top-level roots" in msg


def test_ctcs_at_root_with_wrapped_constraint_passes():
    """CTCS(x <= 5) at root level should be valid, even though the constraint is at depth 1"""
    from openscvx.symbolic.expr import ctcs

    x = State("x", (1,))
    constraint = x <= 5
    wrapped = ctcs(constraint)

    # Should not raise - CTCS at root is OK, and constraint inside CTCS is exempt
    validate_constraints_at_root(wrapped)


def test_nested_ctcs_wrapper_raises():
    """Add(a, CTCS(x <= 5)) should raise error because CTCS is nested"""
    from openscvx.symbolic.expr import ctcs

    a = Constant(np.array([1.0]))
    x = State("x", (1,))
    wrapped = ctcs(x <= 5)
    nested = Add(a, wrapped)

    with pytest.raises(ValueError) as exc:
        validate_constraints_at_root(nested)
    msg = str(exc.value)
    assert "Nested constraint wrapper found at depth 1" in msg
    assert "constraint wrappers must only appear as top-level roots" in msg


def test_single_dynamics_single_state_passes():
    """Test single dynamics expression with single state - valid case"""
    x = State("pos", (2,))
    u = Control("thrust", (2,))

    # State dim = 2, dynamics dim = 2 (matches)
    dynamics = x + u  # shape (2,)

    # Should not raise
    validate_dynamics_dimension(dynamics, x)


def test_single_dynamics_multiple_states_passes():
    """Test single dynamics expression with multiple states - valid case"""
    x1 = State("pos", (2,))
    x2 = State("vel", (3,))

    # Total state dim = 2 + 3 = 5, dynamics dim = 5 (matches)
    dynamics = Concat(x1, x2)  # shape (5,)

    # Should not raise
    validate_dynamics_dimension(dynamics, [x1, x2])


def test_multiple_dynamics_single_state_passes():
    """Test multiple dynamics expressions with single state - valid case"""
    x = State("pos", (4,))

    # State dim = 4
    dynamics1 = x[:2]  # shape (2,)
    dynamics2 = x[2:]  # shape (2,)
    # Combined dynamics dim = 2 + 2 = 4 (matches)

    # Should not raise
    validate_dynamics_dimension([dynamics1, dynamics2], x)


def test_multiple_dynamics_multiple_states_passes():
    """Test multiple dynamics expressions with multiple states - valid case"""
    x1 = State("pos", (2,))
    x2 = State("vel", (2,))
    u = Control("thrust", (2,))

    # Total state dim = 2 + 2 = 4
    dynamics1 = x2  # shape (2,)
    dynamics2 = u  # shape (2,)
    # Combined dynamics dim = 2 + 2 = 4 (matches)

    # Should not raise
    validate_dynamics_dimension([dynamics1, dynamics2], [x1, x2])


def test_dynamics_dimension_mismatch_raises():
    """Test dimension mismatch between dynamics and states"""
    x = State("pos", (3,))
    u = Control("thrust", (2,))

    # State dim = 3, but dynamics dim = 2 (mismatch!)
    dynamics = u  # shape (2,)

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dimension(dynamics, x)

    msg = str(exc.value)
    assert "dimension mismatch" in msg
    assert "dynamics has dimension 2" in msg
    assert "total state dimension is 3" in msg


def test_multiple_dynamics_dimension_mismatch_raises():
    """Test dimension mismatch with multiple dynamics expressions"""
    x1 = State("pos", (2,))
    x2 = State("vel", (2,))
    u = Control("thrust", (2,))

    # Total state dim = 2 + 2 = 4
    dynamics1 = x1  # shape (2,)
    dynamics2 = u  # shape (2,)
    dynamics3 = u[:1]  # shape (1,) - this creates mismatch!
    # Combined dynamics dim = 2 + 2 + 1 = 5 ≠ 4

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dimension([dynamics1, dynamics2, dynamics3], [x1, x2])

    msg = str(exc.value)
    assert "dimension mismatch" in msg
    assert "combined dimension 5" in msg
    assert "total state dimension is 4" in msg


def test_non_vector_dynamics_raises():
    """Test that non-1D dynamics expressions raise an error"""
    x = State("pos", (4,))  # 1D state (flattened)
    matrix_expr = Constant(np.zeros((2, 2)))  # 2D expression

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dimension(matrix_expr, x)

    msg = str(exc.value)
    assert "must be 1-dimensional (vector)" in msg
    assert "got shape (2, 2)" in msg


def test_multiple_dynamics_with_non_vector_raises():
    """Test that non-1D dynamics in a list raises an error with proper indexing"""
    x = State("pos", (4,))
    u = Control("thrust", (2,))

    dynamics1 = u  # shape (2,) - valid vector
    dynamics2 = Constant(np.zeros((2, 2)))  # shape (2, 2) - invalid!

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dimension([dynamics1, dynamics2], x)

    msg = str(exc.value)
    assert "Dynamics expression 1 must be 1-dimensional" in msg
    assert "got shape (2, 2)" in msg


def test_dynamics_from_concat_passes():
    """Test using Concat to build dynamics expression"""
    x1 = State("pos", (2,))
    x2 = State("vel", (3,))
    u = Control("thrust", (2,))

    # Total state dim = 2 + 3 = 5
    # Build dynamics using Concat to match
    dynamics = Concat(x2, u)  # shape (5,) = 3 + 2

    # Should not raise
    validate_dynamics_dimension(dynamics, [x1, x2])


def test_empty_states_list_raises():
    """Test that empty states list raises appropriate error"""
    u = Control("thrust", (2,))
    dynamics = u  # shape (2,)

    # Should work with empty states (total dim = 0)
    # This might be valid in some edge cases
    with pytest.raises(ValueError) as exc:
        validate_dynamics_dimension(dynamics, [])

    msg = str(exc.value)
    assert "dimension mismatch" in msg
    assert "dynamics has dimension 2" in msg
    assert "total state dimension is 0" in msg


def test_validate_dynamics_dict_valid():
    """Test validate_dynamics_dict with matching state names and dynamics keys."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Constant(np.ones(2)),
        "y": Constant(np.zeros(3)),
    }

    # Should not raise
    validate_dynamics_dict(dynamics, states)


def test_validate_dynamics_dict_missing_state():
    """Test validate_dynamics_dict raises when state is missing from dynamics."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Constant(np.ones(2)),
        # Missing "y"
    }

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dict(dynamics, states)

    assert "States missing from dynamics" in str(exc.value)
    assert "y" in str(exc.value)


def test_validate_dynamics_dict_extra_key():
    """Test validate_dynamics_dict raises when dynamics has extra keys."""
    x = State("x", (2,))
    states = [x]

    dynamics = {
        "x": Constant(np.ones(2)),
        "z": Constant(np.ones(3)),  # Extra key
    }

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dict(dynamics, states)

    assert "Extra keys in dynamics" in str(exc.value)
    assert "z" in str(exc.value)


def test_validate_dynamics_dict_both_errors():
    """Test validate_dynamics_dict with both missing and extra keys."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Constant(np.ones(2)),
        "z": Constant(np.ones(1)),  # Extra key, missing "y"
    }

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dict(dynamics, states)

    msg = str(exc.value)
    assert "States missing from dynamics" in msg
    assert "y" in msg
    assert "Extra keys in dynamics" in msg
    assert "z" in msg


def test_validate_dynamics_dict_dimensions_valid():
    """Test validate_dynamics_dict_dimensions with matching dimensions."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Concat(Constant(1.0), Constant(2.0)),  # shape (2,)
        "y": Constant(np.ones(3)),  # shape (3,)
    }

    # Should not raise
    validate_dynamics_dict_dimensions(dynamics, states)


def test_validate_dynamics_dict_dimensions_mismatch():
    """Test validate_dynamics_dict_dimensions raises on dimension mismatch."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Constant(np.ones(2)),  # Correct: (2,)
        "y": Constant(np.ones(2)),  # Wrong: (2,) instead of (3,)
    }

    with pytest.raises(ValueError) as exc:
        validate_dynamics_dict_dimensions(dynamics, states)

    msg = str(exc.value)
    assert "Dynamics for state 'y'" in msg
    assert "has shape (2,)" in msg
    assert "but state has shape (3,)" in msg


def test_validate_dynamics_dict_dimensions_multiple_mismatches():
    """Test validate_dynamics_dict_dimensions with first mismatch reported."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    dynamics = {
        "x": Constant(np.ones(5)),  # Wrong: (5,) instead of (2,)
        "y": Constant(np.ones(1)),  # Wrong: (1,) instead of (3,)
    }

    # Should raise on first mismatch (x)
    with pytest.raises(ValueError) as exc:
        validate_dynamics_dict_dimensions(dynamics, states)

    assert "Dynamics for state 'x'" in str(exc.value)


def test_convert_dynamics_dict_to_expr_basic():
    """Test convert_dynamics_dict_to_expr with basic expressions."""
    x = State("x", (2,))
    y = State("y", (3,))
    states = [x, y]

    x_dyn = Constant(np.ones(2))
    y_dyn = Constant(np.zeros(3))

    dynamics = {
        "x": x_dyn,
        "y": y_dyn,
    }

    dynamics_converted, dynamics_concat = convert_dynamics_dict_to_expr(dynamics, states)

    # Check that dict is returned with same expressions
    assert len(dynamics_converted) == 2
    assert dynamics_converted["x"] is x_dyn
    assert dynamics_converted["y"] is y_dyn

    # Check concatenated expression is in correct order
    assert isinstance(dynamics_concat, Concat)
    assert len(dynamics_concat.exprs) == 2
    assert dynamics_concat.exprs[0] is x_dyn  # x first (states order)
    assert dynamics_concat.exprs[1] is y_dyn  # y second


def test_convert_dynamics_dict_to_expr_scalar_conversion():
    """Test that scalar values are converted to Constant expressions."""
    x = State("x", (1,))
    y = State("y", (1,))
    states = [x, y]

    dynamics = {
        "x": 1.0,  # Scalar
        "y": Constant(2.0),  # Already an Expr
    }

    dynamics_converted, dynamics_concat = convert_dynamics_dict_to_expr(dynamics, states)

    # Check that scalar was converted to Constant
    assert isinstance(dynamics_converted["x"], Constant)
    assert dynamics_converted["x"].value == 1.0
    assert dynamics_converted["y"] is dynamics["y"]

    # Check concatenated expression
    assert isinstance(dynamics_concat, Concat)
    assert len(dynamics_concat.exprs) == 2


def test_convert_dynamics_dict_to_expr_ordering():
    """Test that convert_dynamics_dict_to_expr respects state ordering."""
    a = State("a", (1,))
    b = State("b", (1,))
    c = State("c", (1,))
    states = [a, b, c]

    # Create dynamics in different order than states
    dynamics = {
        "c": Constant(3.0),
        "a": Constant(1.0),
        "b": Constant(2.0),
    }

    dynamics_converted, dynamics_concat = convert_dynamics_dict_to_expr(dynamics, states)

    # Concatenation should follow states order: a, b, c
    assert isinstance(dynamics_concat, Concat)
    assert len(dynamics_concat.exprs) == 3
    assert dynamics_concat.exprs[0].value == 1.0  # a
    assert dynamics_concat.exprs[1].value == 2.0  # b
    assert dynamics_concat.exprs[2].value == 3.0  # c


def test_convert_dynamics_dict_to_expr_doesnt_mutate_input():
    """Test that convert_dynamics_dict_to_expr doesn't mutate the input dict."""
    x = State("x", (1,))
    states = [x]

    original_dynamics = {
        "x": 5.0,  # Scalar that will be converted
    }

    dynamics_converted, _ = convert_dynamics_dict_to_expr(original_dynamics, states)

    # Original should still have scalar value
    assert original_dynamics["x"] == 5.0
    assert isinstance(original_dynamics["x"], float)

    # Converted should have Constant
    assert isinstance(dynamics_converted["x"], Constant)
