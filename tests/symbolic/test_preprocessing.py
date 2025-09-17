import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import Add, Concat, Constant, Hstack, Vstack
from openscvx.backend.preprocessing import (
    collect_and_assign_slices,
    validate_constraints_at_root,
    validate_dynamics_dimension,
    validate_shapes,
    validate_variable_names,
)
from openscvx.backend.state import State


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
    expr = Add(x, Constant(1.0))
    states, controls = collect_and_assign_slices([expr])
    assert x._slice == slice(0, 4)
    assert len(states) == 1
    assert states[0] is x
    assert len(controls) == 0


def test_collect_multiple_states_preserves_order():
    a = State("a", (2,))
    b = State("b", (3,))
    states, controls = collect_and_assign_slices([Add(a, b)])
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
    states, controls = collect_and_assign_slices([Add(s1, c1)])
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
    exprs = [Add(s1, s2), Add(c1, Constant(0.0))]
    states, controls = collect_and_assign_slices(exprs)
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
        collect_and_assign_slices([s])

    msg = str(excinfo.value)
    assert "Manual slice for 's'" in msg
    assert "length 2" in msg
    assert "(3,)" in msg


def test_idempotent_on_repeat_calls():
    s = State("s", (3,))
    states1, controls1 = collect_and_assign_slices([Add(s, Constant(0.0))])
    first = s._slice
    states2, controls2 = collect_and_assign_slices([Add(s, Constant(0.0))])
    assert s._slice is first
    # Same state should be collected each time
    assert len(states1) == len(states2) == 1
    assert states1[0] is s
    assert states2[0] is s


def test_manual_slice_assignment():
    s = State("s", (2,))
    s._slice = slice(0, 2)
    t = State("t", (3,))  # left to auto-assign
    states, controls = collect_and_assign_slices([Add(s, t)])
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
        collect_and_assign_slices([s])


def test_invalid_manual_slice_assignment_gaps():
    # gap/overlap:
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(3, 5)
    with pytest.raises(ValueError):
        collect_and_assign_slices([Add(a, b)])


def test_collect_no_duplicates():
    # Test that the same variable appearing multiple times is only collected once
    x = State("x", (2,))
    y = State("y", (3,))
    # x appears in multiple places
    expr1 = Add(x, y)
    expr2 = Add(x, Constant(1.0))
    expr3 = x * 2.0

    states, controls = collect_and_assign_slices([expr1, expr2, expr3])

    # x should only appear once in the states list
    assert len(states) == 2
    # Use identity checks since __eq__ is overloaded for creating constraints
    assert any(s is x for s in states)
    assert any(s is y for s in states)
    # Count using identity
    assert sum(1 for s in states if s is x) == 1  # x appears exactly once
    assert sum(1 for s in states if s is y) == 1  # y appears exactly once
    assert len(controls) == 0


def test_collect_empty_expressions():
    # Test collecting from empty expression list
    states, controls = collect_and_assign_slices([])
    assert len(states) == 0
    assert len(controls) == 0


def test_collect_only_constants():
    # Test collecting from expressions with no variables
    expr = Add(Constant(1.0), Constant(2.0))
    states, controls = collect_and_assign_slices([expr])
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
    from openscvx.backend.expr import ctcs

    x = State("x", (1,))
    constraint = x <= 5
    wrapped = ctcs(constraint)

    # Should not raise - CTCS at root is OK, and constraint inside CTCS is exempt
    validate_constraints_at_root(wrapped)


def test_nested_ctcs_wrapper_raises():
    """Add(a, CTCS(x <= 5)) should raise error because CTCS is nested"""
    from openscvx.backend.expr import ctcs

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


def test_add_same_shape_passes():
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((2, 3)))
    validate_shapes(a + b)


def test_add_shape_mismatch_raises():
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3, 2)))
    with pytest.raises(ValueError):
        validate_shapes(a + b)


def test_sub_same_shape_passes():
    a = Constant(np.zeros((4,)))
    b = Constant(np.ones((4,)))
    validate_shapes(a - b)


def test_sub_shape_mismatch_raises():
    a = Constant(np.zeros((4,)))
    b = Constant(np.ones((5,)))
    with pytest.raises(ValueError):
        validate_shapes(a - b)


def test_mul_same_shape_passes():
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((2, 2)))
    validate_shapes(a * b)


def test_mul_shape_mismatch_raises():
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((2, 3)))
    with pytest.raises(ValueError):
        validate_shapes(a * b)


def test_div_array_by_scalar_passes():
    a = Constant(np.zeros((3,)))
    b = Constant(np.array(2.0))
    validate_shapes(a / b)


def test_div_shape_mismatch_raises():
    a = Constant(np.zeros((3,)))
    b = Constant(np.zeros((2,)))
    with pytest.raises(ValueError):
        validate_shapes(a / b)


def test_matmul_ok():
    a = Constant(np.zeros((4, 5)))
    b = Constant(np.zeros((5, 2)))
    validate_shapes(a @ b)


def test_matmul_incompatible_raises():
    a = Constant(np.zeros((4, 5)))
    b = Constant(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        validate_shapes(a @ b)


def test_concat_1d_passes():
    a = Constant(np.zeros((2,)))
    b = Constant(np.ones((3,)))
    validate_shapes(Concat(a, b))


def test_concat_rank_mismatch_raises():
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((3, 2, 2)))  # Changed to (3, 2, 2) to avoid squeeze collapsing dimensions
    with pytest.raises(ValueError):
        validate_shapes(Concat(a, b))


def test_concat_nonzero_axes_mismatch_raises():
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3, 4)))
    # shapes (2,3) vs (3,4) agree on rank but not on axis>0
    with pytest.raises(ValueError):
        validate_shapes(Concat(a, b))


def test_index_valid_passes():
    a = Constant(np.zeros((5,)))
    validate_shapes(a[2:4])


def test_index_out_of_bounds_raises():
    a = Constant(np.zeros((3,)))
    with pytest.raises(ValueError):
        validate_shapes(a[5])


def test_constraint_zero_dim_scalar_passes():
    # a true scalar (shape=()) on both sides
    a = Constant(np.array(2.5))
    c = a == 1.0
    validate_shapes(c)


def test_constraint_length1_array_passes():
    # 1-element arrays count as "scalar"
    b = Constant(np.array([7.0]))
    c = b <= np.ones((1,))
    validate_shapes(c)


def test_constraint_vector_passes():
    """Vector constraints should now pass validation (interpreted element-wise)"""
    a = Constant(np.zeros((2,)))
    c = a <= np.ones((2,))
    validate_shapes(c)  # Should NOT raise


def test_constraint_shape_mismatch_raises():
    """Shape mismatches should still error out"""
    a = Constant(np.zeros((2,)))
    c = a == np.zeros((3,))
    with pytest.raises(ValueError):
        validate_shapes(c)


def test_constraint_broadcasting_passes():
    """Test constraint broadcasting: scalar op vector"""
    x = State("x", (3,))
    c = Constant(np.array(0.0)) <= x  # broadcasts to vector constraint
    validate_shapes(c)


def test_ctcs_basic_shape_validation():
    """Test basic CTCS shape validation with penalty expression checking"""
    from openscvx.backend.expr import ctcs

    x = State("x", (3,))
    constraint = x <= np.ones((3,))
    wrapped = ctcs(constraint, penalty="squared_relu")

    # Should validate both constraint and penalty expression shapes
    validate_shapes(wrapped)


def test_ctcs_penalty_shape_consistency():
    """Test that penalty expressions have same shape as constraint LHS"""
    from openscvx.backend.expr import ctcs
    from openscvx.backend.preprocessing import dispatch

    x = State("x", (2, 2))  # matrix state
    constraint = x >= np.zeros((2, 2))
    wrapped = ctcs(constraint, penalty="huber")

    validate_shapes(wrapped)

    # Penalty should have same shape as constraint LHS
    penalty_expr = wrapped.penalty_expr()
    penalty_shape = dispatch(penalty_expr)
    assert penalty_shape == ()


def test_ctcs_constraint_shape_mismatch_raises():
    """Test that CTCS catches underlying constraint shape mismatches"""
    from openscvx.backend.expr import ctcs

    x = State("x", (2,))
    # Create constraint with mismatched shapes
    constraint = x <= np.ones((3,))  # 2 vs 3 mismatch
    wrapped = ctcs(constraint)

    # Should raise due to underlying constraint shape mismatch
    with pytest.raises(ValueError):
        validate_shapes(wrapped)


def test_constant_normalization_invariant():
    """Test that different ways of creating constants are normalized consistently"""
    import numpy as np

    # Test scalar normalization
    scalar = Constant(5.0)
    array_1d = Constant(np.array([5.0]))
    array_2d = Constant(np.array([[5.0]]))

    # All should have same shape and value after normalization
    assert scalar.value.shape == array_1d.value.shape == array_2d.value.shape
    assert np.allclose(scalar.value, array_1d.value)
    assert np.allclose(scalar.value, array_2d.value)

    # Test vector normalization
    vector = Constant(np.array([1.0, 2.0, 3.0]))
    wrapped_vector = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1, 3) shape

    assert vector.value.shape == wrapped_vector.value.shape == (3,)
    assert np.array_equal(vector.value, wrapped_vector.value)

    # Test that meaningful dimensions are preserved
    matrix = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert matrix.value.shape == (2, 2)

    # Test multiple singleton dimensions
    multi_singleton = Constant(np.array([[[[1.0], [2.0]]]]))  # (1, 1, 2, 1)
    assert multi_singleton.value.shape == (2,)
    assert np.array_equal(multi_singleton.value, [1.0, 2.0])

    # Test the exact case from the old canonicalizer tests: (1, 1, 3, 1) -> (3,)
    array_multi_singleton = np.array([[[[1.0], [2.0], [3.0]]]])
    assert array_multi_singleton.shape == (1, 1, 3, 1)
    const = Constant(array_multi_singleton)
    assert const.value.shape == (3,)
    assert np.array_equal(const.value, [1.0, 2.0, 3.0])


def test_constant_normalization_validation_invariant():
    """Test that preprocessing validation catches improperly normalized constants"""
    # This test verifies our validation works, but we shouldn't be able to create
    # improperly normalized constants anymore due to the new __init__ logic

    # Create a properly normalized constant
    c = Constant(np.array([1.0, 2.0]))
    validate_shapes(c)  # Should not raise

    # Test that validation would catch violation if it occurred
    # (Though this shouldn't happen with new __init__ logic)
    assert c.value.shape == np.squeeze(c.value).shape


def test_to_expr_normalization_consistency():
    """Test that to_expr creates properly normalized constants"""
    from openscvx.backend.expr import to_expr

    # Different ways of creating same value through to_expr
    expr1 = to_expr(5.0)
    expr2 = to_expr([5.0])
    expr3 = to_expr([[5.0]])

    # All should be identical after normalization
    assert isinstance(expr1, Constant)
    assert isinstance(expr2, Constant)
    assert isinstance(expr3, Constant)

    assert expr1.value.shape == expr2.value.shape == expr3.value.shape
    assert np.allclose(expr1.value, expr2.value)
    assert np.allclose(expr1.value, expr3.value)


def test_constant_repr_format():
    """Test that constant repr shows clean Python values, not numpy arrays"""

    # Scalar should show as plain number
    scalar = Constant(1.5)
    assert repr(scalar) == "Const(1.5)"

    # Vector should show as Python list
    vector = Constant([1.0, 2.0, 3.0])
    assert repr(vector) == "Const([1.0, 2.0, 3.0])"

    # Matrix should show as nested Python list
    matrix = Constant([[1.0, 2.0], [3.0, 4.0]])
    assert repr(matrix) == "Const([[1.0, 2.0], [3.0, 4.0]])"

    # Verify that constants created with different input types have same repr
    scalar_from_array = Constant(np.array([1.5]))  # Gets squeezed to scalar
    assert repr(scalar_from_array) == "Const(1.5)"

    vector_from_nested = Constant(np.array([[1.0, 2.0, 3.0]]))  # Gets squeezed to vector
    assert repr(vector_from_nested) == "Const([1.0, 2.0, 3.0])"


def test_constant_normalization_preserves_broadcasting():
    """Test that normalized constants still broadcast correctly with other expressions"""

    # These should broadcast correctly after normalization
    scalar = Constant([[5.0]])  # (1,1) -> () after squeeze
    vector = Constant([1.0, 2.0, 3.0])  # (3,) stays (3,)
    matrix = Constant([[1.0, 2.0], [3.0, 4.0]])  # (2,2) stays (2,2)

    # Verify normalization happened
    assert scalar.value.shape == ()
    assert vector.value.shape == (3,)
    assert matrix.value.shape == (2, 2)

    # Broadcasting should still work with normalized constants
    scalar_plus_vector = scalar + vector  # () + (3,) should broadcast to (3,)
    validate_shapes(scalar_plus_vector)  # Should not raise

    # Test broadcasting between normalized constants
    scalar_times_matrix = scalar * matrix  # () * (2,2) should broadcast to (2,2)
    validate_shapes(scalar_times_matrix)  # Should not raise

    # Vector with matrix should fail (non-broadcastable)
    with pytest.raises(ValueError):
        validate_shapes(vector + matrix)  # (3,) + (2,2) should fail


def test_vector_constraints_with_normalized_constants():
    """Test that vector constraints work correctly with normalized constants"""

    x = State("x", (3,))

    # Different ways of creating same constraint bounds - all should normalize to same thing
    bounds1 = Constant(np.array([1.0, 2.0, 3.0]))  # Already (3,)
    bounds2 = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1,3) -> (3,) after squeeze
    bounds3 = Constant(np.array([[[1.0]], [[2.0]], [[3.0]]]))  # (3,1,1) -> (3,) after squeeze

    # All should have same normalized shape
    assert bounds1.value.shape == (3,)
    assert bounds2.value.shape == (3,)
    assert bounds3.value.shape == (3,)
    assert np.array_equal(bounds1.value, bounds2.value)
    assert np.array_equal(bounds1.value, bounds3.value)

    # All constraints should validate successfully
    constraint1 = x <= bounds1
    constraint2 = x <= bounds2
    constraint3 = x <= bounds3

    validate_shapes([constraint1, constraint2, constraint3])  # Should not raise

    # Broadcasting constraint: scalar bound with vector state
    scalar_bound = Constant([[2.0]])  # (1,1) -> () after squeeze
    assert scalar_bound.value.shape == ()

    scalar_constraint = x <= scalar_bound  # (3,) <= () should broadcast
    validate_shapes(scalar_constraint)  # Should not raise


def test_constant_normalization_preserves_dtype():
    """Test that normalization preserves numpy dtypes correctly"""

    # Test different dtypes with singleton dimensions
    int32_array = Constant(np.array([[1, 2, 3]], dtype=np.int32))  # (1,3) -> (3,)
    int64_array = Constant(np.array([[1, 2, 3]], dtype=np.int64))  # (1,3) -> (3,)
    float32_array = Constant(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))  # (1,3) -> (3,)
    float64_array = Constant(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))  # (1,3) -> (3,)
    bool_array = Constant(np.array([[True, False, True]], dtype=np.bool_))  # (1,3) -> (3,)

    # Verify shapes were squeezed
    assert int32_array.value.shape == (3,)
    assert int64_array.value.shape == (3,)
    assert float32_array.value.shape == (3,)
    assert float64_array.value.shape == (3,)
    assert bool_array.value.shape == (3,)

    # Verify dtypes were preserved
    assert int32_array.value.dtype == np.int32
    assert int64_array.value.dtype == np.int64
    assert float32_array.value.dtype == np.float32
    assert float64_array.value.dtype == np.float64
    assert bool_array.value.dtype == np.bool_

    # Test scalar dtypes
    scalar_int = Constant(np.array([[42]], dtype=np.int32))  # (1,1) -> ()
    scalar_float = Constant(np.array([[3.14]], dtype=np.float64))  # (1,1) -> ()

    assert scalar_int.value.shape == ()
    assert scalar_float.value.shape == ()
    assert scalar_int.value.dtype == np.int32
    assert scalar_float.value.dtype == np.float64
    assert scalar_int.value == 42
    assert scalar_float.value == 3.14


def test_hstack_basic_passes():
    """Test basic horizontal stacking functionality"""
    a = Constant(np.array([1.0, 2.0]))  # (2,)
    b = Constant(np.array([3.0, 4.0, 5.0]))  # (3,)

    stacked = Hstack([a, b])
    validate_shapes(stacked)

    from openscvx.backend.preprocessing import dispatch

    result_shape = dispatch(stacked)
    assert result_shape == (5,)  # 2 + 3 = 5


def test_hstack_dimension_mismatch_raises():
    """Test that arrays with different numbers of dimensions raise error"""
    a = Constant(np.zeros((2,)))  # 1D
    b = Constant(np.ones((2, 3)))  # 2D

    with pytest.raises(ValueError) as exc:
        validate_shapes(Hstack([a, b]))
    assert "dimensions" in str(exc.value)


def test_vstack_basic_passes():
    """Test basic vertical stacking functionality"""
    a = Constant(np.zeros((2, 3)))  # (2, 3)
    b = Constant(np.ones((4, 3)))  # (4, 3)

    stacked = Vstack([a, b])
    validate_shapes(stacked)

    from openscvx.backend.preprocessing import dispatch

    result_shape = dispatch(stacked)
    assert result_shape == (6, 3)  # 2 + 4 = 6 rows


def test_vstack_trailing_dimension_mismatch_raises():
    """Test that arrays with mismatched trailing dimensions raise error"""
    a = Constant(np.zeros((2, 3)))  # (2, 3)
    b = Constant(np.ones((4, 5)))  # (4, 5) - different second dim

    with pytest.raises(ValueError) as exc:
        validate_shapes(Vstack([a, b]))
    assert "trailing dimensions" in str(exc.value)
