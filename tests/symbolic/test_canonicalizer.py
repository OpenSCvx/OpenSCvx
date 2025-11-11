import numpy as np

from openscvx.symbolic.expr import (
    Add,
    Concat,
    Constant,
    Div,
    Equality,
    Index,
    Inequality,
    Mul,
    Neg,
    NodalConstraint,
    Parameter,
    State,
    Sub,
    to_expr,
)


def test_flatten_and_fold_add():
    a = to_expr(1)
    b = to_expr(2)
    c = to_expr(3)
    nested = Add(Add(a, b), c, to_expr(4))
    result = nested.canonicalize()
    # should be Add(1,2,3,4) then folded to Constant(10)
    assert isinstance(result, Constant)
    assert result.value == 10


def test_add_eliminate_zero_and_singleton():
    x = to_expr(5)
    zero = to_expr(0)
    # 5 + 0 + 0 ⇒ 5
    expr = Add(zero, x, zero)
    result = expr.canonicalize()
    assert not isinstance(result, Add)
    assert isinstance(result, Constant)
    assert result.value == 5


def test_flatten_and_fold_mul():
    a = to_expr(2)
    b = to_expr(3)
    c = to_expr(4)
    nested = Mul(Mul(a, b), c, to_expr(5))
    result = nested.canonicalize()
    # 2*3*4*5 = 120
    assert isinstance(result, Constant)
    assert result.value == 120


def test_mul_eliminate_one_and_singleton():
    x = to_expr(7)
    one = to_expr(1)
    expr = Mul(one, x, one)
    result = expr.canonicalize()
    assert not isinstance(result, Mul)
    assert isinstance(result, Constant)
    assert result.value == 7


def test_sub_constant_folding():
    expr = Sub(to_expr(10), to_expr(4))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == 6


def test_div_constant_folding():
    expr = Div(to_expr(20), to_expr(5))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == 4


def test_neg_constant_folding():
    expr = Neg(to_expr(8))
    result = expr.canonicalize()
    assert isinstance(result, Constant)
    assert result.value == -8


def test_concat_and_index_recurse():
    # Concat should simply rebuild with canonical children
    x = to_expr([1, 2])
    y = to_expr([3, 4])
    concat = Concat(x, y)
    result = concat.canonicalize()
    assert isinstance(result, Concat)
    # both children are still Constant
    assert all(isinstance(c, Constant) for c in result.exprs)

    # Index should also rebuild
    idx = Index(to_expr([5, 6, 7]), 1)
    result = idx.canonicalize()
    assert isinstance(result, Index)
    assert result.index == 1
    assert isinstance(result.base, Constant)


def test_constraint_recursion_and_type():
    # test an inequality and equality on two equal constants 3+3 == 6
    lhs = Add(to_expr(3), to_expr(3))  # will fold to Constant(6)
    rhs = to_expr(5)
    ineq = lhs <= rhs
    eq = lhs == rhs

    ineq_c = ineq.canonicalize()
    eq_c = eq.canonicalize()

    assert isinstance(ineq_c, Inequality)
    assert isinstance(ineq_c.lhs, Constant) and ineq_c.lhs.value == 1
    assert isinstance(ineq_c.rhs, Constant) and ineq_c.rhs.value == 0

    assert isinstance(eq_c, Equality)
    assert isinstance(eq_c.lhs, Constant) and eq_c.lhs.value == 1
    assert isinstance(eq_c.rhs, Constant) and eq_c.rhs.value == 0


def test_constants_are_unchanged_by_canonicalization():
    """Test that constants are already normalized and unchanged by canonicalization"""

    # Constants are now normalized at construction time, so canonicalization should be a no-op
    const_scalar = Constant(5.0)
    const_vector = Constant([1.0, 2.0, 3.0])
    const_matrix = Constant([[1.0, 2.0], [3.0, 4.0]])

    # Canonicalization should return the same object (no changes needed)
    canon_scalar = const_scalar.canonicalize()
    canon_vector = const_vector.canonicalize()
    canon_matrix = const_matrix.canonicalize()

    assert canon_scalar is const_scalar  # Should be same object
    assert canon_vector is const_vector
    assert canon_matrix is const_matrix

    # Values should be already normalized
    assert const_scalar.value.shape == ()
    assert const_vector.value.shape == (3,)
    assert const_matrix.value.shape == (2, 2)


def test_vector_constraint_equivalence_after_canonicalization():
    """Test that different ways of creating vector constraints become equivalent after
    canonicalization
    """

    x = State("x", shape=(3,))
    bounds = np.array([1.0, 2.0, 3.0])

    # Two ways to create the same constraint - constants are normalized at construction now
    constraint1 = x <= Constant(bounds)
    constraint2 = x <= Constant(np.array([bounds]))  # Extra dimension gets squeezed at construction

    # Constants should already be equivalent at construction time
    assert np.array_equal(constraint1.rhs.value, constraint2.rhs.value)
    assert constraint1.rhs.value.shape == constraint2.rhs.value.shape

    canon1 = constraint1.canonicalize()
    canon2 = constraint2.canonicalize()

    # After canonicalization, they should remain equivalent (and in canonical form)
    assert isinstance(canon1.rhs, Constant)
    assert isinstance(canon2.rhs, Constant)
    assert np.array_equal(canon1.rhs.value, canon2.rhs.value)
    assert canon1.rhs.value.shape == canon2.rhs.value.shape


def test_inequality_preserves_convex_flag():
    """Test that canonicalization preserves the is_convex flag for Inequality constraints"""
    x = State("x", shape=(3,))

    # Create a regular (non-convex) inequality constraint
    constraint_nonconvex = x <= Constant([1, 2, 3])
    assert constraint_nonconvex.is_convex is False

    # Create a convex inequality constraint
    constraint_convex = (x <= Constant([1, 2, 3])).convex()
    assert constraint_convex.is_convex is True

    # Canonicalize both
    canon_nonconvex = constraint_nonconvex.canonicalize()
    canon_convex = constraint_convex.canonicalize()

    # Check that convex flags are preserved
    assert canon_nonconvex.is_convex is False
    assert canon_convex.is_convex is True

    # Verify they're still Inequality objects
    assert isinstance(canon_nonconvex, Inequality)
    assert isinstance(canon_convex, Inequality)


def test_equality_preserves_convex_flag():
    """Test that canonicalization preserves the is_convex flag for Equality constraints"""
    x = State("x", shape=(3,))

    # Create a regular (non-convex) equality constraint
    constraint_nonconvex = x == Constant([1, 2, 3])
    assert constraint_nonconvex.is_convex is False

    # Create a convex equality constraint
    constraint_convex = (x == Constant([1, 2, 3])).convex()
    assert constraint_convex.is_convex is True

    # Canonicalize both
    canon_nonconvex = constraint_nonconvex.canonicalize()
    canon_convex = constraint_convex.canonicalize()

    # Check that convex flags are preserved
    assert canon_nonconvex.is_convex is False
    assert canon_convex.is_convex is True

    # Verify they're still Equality objects
    assert isinstance(canon_nonconvex, Equality)
    assert isinstance(canon_convex, Equality)


def test_nodal_constraint_preserves_inner_convex_flag():
    """Test that canonicalization preserves the is_convex flag for constraints wrapped in
    NodalConstraint"""
    x = State("x", shape=(3,))

    # Create a convex constraint and wrap it in NodalConstraint
    base_constraint = (x <= Constant([1, 2, 3])).convex()
    assert base_constraint.is_convex is True

    nodal_constraint = base_constraint.at([0, 5, 10])
    assert isinstance(nodal_constraint, NodalConstraint)
    assert nodal_constraint.constraint.is_convex is True

    # Canonicalize the nodal constraint
    canon_nodal = nodal_constraint.canonicalize()

    # Check that the inner constraint's convex flag is preserved
    assert isinstance(canon_nodal, NodalConstraint)
    assert canon_nodal.constraint.is_convex is True
    assert canon_nodal.nodes == [0, 5, 10]

    # The inner constraint should still be an Inequality
    assert isinstance(canon_nodal.constraint, Inequality)


def test_mixed_convex_and_nonconvex_constraints():
    """Test canonicalization with a mix of convex and non-convex constraints"""
    x = State("x", shape=(2,))

    # Create various constraints
    constraint1 = x <= Constant([1, 2])  # non-convex
    constraint2 = (x >= Constant([0, 0])).convex()  # convex inequality
    constraint3 = (x == Constant([5, 6])).convex()  # convex equality
    constraint4 = x == Constant([3, 4])  # non-convex equality

    constraints = [constraint1, constraint2, constraint3, constraint4]
    expected_convex = [False, True, True, False]

    # Canonicalize all constraints
    canonical_constraints = [c.canonicalize() for c in constraints]

    # Verify convex flags are preserved
    for canon_c, expected in zip(canonical_constraints, expected_convex):
        assert canon_c.is_convex == expected

    # Verify types are preserved
    assert isinstance(canonical_constraints[0], Inequality)
    assert isinstance(canonical_constraints[1], Inequality)
    assert isinstance(canonical_constraints[2], Equality)
    assert isinstance(canonical_constraints[3], Equality)


def test_mul_preserves_vector_structure_with_parameter():
    """Test that multiplying a Parameter by a vector preserves the vector structure.

    Regression test for bug where Parameter * vector was incorrectly reduced to
    Parameter * scalar during canonicalization.
    """
    # Create a parameter
    g = Parameter("g", value=3.7114)

    # Create a vector
    g_vec = np.array([0.0, 0.0, 1.0])

    # Multiply parameter by vector
    expr = g * g_vec

    # Canonicalize
    result = expr.canonicalize()

    # The result should still be a Mul expression
    assert isinstance(result, Mul)

    # It should contain the parameter and a Constant with the FULL vector
    has_param = False
    has_vector_const = False

    for factor in result.factors:
        if isinstance(factor, Parameter) and factor.name == "g":
            has_param = True
        if isinstance(factor, Constant):
            # The constant should be a vector [0, 0, 1], NOT a scalar 0
            assert factor.value.shape == (3,), (
                f"Expected vector shape (3,), got {factor.value.shape}"
            )
            assert np.allclose(factor.value, [0.0, 0.0, 1.0]), (
                f"Expected [0, 0, 1], got {factor.value}"
            )
            has_vector_const = True

    assert has_param, "Parameter 'g' should be in the canonicalized expression"
    assert has_vector_const, "Vector constant should be preserved in canonicalized expression"


def test_mul_vector_constant_folding():
    """Test that multiplying multiple vector constants correctly performs element-wise multiplication."""
    # Create vector constants
    vec1 = Constant(np.array([2.0, 3.0, 4.0]))
    vec2 = Constant(np.array([5.0, 6.0, 7.0]))

    # Multiply them
    expr = Mul(vec1, vec2)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single constant with element-wise product
    assert isinstance(result, Constant)
    expected = np.array([10.0, 18.0, 28.0])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_matrix_constant_folding():
    """Test that multiplying matrix constants correctly performs element-wise multiplication."""
    # Create matrix constants
    mat1 = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    mat2 = Constant(np.array([[2.0, 3.0], [4.0, 5.0]]))

    # Multiply them
    expr = Mul(mat1, mat2)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single constant with element-wise product
    assert isinstance(result, Constant)
    expected = np.array([[2.0, 6.0], [12.0, 20.0]])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_scalar_times_vector():
    """Test that multiplying a scalar constant by a vector constant works correctly."""
    scalar = Constant(2.0)
    vector = Constant(np.array([1.0, 2.0, 3.0]))

    # Multiply
    expr = Mul(scalar, vector)

    # Canonicalize
    result = expr.canonicalize()

    # Should fold to a single vector constant
    assert isinstance(result, Constant)
    expected = np.array([2.0, 4.0, 6.0])
    assert np.allclose(result.value, expected), f"Expected {expected}, got {result.value}"


def test_mul_multiple_constants_with_parameter():
    """Test that multiplying multiple constants with a parameter preserves structure correctly."""
    # This mimics the real-world case: g_vec * g where g_vec = [0, 0, 1] * g_scalar
    g = Parameter("g", value=3.7114)
    base_vec = np.array([0.0, 0.0, 1.0])
    scalar = 9.807

    # Create expression: Parameter * vector * scalar
    expr = Mul(g, Constant(base_vec), Constant(scalar))

    # Canonicalize
    result = expr.canonicalize()

    # Should have Parameter and a single folded constant (vector * scalar)
    assert isinstance(result, Mul)

    has_param = False
    has_const = False

    for factor in result.factors:
        if isinstance(factor, Parameter):
            has_param = True
        if isinstance(factor, Constant):
            has_const = True
            # The two constants should be multiplied: [0, 0, 1] * 9.807 = [0, 0, 9.807]
            expected = np.array([0.0, 0.0, 9.807])
            assert factor.value.shape == (3,), (
                f"Expected vector shape (3,), got {factor.value.shape}"
            )
            assert np.allclose(factor.value, expected), f"Expected {expected}, got {factor.value}"

    assert has_param, "Parameter should be preserved"
    assert has_const, "Folded constant should be present"


def test_mul_gravity_vector_case():
    """Regression test for the exact gravity vector bug case.

    This tests: g_vec = np.array([0, 0, 1]) * Parameter('g', value=3.7114)
    which should canonicalize to Parameter('g') * Const([0, 0, 1])
    NOT Parameter('g') * Const(0.0)
    """
    g = Parameter("g", value=3.7114)
    g_vec_array = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # This is how it appears in the dynamics
    g_vec = g_vec_array * g

    # Canonicalize
    g_vec_canon = g_vec.canonicalize()

    # Should be Mul(Parameter('g'), Const([0, 0, 1]))
    assert isinstance(g_vec_canon, Mul), "Should be a Mul expression"

    # Extract factors
    param_factor = None
    const_factor = None

    for factor in g_vec_canon.factors:
        if isinstance(factor, Parameter):
            param_factor = factor
        elif isinstance(factor, Constant):
            const_factor = factor

    assert param_factor is not None, "Should have a Parameter factor"
    assert param_factor.name == "g", "Parameter should be 'g'"

    assert const_factor is not None, "Should have a Constant factor"
    assert const_factor.value.shape == (3,), (
        f"Constant should be a 3D vector, got shape {const_factor.value.shape}"
    )
    assert np.allclose(const_factor.value, [0.0, 0.0, 1.0]), (
        f"Expected [0, 0, 1], got {const_factor.value}"
    )

    # The bug would have produced Const(0.0) because np.prod([0, 0, 1]) = 0
    # Verify this doesn't happen
    assert not (const_factor.value.shape == () and const_factor.value == 0.0), (
        "Bug detected: vector was reduced to scalar 0!"
    )


def test_mul_broadcasting_with_parameters():
    """Test that broadcasting works correctly with parameters and arrays of different shapes."""
    param = Parameter("alpha", value=2.0)

    # Scalar * vector
    vec = Constant(np.array([1.0, 2.0, 3.0]))
    expr1 = Mul(param, vec)
    result1 = expr1.canonicalize()
    assert isinstance(result1, Mul)

    # Scalar * matrix
    mat = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expr2 = Mul(param, mat)
    result2 = expr2.canonicalize()
    assert isinstance(result2, Mul)

    # Both should preserve the parameter and the array structure
    for result in [result1, result2]:
        has_param = any(isinstance(f, Parameter) for f in result.factors)
        assert has_param, "Parameter should be preserved in canonicalized expression"
