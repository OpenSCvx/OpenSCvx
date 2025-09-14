import numpy as np

from openscvx.backend.canonicalizer import canonicalize
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Div,
    Equality,
    Index,
    Inequality,
    Mul,
    Neg,
    Sub,
    to_expr,
)
from openscvx.backend.state import State


def test_flatten_and_fold_add():
    a = to_expr(1)
    b = to_expr(2)
    c = to_expr(3)
    nested = Add(Add(a, b), c, to_expr(4))
    result = canonicalize(nested)
    # should be Add(1,2,3,4) then folded to Constant(10)
    assert isinstance(result, Constant)
    assert result.value == 10


def test_add_eliminate_zero_and_singleton():
    x = to_expr(5)
    zero = to_expr(0)
    # 5 + 0 + 0 ⇒ 5
    expr = Add(zero, x, zero)
    result = canonicalize(expr)
    assert not isinstance(result, Add)
    assert isinstance(result, Constant)
    assert result.value == 5


def test_flatten_and_fold_mul():
    a = to_expr(2)
    b = to_expr(3)
    c = to_expr(4)
    nested = Mul(Mul(a, b), c, to_expr(5))
    result = canonicalize(nested)
    # 2*3*4*5 = 120
    assert isinstance(result, Constant)
    assert result.value == 120


def test_mul_eliminate_one_and_singleton():
    x = to_expr(7)
    one = to_expr(1)
    expr = Mul(one, x, one)
    result = canonicalize(expr)
    assert not isinstance(result, Mul)
    assert isinstance(result, Constant)
    assert result.value == 7


def test_sub_constant_folding():
    expr = Sub(to_expr(10), to_expr(4))
    result = canonicalize(expr)
    assert isinstance(result, Constant)
    assert result.value == 6


def test_div_constant_folding():
    expr = Div(to_expr(20), to_expr(5))
    result = canonicalize(expr)
    assert isinstance(result, Constant)
    assert result.value == 4


def test_neg_constant_folding():
    expr = Neg(to_expr(8))
    result = canonicalize(expr)
    assert isinstance(result, Constant)
    assert result.value == -8


def test_concat_and_index_recurse():
    # Concat should simply rebuild with canonical children
    x = to_expr([1, 2])
    y = to_expr([3, 4])
    concat = Concat(x, y)
    result = canonicalize(concat)
    assert isinstance(result, Concat)
    # both children are still Constant
    assert all(isinstance(c, Constant) for c in result.exprs)

    # Index should also rebuild
    idx = Index(to_expr([5, 6, 7]), 1)
    result = canonicalize(idx)
    assert isinstance(result, Index)
    assert result.index == 1
    assert isinstance(result.base, Constant)


def test_constraint_recursion_and_type():
    # test an inequality and equality on two equal constants 3+3 == 6
    lhs = Add(to_expr(3), to_expr(3))  # will fold to Constant(6)
    rhs = to_expr(5)
    ineq = lhs <= rhs
    eq = lhs == rhs

    ineq_c = canonicalize(ineq)
    eq_c = canonicalize(eq)

    assert isinstance(ineq_c, Inequality)
    assert isinstance(ineq_c.lhs, Constant) and ineq_c.lhs.value == 1
    assert isinstance(ineq_c.rhs, Constant) and ineq_c.rhs.value == 0

    assert isinstance(eq_c, Equality)
    assert isinstance(eq_c.lhs, Constant) and eq_c.lhs.value == 1
    assert isinstance(eq_c.rhs, Constant) and eq_c.rhs.value == 0


def test_constants_are_unchanged_by_canonicalization():
    """Test that constants are already normalized and unchanged by canonicalization"""
    import numpy as np

    # Constants are now normalized at construction time, so canonicalization should be a no-op
    const_scalar = Constant(5.0)
    const_vector = Constant([1.0, 2.0, 3.0])
    const_matrix = Constant([[1.0, 2.0], [3.0, 4.0]])

    # Canonicalization should return the same object (no changes needed)
    canon_scalar = canonicalize(const_scalar)
    canon_vector = canonicalize(const_vector)
    canon_matrix = canonicalize(const_matrix)

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

    canon1 = canonicalize(constraint1)
    canon2 = canonicalize(constraint2)

    # After canonicalization, they should remain equivalent (and in canonical form)
    assert isinstance(canon1.rhs, Constant)
    assert isinstance(canon2.rhs, Constant)
    assert np.array_equal(canon1.rhs.value, canon2.rhs.value)
    assert canon1.rhs.value.shape == canon2.rhs.value.shape
