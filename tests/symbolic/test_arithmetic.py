"""Tests for arithmetic operation nodes.

This module tests arithmetic operation nodes: Add, Sub, Mul, Div, MatMul, Neg, Power.
Tests cover:
- Node creation and tree structure
- Operator overloading (__add__, __sub__, etc.)
- Shape inference and broadcasting
- Lowering to JAX
- Lowering to CVXPY
- Canonicalization patterns
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Div,
    Inequality,
    MatMul,
    Mul,
    Neg,
    Power,
    Sub,
    Variable,
)

# =============================================================================
# Node Creation and Tree Structure
# =============================================================================


def test_basic_arithmetic_nodes_and_children_repr():
    a, b = Constant(2), Constant(3)
    add = a + b
    sub = a - b
    mul = a * b
    div = a / b
    neg = -a

    # types
    assert isinstance(add, Add)
    assert isinstance(sub, Sub)
    assert isinstance(mul, Mul)
    assert isinstance(div, Div)
    assert isinstance(neg, Neg)

    # children
    assert add.children() == [a, b]
    assert sub.children() == [a, b]
    assert mul.children() == [a, b]
    assert div.children() == [a, b]
    assert neg.children() == [a]

    # repr should nest correctly
    assert repr(add) == "(Const(2) + Const(3))"
    assert repr(sub) == "(Const(2) - Const(3))"
    assert repr(mul) == "(Const(2) * Const(3))"
    assert repr(div) == "(Const(2) / Const(3))"
    assert repr(neg) == "(-Const(2))"


def test_power_operator_and_node():
    """Test power operation using ** operator and Power node."""
    a, b = Constant(2), Constant(3)

    # Test ** operator
    pow1 = a**b
    assert isinstance(pow1, Power)
    assert pow1.children() == [a, b]
    assert repr(pow1) == "(Const(2))**(Const(3))"

    # Test direct Power node creation
    pow2 = Power(a, b)
    assert isinstance(pow2, Power)
    assert pow2.children() == [a, b]
    assert repr(pow2) == "(Const(2))**(Const(3))"


def test_power_with_mixed_types():
    """Test power operation with mixed numeric and expression types."""
    x = Variable("x", shape=(1,))

    # Expression ** numeric
    pow1 = x**2
    assert isinstance(pow1, Power)
    assert pow1.base is x
    assert isinstance(pow1.exponent, Constant)
    assert pow1.exponent.value == 2
    assert repr(pow1) == "(Var('x'))**(Const(2))"

    # Numeric ** expression (rpow)
    pow2 = 10**x
    assert isinstance(pow2, Power)
    assert isinstance(pow2.base, Constant)
    assert pow2.base.value == 10
    assert pow2.exponent is x
    assert repr(pow2) == "(Const(10))**(Var('x'))"


def test_matmul_vector_and_matrix():
    # 2×2 identity matrix × 2-vector
    M = Constant(np.eye(2))
    v = Constant(np.array([1.0, 2.0]))
    mm = M @ v

    assert isinstance(mm, MatMul)
    children = mm.children()
    assert children[0] is M and children[1] is v

    # repr should reflect operator
    assert "MatMul" in mm.pretty()  # tree form contains the node name
    assert "(" in repr(mm) and "@" not in repr(mm)  # repr is Python‐safe


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2,), (2,)),  # vector + vector
        ((2, 2), (2, 2)),  # matrix + matrix
    ],
)
def test_elementwise_addition_children_for_arrays(shape_a, shape_b):
    A = Constant(np.ones(shape_a))
    B = Constant(np.full(shape_b, 2.0))
    expr = A + B

    # children captured correctly
    left, right = expr.children()
    assert left is A and right is B

    # repr mentions both shapes
    rep = repr(expr)
    assert "Const" in rep


def test_combined_ops_produce_correct_constraint_tree():
    # (x + y) @ z >= 5
    x = Variable("x", (3,))
    y = Variable("y", (3,))
    z = Variable("z", (3,))

    # note: MatMul between two 3-vectors is allowed at AST level
    expr = (x + y) @ z <= 5
    # root is Constraint
    assert isinstance(expr, Inequality)
    # check tree structure via pretty()
    p = expr.pretty().splitlines()
    assert p[0].strip().startswith("Inequality")
    # next line is MatMul
    assert "MatMul" in p[1]

    # children of the constraint:
    assert isinstance(expr.lhs, MatMul)
    assert isinstance(expr.rhs, Constant)


def test_add_mul_accept_many_terms():
    a, b, c, d = Constant(5), Constant(3), Constant(1), Constant(2)
    add = Add(a, b, c, d)
    mul = Mul(a, b, c, d)

    assert add.children() == [a, b, c, d]
    assert mul.children() == [a, b, c, d]

    assert repr(add) == "(Const(5) + Const(3) + Const(1) + Const(2))"
    assert repr(mul) == "(Const(5) * Const(3) * Const(1) * Const(2))"


def test_add_mul_requires_at_least_two_terms():
    with pytest.raises(ValueError):
        Add(Constant(1))
    with pytest.raises(ValueError):
        Mul(Constant(2))
