import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Constant,
    Div,
    Equality,
    Inequality,
    MatMul,
    Mul,
    Neg,
    Sub,
    to_expr,
    traverse,
)
from openscvx.backend.state import State
from openscvx.backend.variable import Variable


def test_to_expr_wraps_numbers_and_arrays():
    # scalars
    c1 = to_expr(5)
    assert isinstance(c1, Constant)
    assert c1.value.shape == () and c1.value == 5

    # 1-D arrays become Constant
    arr = [1, 2, 3]
    c2 = to_expr(arr)
    assert isinstance(c2, Constant)
    assert np.array_equal(c2.value, np.array(arr))

    # passing through an Expr unchanged
    a = Constant(np.array([1.0, 2.0]))
    assert to_expr(a) is a


def test_to_expr_passes_variables_through():
    v = Variable("v", (1,))
    x = State("x", (1,))
    u = Control("u", (1,))
    assert to_expr(v) is v
    assert to_expr(x) is x
    assert to_expr(u) is u


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


def test_equality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x == np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Equality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') == Const(array([0., 1., 2.]))"


def test_inequality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x <= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') <= Const(array([0., 1., 2.]))"


def test_inequality_reverse_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x >= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert rhs is x
    assert isinstance(lhs, Constant)
    assert repr(c) == "Const(array([0., 1., 2.])) <= Var('x')"


def test_pretty_print_tree_structure():
    # build a nested tree: -( (a + b) * c )
    a, b, c = Constant(1), Constant(2), Constant(3)
    tree = -((a + b) * c)
    p = tree.pretty()
    # Should indent like:
    # Neg
    #   Mul
    #     Add
    #       Const
    #       Const
    #     Const
    lines = p.splitlines()
    assert lines[0].strip() == "Neg"
    assert lines[1].strip() == "Mul"
    # deeper indent for Add's children:
    assert "Add" in lines[2]
    assert "Const" in lines[3]  # one of the leaves


def test_traverse_visits_all_nodes_in_preorder():
    # build a small graph: (a + (b * c))
    a, b, c = Constant(1), Constant(2), Constant(3)
    expr = Add(a, Mul(b, c))
    visited = []

    def visit(node):
        visited.append(type(node).__name__)

    traverse(expr, visit)

    # preorder: Add → a → Mul → b → c
    assert visited == ["Add", "Constant", "Mul", "Constant", "Constant"]


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
