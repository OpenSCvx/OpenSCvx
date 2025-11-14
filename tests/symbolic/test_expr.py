import numpy as np
import pytest

from openscvx.symbolic.expr import (
    CTCS,
    Add,
    Constant,
    Control,
    Div,
    Equality,
    Inequality,
    MatMul,
    Mul,
    Neg,
    NodalConstraint,
    Parameter,
    Power,
    State,
    Sub,
    Sum,
    Variable,
    ctcs,
    to_expr,
    traverse,
)


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


def test_equality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x == np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Equality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') == Const([0.0, 1.0, 2.0])"


def test_inequality_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x <= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert lhs is x
    assert isinstance(rhs, Constant)
    assert repr(c) == "Var('x') <= Const([0.0, 1.0, 2.0])"


def test_inequality_reverse_creation_and_children():
    x = Variable("x", shape=(3,))
    # build lhs ≤ rhs
    c = x >= np.array([0.0, 1.0, 2.0])

    assert isinstance(c, Inequality)
    # children are exactly the Expr on each side
    lhs, rhs = c.children()
    assert rhs is x
    assert isinstance(lhs, Constant)
    assert repr(c) == "Const([0.0, 1.0, 2.0]) <= Var('x')"


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


def test_sum_node_creation_and_children():
    """Test Sum node creation and tree structure."""
    from openscvx.symbolic.expr import Sum

    x = Variable("x", shape=(3,))
    sum_expr = Sum(x)

    assert isinstance(sum_expr, Sum)
    assert sum_expr.children() == [x]
    assert repr(sum_expr) == "sum(Var('x'))"


def test_sum_wraps_constants_and_expressions():
    """Test Sum node with various input types."""
    from openscvx.symbolic.expr import Sum

    # Sum of a constant array
    arr = np.array([1.0, 2.0, 3.0])
    sum1 = Sum(arr)
    assert isinstance(sum1.operand, Constant)
    assert np.array_equal(sum1.operand.value, arr)
    assert repr(sum1) == "sum(Const([1.0, 2.0, 3.0]))"

    # Sum of an arithmetic expression
    x = Variable("x", shape=(2,))
    y = Variable("y", shape=(2,))
    sum2 = Sum(x + y)
    assert isinstance(sum2.operand, Add)
    assert len(sum2.operand.children()) == 2
    assert repr(sum2) == "sum((Var('x') + Var('y')))"


# TODO: (norrisg) should be moved to separate ctcs testing file


def test_ctcs_wraps_constraint():
    """CTCS should wrap a Constraint object."""
    x = Variable("x", shape=(3,))
    constraint = x <= 1.0

    ctcs_constraint = CTCS(constraint)

    assert isinstance(ctcs_constraint, CTCS)
    assert ctcs_constraint.constraint is constraint
    assert ctcs_constraint.penalty == "squared_relu"  # default


def test_ctcs_requires_constraint():
    """CTCS should only accept Constraint objects."""
    x = Variable("x", shape=(3,))
    not_a_constraint = x + 1.0

    with pytest.raises(TypeError, match="CTCS must wrap a Constraint"):
        CTCS(not_a_constraint)


def test_ctcs_with_different_penalties():
    """CTCS should accept different penalty types."""
    x = Variable("x", shape=(3,))
    constraint = x >= 0.0

    ctcs_squared = CTCS(constraint, penalty="squared_relu")
    ctcs_huber = CTCS(constraint, penalty="huber")
    ctcs_smooth = CTCS(constraint, penalty="smooth_relu")

    assert ctcs_squared.penalty == "squared_relu"
    assert ctcs_huber.penalty == "huber"
    assert ctcs_smooth.penalty == "smooth_relu"


def test_ctcs_helper_function():
    """The ctcs() helper should create CTCS objects."""
    x = Variable("x", shape=(2,))
    constraint = x == np.array([1.0, 2.0])

    # Default penalty
    ctcs1 = ctcs(constraint)
    assert isinstance(ctcs1, CTCS)
    assert ctcs1.constraint is constraint
    assert ctcs1.penalty == "squared_relu"

    # Custom penalty
    ctcs2 = ctcs(constraint, penalty="huber")
    assert ctcs2.penalty == "huber"


def test_ctcs_children():
    """CTCS should return its constraint as its only child."""
    x = Variable("x", shape=(3,))
    constraint = x <= 5.0
    ctcs_constraint = CTCS(constraint)

    children = ctcs_constraint.children()
    assert len(children) == 1
    assert children[0] is constraint


def test_ctcs_repr():
    """CTCS should have a readable representation."""
    x = Variable("x", shape=(3,))
    constraint = x <= 1.5

    ctcs_default = CTCS(constraint)
    assert repr(ctcs_default) == "CTCS(Var('x') <= Const(1.5), penalty='squared_relu')"

    ctcs_huber = CTCS(constraint, penalty="huber")
    assert repr(ctcs_huber) == "CTCS(Var('x') <= Const(1.5), penalty='huber')"


def test_ctcs_traversal():
    """CTCS should be traversable like other expressions."""
    x = Variable("x", shape=(2,))
    y = Variable("y", shape=(2,))

    # Create a CTCS constraint with some arithmetic
    constraint = (x + y) <= 10.0
    ctcs_constraint = CTCS(constraint)

    visited = []

    def visit(node):
        visited.append(type(node).__name__)

    traverse(ctcs_constraint, visit)

    # Should visit CTCS -> Inequality -> Add -> Variable -> Variable -> Constant
    assert visited[0] == "CTCS"
    assert visited[1] == "Inequality"
    assert visited[2] == "Add"
    assert "Variable" in visited
    assert "Constant" in visited


def test_ctcs_with_equality_constraint():
    """CTCS should work with Equality constraints."""
    x = Variable("x", shape=(3,))
    constraint = x == np.zeros(3)

    ctcs_constraint = ctcs(constraint, penalty="smooth_relu")

    assert isinstance(ctcs_constraint.constraint, Equality)
    assert ctcs_constraint.penalty == "smooth_relu"


def test_multiple_ctcs_constraints():
    """Should be able to create multiple CTCS constraints."""
    x = Variable("x", shape=(2,))
    u = Variable("u", shape=(1,))

    # Different constraints with different penalties
    c1 = ctcs(x <= 1.0, penalty="squared_relu")
    c2 = ctcs(x >= -1.0, penalty="huber")
    c3 = ctcs(u == 0.0, penalty="smooth_relu")

    assert c1.penalty == "squared_relu"
    assert c2.penalty == "huber"
    assert c3.penalty == "smooth_relu"

    # Verify they wrap different constraints
    assert isinstance(c1.constraint, Inequality)
    assert isinstance(c2.constraint, Inequality)
    assert isinstance(c3.constraint, Equality)


def test_ctcs_pretty_print():
    """CTCS should integrate with pretty printing."""
    x = Variable("x", shape=(2,))
    constraint = x <= 5.0
    ctcs_constraint = CTCS(constraint)

    pretty = ctcs_constraint.pretty()
    lines = pretty.splitlines()

    assert lines[0].strip() == "CTCS"
    assert "Inequality" in lines[1]
    # Should show the tree structure
    assert "Variable" in pretty
    assert "Constant" in pretty


def test_penalty_expressions():
    """Test the penalty expression building blocks."""
    from openscvx.symbolic.expr import Huber, PositivePart, SmoothReLU, Square

    x = Variable("x", shape=(1,))

    # PositivePart
    pos = PositivePart(x)
    assert repr(pos) == "pos(Var('x'))"
    assert pos.children() == [x]

    # Square
    sq = Square(x)
    assert repr(sq) == "(Var('x'))^2"
    assert sq.children() == [x]

    # Huber
    hub = Huber(x, delta=0.5)
    assert repr(hub) == "huber(Var('x'), delta=0.5)"
    assert hub.delta == 0.5
    assert hub.children() == [x]

    # SmoothReLU
    smooth = SmoothReLU(x, c=1e-6)
    assert repr(smooth) == "smooth_relu(Var('x'), c=1e-06)"
    assert smooth.c == 1e-6
    assert smooth.children() == [x]


def test_ctcs_penalty_expr_method():
    """Test building penalty expressions from CTCS constraints."""
    from openscvx.symbolic.expr import Huber, PositivePart, SmoothReLU, Square

    x = Variable("x", shape=(2,))
    constraint = x <= 1.0

    # squared_relu penalty
    ctcs1 = CTCS(constraint, penalty="squared_relu")
    penalty1 = ctcs1.penalty_expr()
    assert isinstance(penalty1, Sum)
    assert isinstance(penalty1.operand, Square)
    assert isinstance(penalty1.operand.x, PositivePart)
    assert penalty1.operand.x.x is constraint.lhs

    # huber penalty
    ctcs2 = CTCS(constraint, penalty="huber")
    penalty2 = ctcs2.penalty_expr()
    assert isinstance(penalty2.operand, Huber)
    assert isinstance(penalty2.operand.x, PositivePart)
    assert penalty2.operand.x.x is constraint.lhs

    # smooth_relu penalty
    ctcs3 = CTCS(constraint, penalty="smooth_relu")
    penalty3 = ctcs3.penalty_expr()
    assert isinstance(penalty3.operand, SmoothReLU)
    assert penalty3.operand.x is constraint.lhs


def test_ctcs_unknown_penalty():
    """CTCS should raise error for unknown penalty types."""
    x = Variable("x", shape=(1,))
    constraint = x <= 0.0

    ctcs_constraint = CTCS(constraint, penalty="unknown")

    with pytest.raises(ValueError, match="Unknown penalty"):
        ctcs_constraint.penalty_expr()


def test_nodal_constraint_convex_method_chaining():
    """Test that NodalConstraint.convex() works in both chaining orders."""
    x = Variable("x", shape=(3,))

    # Test .at().convex() chaining
    nodal1 = (x <= [1, 2, 3]).at([0, 5, 10]).convex()
    assert isinstance(nodal1, NodalConstraint)
    assert nodal1.constraint.is_convex is True
    assert nodal1.nodes == [0, 5, 10]

    # Test .convex().at() chaining
    nodal2 = (x <= [1, 2, 3]).convex().at([0, 5, 10])
    assert isinstance(nodal2, NodalConstraint)
    assert nodal2.constraint.is_convex is True
    assert nodal2.nodes == [0, 5, 10]


def test_parameter_creation():
    """Test basic Parameter node creation."""
    p1 = Parameter("mass", value=1.0)
    assert p1.name == "mass"
    assert p1.shape == ()
    assert isinstance(p1, Parameter)

    p2 = Parameter("position", shape=(3,), value=np.array([0.0, 0.0, 0.0]))
    assert p2.name == "position"
    assert p2.shape == (3,)


def test_parameter_arithmetic_operations():
    """Test Parameter in arithmetic operations."""
    p = Parameter("param", value=1.0)
    x = Variable("x", shape=())

    add_expr = p + x
    assert isinstance(add_expr, Add)
    assert p in add_expr.children()
    assert x in add_expr.children()

    mul_expr = p * 2
    assert isinstance(mul_expr, Mul)
    assert p in mul_expr.children()


def test_parameter_in_constraints():
    """Test Parameter in constraint creation."""
    p = Parameter("threshold", value=1.0)
    x = Variable("x", shape=())

    ineq = x <= p
    assert isinstance(ineq, Inequality)
    assert ineq.lhs is x
    assert ineq.rhs is p
