"""Tests for constraint nodes.

This module tests constraint node types:
- Constraint: Base constraint class
- Equality: Equality constraints (==)
- Inequality: Inequality constraints (<=, >=)
- NodalConstraint: Constraints applied at specific nodes
- CTCS: Continuous-Time Constraint Satisfaction

Tests cover:
- Constraint creation and tree structure
- Convexity flags and marking
- CTCS wrapper and penalty expressions
- Lowering to JAX
- Lowering to CVXPY
- Canonicalization
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    CTCS,
    Constant,
    Equality,
    Huber,
    Inequality,
    NodalConstraint,
    PositivePart,
    SmoothReLU,
    Square,
    Sum,
    Variable,
    ctcs,
    traverse,
)

# =============================================================================
# Basic Constraint Creation
# =============================================================================


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


# =============================================================================
# NodalConstraint Tests
# =============================================================================


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


# =============================================================================
# CTCS Wrapper Tests
# =============================================================================


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


# =============================================================================
# CTCS Penalty Expression Tests
# =============================================================================


def test_ctcs_penalty_expr_method():
    """Test building penalty expressions from CTCS constraints."""
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
