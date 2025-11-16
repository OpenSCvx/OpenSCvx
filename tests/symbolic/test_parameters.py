"""Tests for Parameter nodes.

This module tests the Parameter node type and its behavior throughout the symbolic
expression system, including:
- Parameter creation and properties
- Parameter usage in arithmetic expressions
- Parameter usage in constraints
- Parameter lowering to JAX
- Parameter lowering to CVXPY
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Inequality,
    Mul,
    Parameter,
    Variable,
)

# =============================================================================
# Parameter Creation and Properties
# =============================================================================


def test_parameter_creation():
    """Test basic Parameter node creation."""
    p1 = Parameter("mass", value=1.0)
    assert p1.name == "mass"
    assert p1.shape == ()
    assert isinstance(p1, Parameter)

    p2 = Parameter("position", shape=(3,), value=np.array([0.0, 0.0, 0.0]))
    assert p2.name == "position"
    assert p2.shape == (3,)


# =============================================================================
# Parameter in Expressions
# =============================================================================


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
