"""Tests for linear algebra operation nodes.

This module tests linear algebra operation nodes: Transpose, Diag, Sum, Norm.
Tests cover:
- Node creation and properties
- Different norm types (L1, L2, Linf, Frobenius)
- Sum reduction operations
- Lowering to JAX
- Lowering to CVXPY
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Sum,
    Variable,
)

# =============================================================================
# Sum Node Tests
# =============================================================================


def test_sum_node_creation_and_children():
    """Test Sum node creation and tree structure."""
    x = Variable("x", shape=(3,))
    sum_expr = Sum(x)

    assert isinstance(sum_expr, Sum)
    assert sum_expr.children() == [x]
    assert repr(sum_expr) == "sum(Var('x'))"


def test_sum_wraps_constants_and_expressions():
    """Test Sum node with various input types."""
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
