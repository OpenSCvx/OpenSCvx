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


# =============================================================================
# JAX Lowering Tests
# =============================================================================


def test_diag():
    """Test the Diag compact node individually."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Diag, State
    from openscvx.symbolic.lower import lower_to_jax

    # Test with different vectors
    test_vectors = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([0.5, -1.0, 2.5]),
        jnp.array([0.0, 0.0, 0.0]),
    ]

    for v_val in test_vectors:
        # Create vector state
        v = State("v", (3,))
        v._slice = slice(0, 3)

        # Test Diag node
        diag_expr = Diag(v)
        fn = lower_to_jax(diag_expr)
        result = fn(v_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be diagonal
        expected = jnp.diag(v_val)
        assert jnp.allclose(result, expected, atol=1e-12)

        # Off-diagonal elements should be zero
        off_diag_mask = ~jnp.eye(3, dtype=bool)
        assert jnp.allclose(result[off_diag_mask], 0.0, atol=1e-12)


# =============================================================================
# CVXPY Lowering Tests
# =============================================================================


def test_cvxpy_sum():
    """Test sum operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Sum(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_l2():
    """Test L2 norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord=2)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_l1():
    """Test L1 norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord=1)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_inf():
    """Test infinity norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord="inf")

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_fro():
    """Test Frobenius norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((2, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(6,))  # Flattened 2x3 matrix
    expr = Norm(x, ord="fro")

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)
