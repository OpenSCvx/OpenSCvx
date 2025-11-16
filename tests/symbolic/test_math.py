"""Tests for mathematical function nodes.

This module tests mathematical function nodes:
- Trigonometric: Sin, Cos
- Exponential: Exp, Log, Sqrt
- Nonlinear: Square, PositivePart, Huber, SmoothReLU, Max

Tests cover:
- Node creation and properties
- Behavior with constants and variables
- Lowering to JAX
- Lowering to CVXPY (where applicable)
- Special properties (differentiability, convexity)
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Huber,
    PositivePart,
    SmoothReLU,
    Square,
    Variable,
)

# =============================================================================
# Penalty/Nonlinear Function Node Creation
# =============================================================================


def test_penalty_expressions():
    """Test the penalty expression building blocks."""
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


# =============================================================================
# JAX Lowering Tests - Penalty Functions
# =============================================================================


def test_positive_part_constant():
    """Test PositivePart with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    # Test with mixed positive and negative values
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = PositivePart(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert jnp.allclose(result, expected)


def test_positive_part_state():
    """Test PositivePart with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-3.0, -1.0, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = PositivePart(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected)


def test_positive_part_expression():
    """Test PositivePart with a composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, State, Sub
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 2.0, 3.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)
    threshold = Constant(np.array([2.0, 2.0, 2.0]))

    # pos(x - 2)
    expr = PositivePart(Sub(state, threshold))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.array([0.0, 0.0, 1.0])  # max(x - 2, 0)
    assert jnp.allclose(result, expected)


def test_square_constant():
    """Test Square with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = Square(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = values**2
    assert jnp.allclose(result, expected)


def test_square_state():
    """Test Square with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-3.0, -1.5, 0.0, 1.5, 3.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = Square(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = x * x
    assert jnp.allclose(result, expected)


def test_squared_relu_pattern():
    """Test the squared ReLU pattern: (max(x, 0))^2."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Build squared ReLU: Square(PositivePart(x))
    expr = Square(PositivePart(state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Expected: [0, 0, 0, 1, 4]
    expected = jnp.maximum(x, 0.0) ** 2
    assert jnp.allclose(result, expected)


def test_huber_constant():
    """Test Huber penalty with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    delta = 0.25

    expr = Huber(Constant(values), delta=delta)

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # Huber formula:
    # if |x| <= delta: 0.5 * x^2
    # else: delta * (|x| - 0.5 * delta)
    expected = jnp.where(
        jnp.abs(values) <= delta, 0.5 * values**2, delta * (jnp.abs(values) - 0.5 * delta)
    )
    assert jnp.allclose(result, expected)


def test_huber_state_various_deltas():
    """Test Huber penalty with different delta values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Test with small delta
    expr_small = Huber(state, delta=0.1)
    fn_small = lower_to_jax(expr_small)
    result_small = fn_small(x, None, None, None)

    # Most values should be in the linear region
    expected_small = jnp.where(jnp.abs(x) <= 0.1, 0.5 * x**2, 0.1 * (jnp.abs(x) - 0.5 * 0.1))
    assert jnp.allclose(result_small, expected_small)

    # Test with large delta
    expr_large = Huber(state, delta=3.0)
    fn_large = lower_to_jax(expr_large)
    result_large = fn_large(x, None, None, None)

    # All values should be in the quadratic region
    expected_large = 0.5 * x**2  # Since all |x| <= 3.0
    assert jnp.allclose(result_large, expected_large)


def test_huber_with_positive_part():
    """Test Huber applied to positive part (common CTCS pattern)."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Huber(PositivePart(x))
    expr = Huber(PositivePart(state), delta=0.5)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # First apply positive part
    pos_x = jnp.maximum(x, 0.0)
    # Then apply Huber
    expected = jnp.where(jnp.abs(pos_x) <= 0.5, 0.5 * pos_x**2, 0.5 * (jnp.abs(pos_x) - 0.5 * 0.5))
    assert jnp.allclose(result, expected)


def test_smooth_relu_constant():
    """Test SmoothReLU with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = 1e-8

    expr = SmoothReLU(Constant(values), c=c)

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # SmoothReLU: sqrt(max(x, 0)^2 + c^2) - c
    expected = jnp.sqrt(jnp.maximum(values, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_state():
    """Test SmoothReLU with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_approaches_relu():
    """Test that SmoothReLU approaches ReLU as c → 0."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Very small c
    expr = SmoothReLU(state, c=1e-12)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Should be very close to max(x, 0)
    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected, atol=1e-10)


def test_smooth_relu_differentiability_at_zero():
    """Test that SmoothReLU is smooth at x=0."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    # Test values around zero
    x = jnp.array([-0.01, -0.001, 0.0, 0.001, 0.01])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # At x=0, should equal 0 (since sqrt(c^2) - c = 0)
    assert jnp.abs(result[2]) < 1e-10

    # Should be continuous and smooth
    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_penalty_in_constraint_expression():
    """Test penalty functions used in a constraint-like expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, State, Sub
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.5, 1.5, 2.5])
    u = jnp.array([0.0, 0.0, 0.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 2.0, violation when x > 2.0
    # Penalty: Square(PositivePart(x - 2.0))
    limit = Constant(np.array([2.0, 2.0, 2.0]))
    violation = Sub(state, limit)
    penalty = Square(PositivePart(violation))

    fn = lower_to_jax(penalty)
    result = fn(x, u, None, None)

    # Expected: [0, 0, 0.25] since only x[2]=2.5 violates
    expected = jnp.array([0.0, 0.0, 0.25])
    assert jnp.allclose(result, expected)


def test_combined_penalties():
    """Test combining different penalty functions."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Different penalties for testing
    squared_relu = Square(PositivePart(state))
    huber_penalty = Huber(PositivePart(state), delta=0.5)
    smooth_relu = SmoothReLU(state, c=0.1)

    fn_sq = lower_to_jax(squared_relu)
    fn_hub = lower_to_jax(huber_penalty)
    fn_smooth = lower_to_jax(smooth_relu)

    result_sq = fn_sq(x, None, None, None)
    result_hub = fn_hub(x, None, None, None)
    result_smooth = fn_smooth(x, None, None, None)

    # Squared ReLU should be most aggressive for large violations
    assert result_sq[4] > result_hub[4]  # At x=2.0

    # Huber should be linear for large values
    assert jnp.allclose(result_hub[4], 0.5 * (2.0 - 0.5 * 0.5))

    # All should be zero for negative values
    assert jnp.allclose(result_sq[0], 0.0)
    assert jnp.allclose(result_hub[0], 0.0)
    assert jnp.allclose(result_smooth[0], 0.0, atol=1e-8)


def test_penalty_with_control():
    """Test penalty functions with control variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Control, Sub
    from openscvx.symbolic.lower import lower_to_jax

    u = jnp.array([-1.0, 0.5, 2.0])

    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Control constraint: |u| <= 1.0
    # Penalty for upper bound: Square(PositivePart(u - 1.0))
    upper_limit = Constant(np.array([1.0, 1.0, 1.0]))
    upper_violation = Sub(control, upper_limit)
    penalty = Square(PositivePart(upper_violation))

    fn = lower_to_jax(penalty)
    result = fn(None, u, None, None)

    # Expected: [0, 0, 1] since only u[2]=2.0 violates
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(result, expected)


# =============================================================================
# JAX Lowering Tests - Exp and Log
# =============================================================================


def test_exp_constant():
    """Test Exp with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Exp
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([0.0, 1.0, -1.0, 2.0])
    expr = Exp(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.exp(values)
    assert jnp.allclose(result, expected)


def test_exp_state_and_control():
    """Test Exp with state and control variables in expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control, Exp, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 0.0, -0.5])
    u = jnp.array([0.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (1,))
    control._slice = slice(0, 1)

    # Expression: exp(x[0] + u[0])
    expr = Exp(state[0] + control[0])

    fn = lower_to_jax(expr)
    result = fn(x, u, None, None)

    # Expected: exp(1.0 + 0.5) = exp(1.5)
    expected = jnp.exp(1.5)
    assert jnp.allclose(result, expected)


def test_log_constant():
    """Test Log with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Log
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([1.0, np.e, 2.0, 0.5])
    expr = Log(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.log(values)
    assert jnp.allclose(result, expected)


def test_log_with_exp_identity():
    """Test that log(exp(x)) = x for reasonable values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Exp, Log, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, 1.0, -1.0, 2.0])

    state = State("x", (4,))
    state._slice = slice(0, 4)

    # Expression: log(exp(x))
    expr = Log(Exp(state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Should recover original values
    assert jnp.allclose(result, x, atol=1e-12)


# =============================================================================
# CVXPY Lowering Tests
# =============================================================================


def test_cvxpy_positive_part():
    """Test positive part function"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = PositivePart(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_square():
    """Test square function"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Square(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_huber():
    """Test Huber loss function"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Huber(x, delta=0.5)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_smooth_relu():
    """Test smooth ReLU function"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = SmoothReLU(x, c=1e-6)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_sin_not_implemented():
    """Test that Sin raises NotImplementedError"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Cos, Sin, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Sin(x)

    with pytest.raises(NotImplementedError, match="Trigonometric functions like Sin"):
        lowerer.lower(expr)


def test_cvxpy_cos_not_implemented():
    """Test that Cos raises NotImplementedError"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Cos, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Cos(x)

    with pytest.raises(NotImplementedError, match="Trigonometric functions like Cos"):
        lowerer.lower(expr)
