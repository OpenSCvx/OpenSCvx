"""Tests for mathematical function nodes.

This module tests mathematical function nodes:

- Trigonometric: Sin, Cos
- Exponential: Exp, Log, Sqrt
- Nonlinear: Square, PositivePart, Huber, SmoothReLU, Max
- Absolute value: Abs
- Smooth maximum: LogSumExp

Tests are organized by node/node-group, with each section containing:

1. Node creation and properties
2. Shape Checking
3. Canonicalization
4. JAX lowering tests
5. CVXPY lowering tests (where applicable)
6. Integration tests (where applicable)
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Abs,
    Huber,
    LogSumExp,
    PositivePart,
    SmoothReLU,
    Square,
    Variable,
)

# =============================================================================
# PositivePart
# =============================================================================


def test_positive_part_creation():
    """Test PositivePart node creation and properties."""
    x = Variable("x", shape=(1,))

    pos = PositivePart(x)
    assert repr(pos) == "pos(Var('x'))"
    assert pos.children() == [x]


# --- PositivePart: Shape Checking ---


def test_positive_part_shape_preserves_input():
    """Test that PositivePart preserves the shape of its input."""
    x = Variable("x", shape=(3, 4))

    pos = PositivePart(x)
    assert pos.check_shape() == (3, 4)


def test_positive_part_shape_with_scalar():
    """Test PositivePart shape with scalar input."""
    x = Variable("x", shape=())

    pos = PositivePart(x)
    assert pos.check_shape() == ()


# --- PositivePart: Canonicalization ---


def test_positive_part_canonicalize_preserves_structure():
    """Test that PositivePart canonicalization preserves structure."""

    x = Variable("x", shape=(3,))

    pos = PositivePart(x)
    canonical = pos.canonicalize()

    assert isinstance(canonical, PositivePart)
    assert canonical.x == x


def test_positive_part_canonicalize_recursively():
    """Test that PositivePart canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(3,))
    # PositivePart(x + 0) should canonicalize to PositivePart(x)
    expr = PositivePart(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, PositivePart)
    # The Add(x, 0) should be canonicalized to just x
    assert canonical.x == x


# --- PositivePart: JAX Lowering ---


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


# --- PositivePart: CVXPy Lowering ---


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


# =============================================================================
# Square
# =============================================================================


def test_square_creation():
    """Test Square node creation and properties."""
    x = Variable("x", shape=(1,))

    sq = Square(x)
    assert repr(sq) == "(Var('x'))^2"
    assert sq.children() == [x]


# --- Square: Shape Checking ---


def test_square_shape_preserves_input():
    """Test that Square preserves the shape of its input."""
    x = Variable("x", shape=(2, 3, 4))

    sq = Square(x)
    assert sq.check_shape() == (2, 3, 4)


def test_square_shape_with_vector():
    """Test Square shape with vector input."""
    x = Variable("x", shape=(5,))

    sq = Square(x)
    assert sq.check_shape() == (5,)


# --- Square: Canonicalization ---


def test_square_canonicalize_preserves_structure():
    """Test that Square canonicalization preserves structure."""
    x = Variable("x", shape=(2,))

    sq = Square(x)
    canonical = sq.canonicalize()

    assert isinstance(canonical, Square)
    assert canonical.x == x


def test_square_canonicalize_recursively():
    """Test that Square canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(2,))
    # Square(x + 0) should canonicalize to Square(x)
    expr = Square(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Square)
    assert canonical.x == x


# --- Square: JAX Lowering ---


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


# --- Square: CVXPy Lowering ---


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


# =============================================================================
# Huber
# =============================================================================


def test_huber_creation():
    """Test Huber node creation and properties."""
    x = Variable("x", shape=(1,))

    hub = Huber(x, delta=0.5)
    assert repr(hub) == "huber(Var('x'), delta=0.5)"
    assert hub.delta == 0.5
    assert hub.children() == [x]


# --- Huber: Shape Checking ---


def test_huber_shape_preserves_input():
    """Test that Huber preserves the shape of its input."""
    x = Variable("x", shape=(10,))

    hub = Huber(x, delta=0.5)
    assert hub.check_shape() == (10,)


def test_huber_shape_with_matrix():
    """Test Huber shape with matrix input."""
    x = Variable("x", shape=(4, 4))

    hub = Huber(x, delta=1.0)
    assert hub.check_shape() == (4, 4)


# --- Huber: Canonicalization ---


def test_huber_canonicalize_preserves_structure_and_delta():
    """Test that Huber canonicalization preserves structure and delta parameter."""
    x = Variable("x", shape=(3,))

    hub = Huber(x, delta=0.75)
    canonical = hub.canonicalize()

    assert isinstance(canonical, Huber)
    assert canonical.x == x
    assert canonical.delta == 0.75


def test_huber_canonicalize_recursively():
    """Test that Huber canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(3,))
    # Huber(x + 0) should canonicalize to Huber(x)
    expr = Huber(Add(x, Constant(0.0)), delta=0.5)
    canonical = expr.canonicalize()

    assert isinstance(canonical, Huber)
    assert canonical.x == x
    assert canonical.delta == 0.5


# --- Huber: JAX Lowering ---


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


# --- Huber: CVXPy Lowering ---


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


# =============================================================================
# SmoothReLU
# =============================================================================


def test_smooth_relu_creation():
    """Test SmoothReLU node creation and properties."""
    x = Variable("x", shape=(1,))

    smooth = SmoothReLU(x, c=1e-6)
    assert repr(smooth) == "smooth_relu(Var('x'), c=1e-06)"
    assert smooth.c == 1e-6
    assert smooth.children() == [x]


# --- SmoothReLU: Shape Checking ---


def test_smooth_relu_shape_preserves_input():
    """Test that SmoothReLU preserves the shape of its input."""
    x = Variable("x", shape=(7,))

    smooth = SmoothReLU(x, c=1e-6)
    assert smooth.check_shape() == (7,)


def test_smooth_relu_shape_with_3d_tensor():
    """Test SmoothReLU shape with 3D tensor input."""
    x = Variable("x", shape=(2, 3, 4))

    smooth = SmoothReLU(x, c=1e-8)
    assert smooth.check_shape() == (2, 3, 4)


# --- SmoothReLU: Canonicalization ---


def test_smooth_relu_canonicalize_preserves_structure_and_c():
    """Test that SmoothReLU canonicalization preserves structure and c parameter."""
    x = Variable("x", shape=(5,))

    smooth = SmoothReLU(x, c=1e-5)
    canonical = smooth.canonicalize()

    assert isinstance(canonical, SmoothReLU)
    assert canonical.x == x
    assert canonical.c == 1e-5


def test_smooth_relu_canonicalize_recursively():
    """Test that SmoothReLU canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(5,))
    # SmoothReLU(x + 0) should canonicalize to SmoothReLU(x)
    expr = SmoothReLU(Add(x, Constant(0.0)), c=1e-6)
    canonical = expr.canonicalize()

    assert isinstance(canonical, SmoothReLU)
    assert canonical.x == x
    assert canonical.c == 1e-6


# --- SmoothReLU: JAX Lowering ---


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


# --- SmoothReLU: CVXPy Lowering ---


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


# =============================================================================
# Exp & Log
# =============================================================================


def test_exp_creation():
    """Test Exp node creation and properties."""
    from openscvx.symbolic.expr import Exp

    x = Variable("x", shape=(1,))

    exp_x = Exp(x)
    assert repr(exp_x) == "exp(Var('x'))"
    assert exp_x.children() == [x]


def test_log_creation():
    """Test Log node creation and properties."""
    from openscvx.symbolic.expr import Log

    x = Variable("x", shape=(1,))

    log_x = Log(x)
    assert repr(log_x) == "log(Var('x'))"
    assert log_x.children() == [x]


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


# --- Exp & Log: Shape Checking ---


def test_exp_shape_preserves_input():
    """Test that Exp preserves the shape of its input."""
    from openscvx.symbolic.expr import Exp

    x = Variable("x", shape=(3, 3))

    exp_x = Exp(x)
    assert exp_x.check_shape() == (3, 3)


def test_log_shape_preserves_input():
    """Test that Log preserves the shape of its input."""
    from openscvx.symbolic.expr import Log

    x = Variable("x", shape=(5, 2))

    log_x = Log(x)
    assert log_x.check_shape() == (5, 2)


def test_exp_shape_with_scalar():
    """Test Exp shape with scalar input."""
    from openscvx.symbolic.expr import Exp

    x = Variable("x", shape=())

    exp_x = Exp(x)
    assert exp_x.check_shape() == ()


def test_log_shape_with_vector():
    """Test Log shape with vector input."""
    from openscvx.symbolic.expr import Log

    x = Variable("x", shape=(10,))

    log_x = Log(x)
    assert log_x.check_shape() == (10,)


# --- Exp & Log: Canonicalization ---


def test_exp_canonicalize_preserves_structure():
    """Test that Exp canonicalization preserves structure."""
    from openscvx.symbolic.expr import Exp

    x = Variable("x", shape=(3,))

    exp_x = Exp(x)
    canonical = exp_x.canonicalize()

    assert isinstance(canonical, Exp)
    assert canonical.operand == x


def test_log_canonicalize_preserves_structure():
    """Test that Log canonicalization preserves structure."""
    from openscvx.symbolic.expr import Log

    x = Variable("x", shape=(3,))

    log_x = Log(x)
    canonical = log_x.canonicalize()

    assert isinstance(canonical, Log)
    assert canonical.operand == x


def test_exp_canonicalize_recursively():
    """Test that Exp canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Exp

    x = Variable("x", shape=(3,))
    # Exp(x + 0) should canonicalize to Exp(x)
    expr = Exp(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Exp)
    assert canonical.operand == x


def test_log_canonicalize_recursively():
    """Test that Log canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Log

    x = Variable("x", shape=(3,))
    # Log(x + 0) should canonicalize to Log(x)
    expr = Log(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Log)
    assert canonical.operand == x


# --- Exp & Log: JAX Lowering ---


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


# --- Exp & Log: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# Abs
# =============================================================================


def test_abs_creation():
    """Test Abs node creation and properties."""
    x = Variable("x", shape=(1,))

    abs_x = Abs(x)
    assert repr(abs_x) == "abs(Var('x'))"
    assert abs_x.children() == [x]


# --- Abs: Shape Checking ---


def test_abs_shape_preserves_input():
    """Test that Abs preserves the shape of its input."""
    x = Variable("x", shape=(3, 4))

    abs_x = Abs(x)
    assert abs_x.check_shape() == (3, 4)


def test_abs_shape_with_scalar():
    """Test Abs shape with scalar input."""
    x = Variable("x", shape=())

    abs_x = Abs(x)
    assert abs_x.check_shape() == ()


def test_abs_shape_with_vector():
    """Test Abs shape with vector input."""
    x = Variable("x", shape=(10,))

    abs_x = Abs(x)
    assert abs_x.check_shape() == (10,)


# --- Abs: Canonicalization ---


def test_abs_canonicalize_preserves_structure():
    """Test that Abs canonicalization preserves structure."""
    x = Variable("x", shape=(3,))

    abs_x = Abs(x)
    canonical = abs_x.canonicalize()

    assert isinstance(canonical, Abs)
    assert canonical.operand == x


def test_abs_canonicalize_recursively():
    """Test that Abs canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(3,))
    # Abs(x + 0) should canonicalize to Abs(x)
    expr = Abs(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Abs)
    assert canonical.operand == x


# --- Abs: JAX Lowering ---


def test_abs_constant():
    """Test Abs with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = Abs(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.abs(values)
    assert jnp.allclose(result, expected)


def test_abs_state():
    """Test Abs with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-3.0, -1.5, 0.0, 1.5, 3.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = Abs(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.abs(x)
    assert jnp.allclose(result, expected)


def test_abs_expression():
    """Test Abs with a composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, State, Sub
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 2.0, 3.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)
    threshold = Constant(np.array([2.0, 2.0, 2.0]))

    # abs(x - 2)
    expr = Abs(Sub(state, threshold))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.abs(x - 2.0)
    assert jnp.allclose(result, expected)


def test_abs_matrix():
    """Test Abs with matrix values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([[-2.0, 1.0], [3.0, -4.0]])
    expr = Abs(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.abs(values)
    assert jnp.allclose(result, expected)


# --- Abs: CVXPy Lowering ---


def test_cvxpy_abs():
    """Test absolute value function."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Abs(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Sqrt
# =============================================================================


def test_sqrt_creation():
    """Test Sqrt node creation and properties."""
    from openscvx.symbolic.expr import Sqrt

    x = Variable("x", shape=(1,))

    sqrt_x = Sqrt(x)
    assert repr(sqrt_x) == "sqrt(Var('x'))"
    assert sqrt_x.children() == [x]


# --- Sqrt: Shape Checking ---


def test_sqrt_shape_preserves_input():
    """Test that Sqrt preserves the shape of its input."""
    from openscvx.symbolic.expr import Sqrt

    x = Variable("x", shape=(6,))

    sqrt_x = Sqrt(x)
    assert sqrt_x.check_shape() == (6,)


def test_sqrt_shape_with_matrix():
    """Test Sqrt shape with matrix input."""
    from openscvx.symbolic.expr import Sqrt

    x = Variable("x", shape=(3, 4))

    sqrt_x = Sqrt(x)
    assert sqrt_x.check_shape() == (3, 4)


# --- Sqrt: Canonicalization ---


def test_sqrt_canonicalize_preserves_structure():
    """Test that Sqrt canonicalization preserves structure."""
    from openscvx.symbolic.expr import Sqrt

    x = Variable("x", shape=(4,))

    sqrt_x = Sqrt(x)
    canonical = sqrt_x.canonicalize()

    assert isinstance(canonical, Sqrt)
    assert canonical.operand == x


def test_sqrt_canonicalize_recursively():
    """Test that Sqrt canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Sqrt

    x = Variable("x", shape=(4,))
    # Sqrt(x + 0) should canonicalize to Sqrt(x)
    expr = Sqrt(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Sqrt)
    assert canonical.operand == x


# --- Sqrt: JAX Lowering ---


def test_sqrt_constant():
    """Test Sqrt with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Sqrt
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([1.0, 4.0, 9.0, 16.0])
    expr = Sqrt(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.sqrt(values)
    assert jnp.allclose(result, expected)


def test_sqrt_state():
    """Test Sqrt with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Sqrt, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 2.0, 3.0, 4.0])

    state = State("s", (4,))
    state._slice = slice(0, 4)
    expr = Sqrt(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sqrt(x)
    assert jnp.allclose(result, expected)


def test_sqrt_expression():
    """Test Sqrt with a composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Sqrt, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([1.0, 4.0, 9.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)

    # sqrt(Square(x)) should give |x|, but for positive values it's just x
    expr = Sqrt(Square(state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sqrt(x * x)
    assert jnp.allclose(result, expected)


# --- Sqrt: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# Max
# =============================================================================


def test_max_creation():
    """Test Max node creation and properties."""
    from openscvx.symbolic.expr import Max

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))

    max_xy = Max(x, y)
    assert repr(max_xy) == "max(Var('x'), Var('y'))"
    assert max_xy.children() == [x, y]


def test_max_creation_with_multiple_operands():
    """Test Max node creation with more than two operands."""
    from openscvx.symbolic.expr import Constant, Max

    x = Variable("x", shape=(1,))
    y = Variable("y", shape=(1,))
    z = Constant(0.0)

    max_xyz = Max(x, y, z)
    assert len(max_xyz.children()) == 3
    assert repr(max_xyz) == "max(Var('x'), Var('y'), Const(0.0))"


def test_max_requires_at_least_two_operands():
    """Test that Max raises ValueError with fewer than two operands."""
    from openscvx.symbolic.expr import Max

    x = Variable("x", shape=(1,))

    with pytest.raises(ValueError, match="Max requires two or more operands"):
        Max(x)


# --- Max: Shape Checking ---


def test_max_shape_with_same_shapes():
    """Test Max shape when all operands have the same shape."""
    from openscvx.symbolic.expr import Max

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))
    z = Variable("z", shape=(3,))

    max_xyz = Max(x, y, z)
    assert max_xyz.check_shape() == (3,)


def test_max_shape_with_broadcasting():
    """Test Max shape with broadcasting."""
    from openscvx.symbolic.expr import Constant, Max

    x = Variable("x", shape=(3, 4))
    y = Constant(np.array([1.0, 2.0, 3.0, 4.0]))  # shape (4,)

    max_xy = Max(x, y)
    assert max_xy.check_shape() == (3, 4)


def test_max_shape_with_scalar_broadcast():
    """Test Max shape with scalar broadcasting."""
    from openscvx.symbolic.expr import Constant, Max

    x = Variable("x", shape=(5, 5))
    scalar = Constant(0.0)  # scalar

    max_x = Max(x, scalar)
    assert max_x.check_shape() == (5, 5)


def test_max_shape_incompatible_raises_error():
    """Test that Max raises ValueError for incompatible shapes."""
    from openscvx.symbolic.expr import Max

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(4,))

    max_xy = Max(x, y)
    with pytest.raises(ValueError, match="Max shapes not broadcastable"):
        max_xy.check_shape()


# --- Max: Canonicalization ---


def test_max_canonicalize_flattens_nested_max():
    """Test that Max canonicalization flattens nested Max operations."""
    from openscvx.symbolic.expr import Max

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))
    z = Variable("z", shape=(3,))

    # Max(x, Max(y, z)) should flatten to Max(x, y, z)
    nested = Max(x, Max(y, z))
    canonical = nested.canonicalize()

    assert isinstance(canonical, Max)
    assert len(canonical.operands) == 3
    assert x in canonical.operands
    assert y in canonical.operands
    assert z in canonical.operands


def test_max_canonicalize_folds_constants():
    """Test that Max canonicalization folds constant values."""
    from openscvx.symbolic.expr import Constant, Max

    x = Variable("x", shape=(3,))
    c1 = Constant(np.array([1.0, 2.0, 3.0]))
    c2 = Constant(np.array([2.0, 1.0, 4.0]))

    # Max(x, c1, c2) should fold c1 and c2 into a single constant
    expr = Max(x, c1, c2)
    canonical = expr.canonicalize()

    assert isinstance(canonical, Max)
    assert len(canonical.operands) == 2
    # One should be x, one should be the folded constant
    non_const = [op for op in canonical.operands if not isinstance(op, Constant)]
    consts = [op for op in canonical.operands if isinstance(op, Constant)]
    assert len(non_const) == 1
    assert non_const[0] == x
    assert len(consts) == 1
    # Check that the constant is the element-wise max
    expected = np.maximum(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0]))
    assert np.allclose(consts[0].value, expected)


def test_max_canonicalize_single_operand_collapses():
    """Test that Max with single operand after canonicalization returns that operand."""
    from openscvx.symbolic.expr import Constant, Max

    # Max with only constants should fold to a single constant
    c1 = Constant(np.array([1.0, 2.0]))
    c2 = Constant(np.array([0.5, 3.0]))

    expr = Max(c1, c2)
    canonical = expr.canonicalize()

    # Should collapse to just a constant
    assert isinstance(canonical, Constant)
    expected = np.maximum(np.array([1.0, 2.0]), np.array([0.5, 3.0]))
    assert np.allclose(canonical.value, expected)


def test_max_canonicalize_recursively():
    """Test that Max canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Max

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))

    # Max(x + 0, y + 0) should canonicalize to Max(x, y)
    expr = Max(Add(x, Constant(0.0)), Add(y, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Max)
    assert len(canonical.operands) == 2
    assert x in canonical.operands
    assert y in canonical.operands


# --- Max: JAX Lowering ---


def test_max_constant():
    """Test Max with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Max
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([1.0, 5.0, 2.0])
    y = np.array([3.0, 2.0, 4.0])

    expr = Max(Constant(x), Constant(y))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.maximum(x, y)
    assert jnp.allclose(result, expected)


def test_max_state_variables():
    """Test Max with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Max, State
    from openscvx.symbolic.lower import lower_to_jax

    x_val = jnp.array([1.0, 2.0, 3.0])
    y_val = jnp.array([2.0, 1.5, 3.5])

    x = State("x", (3,))
    x._slice = slice(0, 3)
    y = State("y", (3,))
    y._slice = slice(3, 6)

    expr = Max(x, y)

    state_vec = jnp.concatenate([x_val, y_val])
    fn = lower_to_jax(expr)
    result = fn(state_vec, None, None, None)

    expected = jnp.maximum(x_val, y_val)
    assert jnp.allclose(result, expected)


def test_max_multiple_operands():
    """Test Max with more than two operands."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Max
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([1.0, 5.0, 2.0])
    y = np.array([3.0, 2.0, 4.0])
    z = np.array([2.0, 6.0, 1.0])

    expr = Max(Constant(x), Constant(y), Constant(z))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.maximum(jnp.maximum(x, y), z)
    assert jnp.allclose(result, expected)


def test_max_with_zero_relu_pattern():
    """Test Max(x, 0) which implements ReLU."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Max, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Max(x, 0) is equivalent to ReLU
    expr = Max(state, Constant(0.0))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected)


# --- Max: CVXPy Lowering --- TODO: (norrisg)


# =============================================================================
# LogSumExp
# =============================================================================


def test_logsumexp_creation():
    """Test LogSumExp node creation and properties."""
    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))

    lse = LogSumExp(x, y)
    assert repr(lse) == "logsumexp(Var('x'), Var('y'))"
    assert lse.children() == [x, y]


def test_logsumexp_creation_with_multiple_operands():
    """Test LogSumExp node creation with more than two operands."""
    from openscvx.symbolic.expr import Constant

    x = Variable("x", shape=(1,))
    y = Variable("y", shape=(1,))
    z = Constant(0.0)

    lse = LogSumExp(x, y, z)
    assert len(lse.children()) == 3
    assert repr(lse) == "logsumexp(Var('x'), Var('y'), Const(0.0))"


def test_logsumexp_requires_at_least_two_operands():
    """Test that LogSumExp raises ValueError with fewer than two operands."""
    x = Variable("x", shape=(1,))

    with pytest.raises(ValueError, match="LogSumExp requires two or more operands"):
        LogSumExp(x)


# --- LogSumExp: Shape Checking ---


def test_logsumexp_shape_with_same_shapes():
    """Test LogSumExp shape when all operands have the same shape."""
    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))
    z = Variable("z", shape=(3,))

    lse = LogSumExp(x, y, z)
    assert lse.check_shape() == (3,)


def test_logsumexp_shape_with_broadcasting():
    """Test LogSumExp shape with broadcasting."""
    from openscvx.symbolic.expr import Constant

    x = Variable("x", shape=(3, 4))
    y = Constant(np.array([1.0, 2.0, 3.0, 4.0]))  # shape (4,)

    lse = LogSumExp(x, y)
    assert lse.check_shape() == (3, 4)


def test_logsumexp_shape_with_scalar_broadcast():
    """Test LogSumExp shape with scalar broadcasting."""
    from openscvx.symbolic.expr import Constant

    x = Variable("x", shape=(5, 5))
    scalar = Constant(0.0)  # scalar

    lse = LogSumExp(x, scalar)
    assert lse.check_shape() == (5, 5)


def test_logsumexp_shape_incompatible_raises_error():
    """Test that LogSumExp raises ValueError for incompatible shapes."""
    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(4,))

    lse = LogSumExp(x, y)
    with pytest.raises(ValueError, match="LogSumExp shapes not broadcastable"):
        lse.check_shape()


# --- LogSumExp: Canonicalization ---


def test_logsumexp_canonicalize_flattens_nested():
    """Test that LogSumExp canonicalization flattens nested LogSumExp operations."""
    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))
    z = Variable("z", shape=(3,))

    # LogSumExp(x, LogSumExp(y, z)) should flatten to LogSumExp(x, y, z)
    nested = LogSumExp(x, LogSumExp(y, z))
    canonical = nested.canonicalize()

    assert isinstance(canonical, LogSumExp)
    assert len(canonical.operands) == 3
    assert x in canonical.operands
    assert y in canonical.operands
    assert z in canonical.operands


def test_logsumexp_canonicalize_folds_constants():
    """Test that LogSumExp canonicalization folds constant values."""
    from openscvx.symbolic.expr import Constant

    x = Variable("x", shape=())
    c1 = Constant(1.0)
    c2 = Constant(2.0)

    # LogSumExp(x, c1, c2) should fold c1 and c2 into a single constant
    expr = LogSumExp(x, c1, c2)
    canonical = expr.canonicalize()

    assert isinstance(canonical, LogSumExp)
    assert len(canonical.operands) == 2
    # One should be x, one should be the folded constant
    non_const = [op for op in canonical.operands if not isinstance(op, Constant)]
    consts = [op for op in canonical.operands if isinstance(op, Constant)]
    assert len(non_const) == 1
    assert non_const[0] == x
    assert len(consts) == 1
    # Check that the constant is the logsumexp of c1 and c2
    expected = np.log(np.exp(1.0) + np.exp(2.0))
    assert np.allclose(consts[0].value, expected)


def test_logsumexp_canonicalize_single_operand_collapses():
    """Test that LogSumExp with single operand after canonicalization returns that operand."""
    from openscvx.symbolic.expr import Constant

    # LogSumExp with only constants should fold to a single constant
    c1 = Constant(1.0)
    c2 = Constant(2.0)

    expr = LogSumExp(c1, c2)
    canonical = expr.canonicalize()

    # Should collapse to just a constant
    assert isinstance(canonical, Constant)
    expected = np.log(np.exp(1.0) + np.exp(2.0))
    assert np.allclose(canonical.value, expected)


def test_logsumexp_canonicalize_recursively():
    """Test that LogSumExp canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant

    x = Variable("x", shape=(3,))
    y = Variable("y", shape=(3,))

    # LogSumExp(x + 0, y + 0) should canonicalize to LogSumExp(x, y)
    expr = LogSumExp(Add(x, Constant(0.0)), Add(y, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, LogSumExp)
    assert len(canonical.operands) == 2
    assert x in canonical.operands
    assert y in canonical.operands


# --- LogSumExp: JAX Lowering ---


def test_logsumexp_constant():
    """Test LogSumExp with constant values."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([1.0, 5.0, 2.0])
    y = np.array([3.0, 2.0, 4.0])

    expr = LogSumExp(Constant(x), Constant(y))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # Element-wise logsumexp
    stacked = jnp.stack([x, y], axis=0)
    expected = logsumexp(stacked, axis=0)
    assert jnp.allclose(result, expected)


def test_logsumexp_state_variables():
    """Test LogSumExp with state variables."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x_val = jnp.array([1.0, 2.0, 3.0])
    y_val = jnp.array([2.0, 1.5, 3.5])

    x = State("x", (3,))
    x._slice = slice(0, 3)
    y = State("y", (3,))
    y._slice = slice(3, 6)

    expr = LogSumExp(x, y)

    state_vec = jnp.concatenate([x_val, y_val])
    fn = lower_to_jax(expr)
    result = fn(state_vec, None, None, None)

    stacked = jnp.stack([x_val, y_val], axis=0)
    expected = logsumexp(stacked, axis=0)
    assert jnp.allclose(result, expected)


def test_logsumexp_multiple_operands():
    """Test LogSumExp with more than two operands."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([1.0, 5.0, 2.0])
    y = np.array([3.0, 2.0, 4.0])
    z = np.array([2.0, 6.0, 1.0])

    expr = LogSumExp(Constant(x), Constant(y), Constant(z))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    stacked = jnp.stack([x, y, z], axis=0)
    expected = logsumexp(stacked, axis=0)
    assert jnp.allclose(result, expected)


def test_logsumexp_approximates_max():
    """Test that LogSumExp approximates the maximum function."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Max
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([1.0, 5.0, 2.0])
    y = np.array([3.0, 2.0, 4.0])

    lse_expr = LogSumExp(Constant(x), Constant(y))
    max_expr = Max(Constant(x), Constant(y))

    lse_fn = lower_to_jax(lse_expr)
    max_fn = lower_to_jax(max_expr)

    lse_result = lse_fn(None, None, None, None)
    max_result = max_fn(None, None, None, None)

    # LogSumExp should be >= max
    assert jnp.all(lse_result >= max_result)

    # LogSumExp should be <= max + log(2)
    assert jnp.all(lse_result <= max_result + np.log(2))


def test_logsumexp_scalar_operands():
    """Test LogSumExp with scalar operands."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    x = 1.0
    y = 2.0
    z = 0.5

    expr = LogSumExp(Constant(x), Constant(y), Constant(z))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # Should compute log(exp(1) + exp(2) + exp(0.5))
    expected = np.log(np.exp(x) + np.exp(y) + np.exp(z))
    assert jnp.allclose(result, expected)


def test_logsumexp_with_broadcasting():
    """Test LogSumExp with broadcasting."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from openscvx.symbolic.expr import Constant
    from openscvx.symbolic.lower import lower_to_jax

    x = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    y = np.array([0.5, 1.5])  # shape (2,)

    expr = LogSumExp(Constant(x), Constant(y))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # Broadcast and stack
    broadcasted = jnp.broadcast_arrays(x, y)
    stacked = jnp.stack(list(broadcasted), axis=0)
    expected = logsumexp(stacked, axis=0)
    assert jnp.allclose(result, expected)


# --- LogSumExp: CVXPy Lowering ---


def test_cvxpy_logsumexp():
    """Test log-sum-exp function."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    y_cvx = cp.Variable((10, 3), name="y")
    variable_map = {"x": x_cvx, "y": y_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    y = State("y", shape=(3,))
    expr = LogSumExp(x, y)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Sin & Cos
# =============================================================================


def test_sin_creation():
    """Test Sin node creation and properties."""
    from openscvx.symbolic.expr import Sin

    x = Variable("x", shape=(1,))

    sin_x = Sin(x)
    assert repr(sin_x) == "(sin(Var('x')))"
    assert sin_x.children() == [x]


def test_cos_creation():
    """Test Cos node creation and properties."""
    from openscvx.symbolic.expr import Cos

    x = Variable("x", shape=(1,))

    cos_x = Cos(x)
    assert repr(cos_x) == "(cos(Var('x')))"
    assert cos_x.children() == [x]


# --- Sin & Cos: Shape Checking ---


def test_sin_shape_preserves_input():
    """Test that Sin preserves the shape of its input."""
    from openscvx.symbolic.expr import Sin

    x = Variable("x", shape=(4, 4))

    sin_x = Sin(x)
    assert sin_x.check_shape() == (4, 4)


def test_cos_shape_preserves_input():
    """Test that Cos preserves the shape of its input."""
    from openscvx.symbolic.expr import Cos

    x = Variable("x", shape=(2, 3))

    cos_x = Cos(x)
    assert cos_x.check_shape() == (2, 3)


def test_sin_shape_with_vector():
    """Test Sin shape with vector input."""
    from openscvx.symbolic.expr import Sin

    x = Variable("x", shape=(10,))

    sin_x = Sin(x)
    assert sin_x.check_shape() == (10,)


def test_cos_shape_with_scalar():
    """Test Cos shape with scalar input."""
    from openscvx.symbolic.expr import Cos

    x = Variable("x", shape=())

    cos_x = Cos(x)
    assert cos_x.check_shape() == ()


# --- Sin & Cos: Canonicalization ---


def test_sin_canonicalize_preserves_structure():
    """Test that Sin canonicalization preserves structure."""
    from openscvx.symbolic.expr import Sin

    x = Variable("x", shape=(3,))

    sin_x = Sin(x)
    canonical = sin_x.canonicalize()

    assert isinstance(canonical, Sin)
    assert canonical.operand == x


def test_cos_canonicalize_preserves_structure():
    """Test that Cos canonicalization preserves structure."""
    from openscvx.symbolic.expr import Cos

    x = Variable("x", shape=(3,))

    cos_x = Cos(x)
    canonical = cos_x.canonicalize()

    assert isinstance(canonical, Cos)
    assert canonical.operand == x


def test_sin_canonicalize_recursively():
    """Test that Sin canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Sin

    x = Variable("x", shape=(3,))
    # Sin(x + 0) should canonicalize to Sin(x)
    expr = Sin(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Sin)
    assert canonical.operand == x


def test_cos_canonicalize_recursively():
    """Test that Cos canonicalization recurses into operands."""
    from openscvx.symbolic.expr import Add, Constant, Cos

    x = Variable("x", shape=(3,))
    # Cos(x + 0) should canonicalize to Cos(x)
    expr = Cos(Add(x, Constant(0.0)))
    canonical = expr.canonicalize()

    assert isinstance(canonical, Cos)
    assert canonical.operand == x


# --- Sin & Cos: JAX Lowering ---


def test_sin_constant():
    """Test Sin with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Sin
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    expr = Sin(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.sin(values)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_cos_constant():
    """Test Cos with constant values."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Cos
    from openscvx.symbolic.lower import lower_to_jax

    values = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    expr = Cos(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.cos(values)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_sin_state():
    """Test Sin with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Sin, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])

    state = State("s", (4,))
    state._slice = slice(0, 4)
    expr = Sin(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sin(x)
    assert jnp.allclose(result, expected)


def test_cos_state():
    """Test Cos with state variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Cos, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3])

    state = State("s", (4,))
    state._slice = slice(0, 4)
    expr = Cos(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.cos(x)
    assert jnp.allclose(result, expected)


def test_sin_cos_identity():
    """Test that sin^2 + cos^2 = 1."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Add, Cos, Sin, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, np.pi / 4, np.pi / 2, np.pi])

    state = State("s", (4,))
    state._slice = slice(0, 4)

    # sin^2(x) + cos^2(x)
    expr = Add(Square(Sin(state)), Square(Cos(state)))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.ones_like(x)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_sin_with_expression():
    """Test Sin with a composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Mul, Sin, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, 1.0, 2.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)

    # sin(2*x)
    expr = Sin(Mul(Constant(2.0), state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sin(2.0 * x)
    assert jnp.allclose(result, expected)


def test_cos_with_expression():
    """Test Cos with a composite expression."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Constant, Cos, Mul, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([0.0, 1.0, 2.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)

    # cos(2*x)
    expr = Cos(Mul(Constant(2.0), state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.cos(2.0 * x)
    assert jnp.allclose(result, expected)


# --- Sin & Cos: CVXPy Lowering ---


def test_cvxpy_sin_not_implemented():
    """Test that Sin raises NotImplementedError"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Sin, State
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


# =============================================================================
# Integration Tests - Combined Penalty Functions
# =============================================================================


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
