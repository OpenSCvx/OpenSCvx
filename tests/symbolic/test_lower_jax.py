import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Div,
    Equality,
    Expr,
    Huber,
    Inequality,
    MatMul,
    Mul,
    Neg,
    PositivePart,
    SmoothReLU,
    Square,
    Sub,
)
from openscvx.backend.lower import lower, lower_to_jax
from openscvx.backend.lowerers.jax import JaxLowerer
from openscvx.backend.state import State


class UnregisteredExpr(Expr):
    pass


def test_jaxlowerer_raises_when_no_visitor_registered():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    with pytest.raises(NotImplementedError) as excinfo:
        # this should internally call dispatch() and fail
        jl.lower(node)

    msg = str(excinfo.value)
    assert "JaxLowerer" in msg, "should mention the lowerer class name"
    assert "UnregisteredExpr" in msg, "should mention the Expr subclass name"


def test_top_level_lower_raises_for_unregistered_expr():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    # our top-level lower() simply forwards to jl.lower(...)
    with pytest.raises(NotImplementedError):
        lower(node, jl)


def test_jax_lower_constant():
    const = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    jl = JaxLowerer()
    f = jl.visit_constant(const)
    out = f(None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1, 2], [3, 4]]))


def test_jax_lower_state_without_slice_raises():
    s = State("s", (3,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl.visit_state(s)


def test_jax_lower_control_without_slice_raises():
    c = Control("c", (2,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl.visit_control(c)


def test_jax_lower_state_with_slice():
    x = jnp.arange(10.0)
    s = State("s", (4,))
    s._slice = slice(2, 6)
    jl = JaxLowerer()
    f = jl.visit_state(s)
    out = f(x, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert jnp.allclose(out, x[2:6])


def test_jax_lower_control_with_slice():
    u = jnp.arange(8.0)
    c = Control("c", (3,))
    c._slice = slice(5, 8)
    jl = JaxLowerer()
    f = jl.visit_control(c)
    out = f(None, u)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, u[5:8])


def test_jax_lower_add_and_mul_of_slices():
    x = jnp.arange(8.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr_add = Add(a, b)
    expr_mul = Mul(a, b)

    jl = JaxLowerer()
    f_res_add = jl.visit_add(expr_add)
    res_add = f_res_add(x, None)
    f_res_mul = jl.visit_mul(expr_mul)
    res_mul = f_res_mul(x, None)

    assert jnp.allclose(res_add, x[0:3] + x[3:6])
    assert jnp.allclose(res_mul, x[0:3] * x[3:6])


def test_jax_lower_sub_and_div_of_slices():
    x = jnp.arange(8.0)
    u = jnp.arange(8.0) * 3.0
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = Control("b", (3,))
    b._slice = slice(0, 3)
    c = Constant(2.0)
    expr_sub = Sub(a, b)
    expr_div = Div(a, c)

    jl = JaxLowerer()
    f_res_sub = jl.visit_sub(expr_sub)
    res_sub = f_res_sub(x, u)
    f_res_div = jl.visit_div(expr_div)
    res_div = f_res_div(x, u)

    assert jnp.allclose(res_sub, x[0:3] - u[0:3])
    assert jnp.allclose(res_div, x[0:3] / c.value)


def test_jax_lower_matmul_vector_matrix():
    # (2×2 matrix) @ (2-vector)
    M = Constant(np.array([[1.0, 0.0], [0.0, 2.0]]))
    v = Constant(np.array([3.0, 4.0]))
    expr = MatMul(M, v)

    jl = JaxLowerer()
    f = jl.visit_matmul(expr)
    out = f(None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([3.0, 8.0]))


def test_jax_lower_neg_and_composite():
    x = jnp.arange(6.0)
    u = jnp.arange(6.0) * 3
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = Control("b", (2,))
    b._slice = slice(0, 2)
    c = Constant(np.array([1.0, 1.0]))

    # expr = -((a + b) * c)
    expr = Neg(Mul(Add(a, b), c))
    jl = JaxLowerer()
    f = jl.visit_neg(expr)
    out = f(x, u)

    expected = -((x[0:2] + u[0:2]) * jnp.array([1.0, 1.0]))
    assert jnp.allclose(out, expected)


def test_lower_to_jax_constant_produces_callable():
    c = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    fns = lower_to_jax([c])
    assert isinstance(fns, list) and len(fns) == 1
    fn = fns[0]
    out = fn(None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_lower_to_jax_add_with_slices():
    x = jnp.arange(6.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr = Add(a, b)

    fn = lower_to_jax(expr)
    out = fn(x, None)
    expected = x[0:3] + x[3:6]
    assert jnp.allclose(out, expected)


def test_lower_to_jax_multiple_exprs_returns_in_order():
    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([1.0, 2.0, 3.0])
    # expr1: constant, expr2: identity of x
    c = Constant(np.array([1.0, 2.0, 3.0]))
    v = State("v", (3,))
    v._slice = slice(0, 3)
    exprs = [c, v]

    fns = lower_to_jax(exprs)
    assert len(fns) == 2

    f_const, f_x = fns
    assert jnp.allclose(f_const(x, None), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(f_x(x, u), x)


def test_equality_constraint_lowering():
    """Test that equality constraints are lowered to residual form (lhs - rhs)."""
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0, 1.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Constraint: x == 2*u (should become x - 2*u == 0)
    lhs = state
    rhs = Mul(Constant(2.0), control)
    constraint = Equality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl.visit_constraint(constraint)
    residual = fn(x, u)

    # Residual should be lhs - rhs = x - 2*u
    expected = x - 2.0 * u
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)


def test_inequality_constraint_lowering():
    """Test that inequality constraints are lowered to residual form (lhs - rhs)."""
    x = jnp.array([0.5, 1.5, 2.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 2.0 (should become x - 2.0 <= 0)
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl.visit_constraint(constraint)  # Both use the same visitor
    residual = fn(x, None)

    # Residual should be lhs - rhs = x - 2.0
    expected = x - 2.0
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)

    # Check that residual is negative when constraint is satisfied
    # and positive when violated
    assert residual[0] < 0  # 0.5 - 2.0 = -1.5 (satisfied)
    assert residual[1] < 0  # 1.5 - 2.0 = -0.5 (satisfied)
    assert residual[2] > 0  # 2.5 - 2.0 = 0.5 (violated)


def test_constraint_lowering_with_lower_to_jax():
    """Test constraint lowering through the top-level lower_to_jax function."""
    x = jnp.array([1.0, 3.0])
    u = jnp.array([0.5])

    pos = State("pos", (2,))
    pos._slice = slice(0, 2)
    vel = Control("vel", (1,))
    vel._slice = slice(0, 1)

    # Mixed constraint: pos[0] + 2*vel <= pos[1]
    # Rearranged: pos[0] + 2*vel - pos[1] <= 0
    lhs = Add(pos[0], Mul(Constant(2.0), vel))
    rhs = pos[1]
    constraint = Inequality(lhs, rhs)

    # Lower using the top-level function
    fn = lower_to_jax(constraint)
    residual = fn(x, u)

    # Expected: pos[0] + 2*vel - pos[1] = 1.0 + 2*0.5 - 3.0 = -1.0
    expected = 1.0 + 2.0 * 0.5 - 3.0
    assert jnp.allclose(residual, expected)
    assert residual < 0  # Constraint is satisfied


def test_concat_simple():
    x = jnp.arange(5.0)
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(2, 4)
    c = Constant(9.0)
    expr = Concat(a, b, c)

    fn = lower_to_jax(expr)
    out = fn(x, None)
    expected = jnp.concatenate([x[0:2], x[2:4], jnp.array([9.0])], axis=0)
    assert jnp.allclose(out, expected)
    assert out.shape == (5,)


def test_lower_to_jax_double_integrator():
    x = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0
    pos = State("pos", (3,))
    pos._slice = slice(0, 3)
    vel = State("vel", (3,))
    vel._slice = slice(3, 6)

    acc = Control("acc", (3,))
    acc._slice = slice(0, 3)

    pos_dot = vel
    vel_dot = acc / m + Constant(np.array([0.0, 0.0, g]))

    dynamics_expr = Concat(pos_dot, vel_dot)
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x, u)

    expected = jnp.concatenate([x[3:6], u / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


def test_index_and_slice():
    # make a 4-vector state
    x = jnp.array([10.0, 20.0, 30.0, 40.0])
    s = State("s", (4,))
    s._slice = slice(0, 4)

    # index it and slice it
    expr_elem = s[2]
    expr_slice = s[1:3]

    # lower → callables
    fn_elem = lower_to_jax(expr_elem)
    fn_slice = lower_to_jax(expr_slice)

    # check results
    out_elem = fn_elem(x, None)
    out_slice = fn_slice(x, None)

    assert out_elem.shape == () or out_elem.shape == ()  # scalar or 0-D
    assert out_elem == x[2]

    assert out_slice.shape == (2,)
    assert jnp.allclose(out_slice, x[1:3])


def test_lower_to_jax_double_integrator_indexed():
    # numeric inputs
    x_jax = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u_jax = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0

    # one 6-vector state
    x = State("x", (6,))
    x._slice = slice(0, 6)

    # 3-vector control
    u = Control("u", (3,))
    u._slice = slice(0, 3)

    pos_dot = x[3:6]
    vel_dot = u / m + Constant(np.array([0.0, 0.0, g]))
    dynamics_expr = Concat(pos_dot, vel_dot)

    # lower and execute
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x_jax, u_jax)

    # expected by hand
    expected = jnp.concatenate([x_jax[3:6], u_jax / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


# TODO: (norrisg) move to separate file
# Tests for lowering penalty function expressions to JAX
def test_positive_part_constant():
    """Test PositivePart with constant values."""
    # Test with mixed positive and negative values
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = PositivePart(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert jnp.allclose(result, expected)


def test_positive_part_state():
    """Test PositivePart with state variables."""
    x = jnp.array([-3.0, -1.0, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = PositivePart(state)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected)


def test_positive_part_expression():
    """Test PositivePart with a composite expression."""
    x = jnp.array([1.0, 2.0, 3.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)
    threshold = Constant(np.array([2.0, 2.0, 2.0]))

    # pos(x - 2)
    expr = PositivePart(Sub(state, threshold))

    fn = lower_to_jax(expr)
    result = fn(x, None)

    expected = jnp.array([0.0, 0.0, 1.0])  # max(x - 2, 0)
    assert jnp.allclose(result, expected)


def test_square_constant():
    """Test Square with constant values."""
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = Square(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None)

    expected = values**2
    assert jnp.allclose(result, expected)


def test_square_state():
    """Test Square with state variables."""
    x = jnp.array([-3.0, -1.5, 0.0, 1.5, 3.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = Square(state)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    expected = x * x
    assert jnp.allclose(result, expected)


def test_squared_relu_pattern():
    """Test the squared ReLU pattern: (max(x, 0))^2."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Build squared ReLU: Square(PositivePart(x))
    expr = Square(PositivePart(state))

    fn = lower_to_jax(expr)
    result = fn(x, None)

    # Expected: [0, 0, 0, 1, 4]
    expected = jnp.maximum(x, 0.0) ** 2
    assert jnp.allclose(result, expected)


def test_huber_constant():
    """Test Huber penalty with constant values."""
    values = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    delta = 0.25

    expr = Huber(Constant(values), delta=delta)

    fn = lower_to_jax(expr)
    result = fn(None, None)

    # Huber formula:
    # if |x| <= delta: 0.5 * x^2
    # else: delta * (|x| - 0.5 * delta)
    expected = jnp.where(
        jnp.abs(values) <= delta, 0.5 * values**2, delta * (jnp.abs(values) - 0.5 * delta)
    )
    assert jnp.allclose(result, expected)


def test_huber_state_various_deltas():
    """Test Huber penalty with different delta values."""
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Test with small delta
    expr_small = Huber(state, delta=0.1)
    fn_small = lower_to_jax(expr_small)
    result_small = fn_small(x, None)

    # Most values should be in the linear region
    expected_small = jnp.where(jnp.abs(x) <= 0.1, 0.5 * x**2, 0.1 * (jnp.abs(x) - 0.5 * 0.1))
    assert jnp.allclose(result_small, expected_small)

    # Test with large delta
    expr_large = Huber(state, delta=3.0)
    fn_large = lower_to_jax(expr_large)
    result_large = fn_large(x, None)

    # All values should be in the quadratic region
    expected_large = 0.5 * x**2  # Since all |x| <= 3.0
    assert jnp.allclose(result_large, expected_large)


def test_huber_with_positive_part():
    """Test Huber applied to positive part (common CTCS pattern)."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Huber(PositivePart(x))
    expr = Huber(PositivePart(state), delta=0.5)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    # First apply positive part
    pos_x = jnp.maximum(x, 0.0)
    # Then apply Huber
    expected = jnp.where(jnp.abs(pos_x) <= 0.5, 0.5 * pos_x**2, 0.5 * (jnp.abs(pos_x) - 0.5 * 0.5))
    assert jnp.allclose(result, expected)


def test_smooth_relu_constant():
    """Test SmoothReLU with constant values."""
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = 1e-8

    expr = SmoothReLU(Constant(values), c=c)

    fn = lower_to_jax(expr)
    result = fn(None, None)

    # SmoothReLU: sqrt(max(x, 0)^2 + c^2) - c
    expected = jnp.sqrt(jnp.maximum(values, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_state():
    """Test SmoothReLU with state variables."""
    x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_approaches_relu():
    """Test that SmoothReLU approaches ReLU as c → 0."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Very small c
    expr = SmoothReLU(state, c=1e-12)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    # Should be very close to max(x, 0)
    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected, atol=1e-10)


def test_smooth_relu_differentiability_at_zero():
    """Test that SmoothReLU is smooth at x=0."""
    # Test values around zero
    x = jnp.array([-0.01, -0.001, 0.0, 0.001, 0.01])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None)

    # At x=0, should equal 0 (since sqrt(c^2) - c = 0)
    assert jnp.abs(result[2]) < 1e-10

    # Should be continuous and smooth
    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_penalty_in_constraint_expression():
    """Test penalty functions used in a constraint-like expression."""
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
    result = fn(x, u)

    # Expected: [0, 0, 0.25] since only x[2]=2.5 violates
    expected = jnp.array([0.0, 0.0, 0.25])
    assert jnp.allclose(result, expected)


def test_combined_penalties():
    """Test combining different penalty functions."""
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

    result_sq = fn_sq(x, None)
    result_hub = fn_hub(x, None)
    result_smooth = fn_smooth(x, None)

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
    u = jnp.array([-1.0, 0.5, 2.0])

    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Control constraint: |u| <= 1.0
    # Penalty for upper bound: Square(PositivePart(u - 1.0))
    upper_limit = Constant(np.array([1.0, 1.0, 1.0]))
    upper_violation = Sub(control, upper_limit)
    penalty = Square(PositivePart(upper_violation))

    fn = lower_to_jax(penalty)
    result = fn(None, u)

    # Expected: [0, 0, 1] since only u[2]=2.0 violates
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(result, expected)
