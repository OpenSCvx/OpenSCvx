import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import Add, Constant, Div, MatMul, Mul, Neg, Sub
from openscvx.backend.lower import lower_to_jax
from openscvx.backend.lowerers.jax import JaxLowerer
from openscvx.backend.state import State


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

    [fn] = lower_to_jax([expr])
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
