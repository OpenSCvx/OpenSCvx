import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.backend.expr import Add, Constant, MatMul, Mul, Neg
from openscvx.backend.lower import lower_to_jax
from openscvx.backend.lowerers.jax import JaxLowerer
from openscvx.backend.variable import Variable


def test_jax_lower_constant():
    const = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    jl = JaxLowerer()
    f_out = jl.visit_constant(const)
    out = f_out(None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1, 2], [3, 4]]))


def test_jax_lower_variable_without_slice_raises():
    v = Variable("v", (3,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl.visit_variable(v)


def test_jax_lower_variable_with_slice():
    full_x = jnp.arange(10.0)
    v = Variable("v", (4,))
    v._slice = slice(2, 6)
    jl = JaxLowerer()
    f_out = jl.visit_variable(v)
    out = f_out(full_x, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert jnp.allclose(out, full_x[2:6])


def test_jax_lower_add_and_mul_of_slices():
    full_x = jnp.arange(8.0)
    a = Variable("a", (3,))
    a._slice = slice(0, 3)
    b = Variable("b", (3,))
    b._slice = slice(3, 6)
    expr_add = Add(a, b)
    expr_mul = Mul(a, b)

    jl = JaxLowerer()
    f_res_add = jl.visit_add(expr_add)
    res_add = f_res_add(full_x, None)
    f_res_mul = jl.visit_mul(expr_mul)
    res_mul = f_res_mul(full_x, None)

    assert jnp.allclose(res_add, full_x[0:3] + full_x[3:6])
    assert jnp.allclose(res_mul, full_x[0:3] * full_x[3:6])


def test_jax_lower_matmul_vector_matrix():
    # (2×2 matrix) @ (2-vector)
    M = Constant(np.array([[1.0, 0.0], [0.0, 2.0]]))
    v = Constant(np.array([3.0, 4.0]))
    expr = MatMul(M, v)

    jl = JaxLowerer()
    f_out = jl.visit_matmul(expr)
    out = f_out(None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([3.0, 8.0]))


def test_jax_lower_neg_and_composite():
    full_x = jnp.arange(6.0)
    a = Variable("a", (2,))
    a._slice = slice(0, 2)
    b = Variable("b", (2,))
    b._slice = slice(2, 4)
    c = Constant(np.array([1.0, 1.0]))

    # expr = -((a + b) * c)
    expr = Neg(Mul(Add(a, b), c))
    jl = JaxLowerer()
    f_out = jl.visit_neg(expr)
    out = f_out(full_x, None)

    expected = -((full_x[0:2] + full_x[2:4]) * jnp.array([1.0, 1.0]))
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
    full_x = jnp.arange(6.0)
    a = Variable("a", (3,))
    a._slice = slice(0, 3)
    b = Variable("b", (3,))
    b._slice = slice(3, 6)
    expr = Add(a, b)

    [fn] = lower_to_jax([expr])
    out = fn(full_x, None)
    expected = full_x[0:3] + full_x[3:6]
    assert jnp.allclose(out, expected)


def test_lower_to_jax_multiple_exprs_returns_in_order():
    full_x = jnp.array([10.0, 20.0, 30.0])
    # expr1: constant, expr2: identity of x
    c = Constant(np.array([1.0, 2.0, 3.0]))
    x = Variable("x", (3,))
    x._slice = slice(0, 3)
    exprs = [c, x]

    fns = lower_to_jax(exprs)
    assert len(fns) == 2

    f_const, f_x = fns
    assert jnp.allclose(f_const(full_x, None), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(f_x(full_x, None), full_x)