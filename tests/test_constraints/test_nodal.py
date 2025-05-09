import pytest
import jax.numpy as jnp

from openscvx.constraints.nodal import NodalConstraint, nodal


def simple_dot(x, u):
    # f(x,u) = sum_i x_i * u_i
    return jnp.dot(x, u)


def test___call___uses_original_func():
    # __call__ should bypass jax-transformed methods
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=True)
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.1, 0.2, 0.3])
    # same as simple_dot(x,u)
    assert c(x, u) == pytest.approx(jnp.dot(x, u))


def test_non_vectorized_batched_g_and_grads():
    # vectorized=False (default), convex=False
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=False)
    # batch of two 2-vectors
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    # g = vmap(jit(func), in_axes=(0,0))
    expected = jnp.array([jnp.dot(x[0], u[0]), jnp.dot(x[1], u[1])])
    out = c.g(x, u)
    assert out.shape == (2,)
    assert jnp.allclose(out, expected)
    # gradient w.r.t. x should be u
    grad_x = c.grad_g_x(x, u)
    assert grad_x.shape == x.shape
    assert jnp.allclose(grad_x, u)
    # gradient w.r.t. u should be x
    grad_u = c.grad_g_u(x, u)
    assert jnp.allclose(grad_u, x)


def test_vectorized_single_node_jit_path():
    # vectorized=True, convex=False
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=True)
    x = jnp.array([2.0, 3.0])
    u = jnp.array([4.0, 5.0])
    # g = jit(func)
    out = c.g(x, u)
    assert out.shape == ()  # scalar
    assert out == pytest.approx(2 * 4 + 3 * 5)
    # grads
    grad_x = c.grad_g_x(x, u)
    grad_u = c.grad_g_u(x, u)
    assert jnp.allclose(grad_x, u)
    assert jnp.allclose(grad_u, x)


def test_convex_skips_jax_transforms():
    # convex=True should not define g, grad_g_x, or grad_g_u
    c = NodalConstraint(func=simple_dot, convex=True, vectorized=True)
    for attr in ("g", "grad_g_x", "grad_g_u"):
        with pytest.raises(AttributeError):
            getattr(c, attr)


def test_nodal_decorator_passes_parameters_through():
    @nodal(nodes=[10, 20, 30], convex=True, vectorized=False)
    def f2(x, u):
        return jnp.sum(x) + jnp.sum(u)

    # decorator returns a fully initialized NodalConstraint
    assert isinstance(f2, NodalConstraint)
    assert f2.nodes == [10, 20, 30]
    assert f2.convex is True
    assert f2.vectorized is False
