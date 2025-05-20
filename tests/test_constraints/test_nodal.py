import pytest
import jax.numpy as jnp

from openscvx.constraints.nodal import NodalConstraint, nodal


def simple_dot(x, u):
    # f(x,u) = sum_i x_i * u_i
    return jnp.dot(x, u)


def test___call___uses_original_func():
    # __call__ should always call the raw func, even when vectorized=True
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=True)
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.1, 0.2, 0.3])
    assert c(x, u) == pytest.approx(jnp.dot(x, u))


def test_non_vectorized_batched_g_and_grads():
    # vectorized=False (default), convex=False ⇒ g and grads are vmapped
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=False)
    # batch of two 2-vectors
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    # g = vmap(func)
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


def test_vectorized_single_node_path():
    # vectorized=True, convex=False ⇒ g = raw func, grads = raw jacfwd
    c = NodalConstraint(func=simple_dot, convex=False, vectorized=True)
    x = jnp.array([2.0, 3.0])
    u = jnp.array([4.0, 5.0])
    # g should behave like simple_dot
    out = c.g(x, u)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == ()  # scalar
    assert out == pytest.approx(2 * 4 + 3 * 5)
    # grads
    grad_x = c.grad_g_x(x, u)
    grad_u = c.grad_g_u(x, u)
    assert jnp.allclose(grad_x, u)
    assert jnp.allclose(grad_u, x)


def test_convex_skips_jax_transforms():
    # convex=True should not define g, and leave grad_g_x/grad_g_u at their defaults (None)
    c = NodalConstraint(func=simple_dot, convex=True, vectorized=True)
    assert not hasattr(c, 'g')
    assert c.grad_g_x is None
    assert c.grad_g_u is None


def test_custom_gradients_override_default():
    # passing custom grad_g_x / grad_g_u should override jacfwd
    def custom_grad_x(x, u):
        return jnp.full_like(x, 7.0)

    def custom_grad_u(x, u):
        return jnp.full_like(u, 9.0)

    c = NodalConstraint(
        func=simple_dot,
        convex=False,
        vectorized=True,
        grad_g_x=custom_grad_x,
        grad_g_u=custom_grad_u,
    )
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0, 4.0])
    assert jnp.allclose(c.grad_g_x(x, u), jnp.array([7.0, 7.0]))
    assert jnp.allclose(c.grad_g_u(x, u), jnp.array([9.0, 9.0]))


def test_nodal_decorator_passes_parameters_through():
    def custom_grad_x(x, u):
        return jnp.array([42.0, 43.0])

    def custom_grad_u(x, u):
        return jnp.array([84.0, 85.0])

    @nodal(
        nodes=[10, 20, 30],
        convex=False,
        vectorized=True,
        grad_g_x=custom_grad_x,
        grad_g_u=custom_grad_u,
    )
    def f2(x, u):
        return jnp.sum(x) + jnp.sum(u)

    assert isinstance(f2, NodalConstraint)
    assert f2.nodes == [10, 20, 30]
    assert f2.convex is False
    assert f2.vectorized is True
    # decorator‐provided grad hooks should be set directly
    assert f2.grad_g_x is custom_grad_x
    assert f2.grad_g_u is custom_grad_u
