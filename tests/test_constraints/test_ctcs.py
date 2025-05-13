import pytest
import types
import jax.numpy as jnp
from openscvx.constraints.ctcs import CTCSConstraint, ctcs


@pytest.mark.parametrize(
    "penalty_name, values, expected_sum",
    [
        ("squared_relu", jnp.array([-1.0, 0.0, 2.0]), 4.0),
        ("huber", jnp.array([0.1, 0.3]), 0.005 + (0.3 - 0.125)),
    ],
)
def test_penalty_within_interval(penalty_name, values, expected_sum):
    """Both penalties should sum as expected when node is inside [nodes[0], nodes[1])."""

    @ctcs(nodes=(0, 5), penalty=penalty_name)
    def f(x, u):
        return values

    result = f(jnp.zeros(1), jnp.zeros(1), node=2)
    assert pytest.approx(float(result), rel=1e-6) == expected_sum


def test_any_penalty_outside_interval_returns_zero():
    """Regardless of the penalty, outside the interval the constraint yields 0."""

    @ctcs(nodes=(10, 20), penalty="squared_relu")
    def f(x, u):
        return jnp.array([100.0])

    # test boundary and below
    assert float(f(jnp.zeros(1), jnp.zeros(1), node=20)) == 0.0
    assert float(f(jnp.zeros(1), jnp.zeros(1), node=9)) == 0.0


def test_unknown_penalty_raises_immediately():
    with pytest.raises(ValueError) as exc:
        ctcs(lambda x, u: x, penalty="not_a_real_penalty")
    assert "Unknown penalty not_a_real_penalty" in str(exc.value)


def test_decorator_sets_attributes_and_type():
    @ctcs(nodes=(1, 3), idx=7, penalty="squared_relu")
    def my_cons(x, u):
        return x + u

    assert isinstance(my_cons, CTCSConstraint)
    assert my_cons.nodes == (1, 3)
    assert my_cons.idx == 7
    # and it still uses squared-relu under the hood
    out = my_cons(jnp.array([2.0]), jnp.array([3.0]), node=2)
    assert float(out) == 25.0  # (2+3)=5 → relu² → 25


def test_ctcs_called_directly_without_parentheses():
    """Using `c = ctcs(fn)` should wrap but leave nodes=None, idx=None."""

    def raw_fn(x, u):
        return jnp.array([4.0, -1.0])

    c = ctcs(raw_fn)
    assert isinstance(c, CTCSConstraint)
    assert c.func is raw_fn
    assert c.nodes is None and c.idx is None
    # calling without nodes ought to complain about comparing None to int
    with pytest.raises(TypeError):
        _ = c(jnp.zeros(1), jnp.zeros(1), node=0)


def test_custom_penalty_callable():
    """Allow passing a custom callable as `penalty`."""
    custom_pen = lambda x: x * 2.0
    values = jnp.array([1.0, 2.0, 3.0])

    @ctcs(nodes=(0, 5), penalty=custom_pen)
    def f(x, u):
        return values

    result = f(jnp.zeros(1), jnp.zeros(1), node=3)
    expected = float(jnp.sum(custom_pen(values)))
    assert float(result) == pytest.approx(expected)


def test_default_grad_attrs_none():
    """By default, grad_f_x and grad_f_u should be None."""
    @ctcs(nodes=(0, 5))
    def f(x, u):
        return jnp.array([0.0])

    assert f.grad_f_x is None
    assert f.grad_f_u is None


def test_grad_functions_wrapped_and_callable():
    """Passing grad_f_x and grad_f_u should wrap them to accept (x, u, node)."""
    def base_func(x, u):
        return x + u

    def grad_x(x, u):
        return jnp.array([42.0])

    def grad_u(x, u):
        return jnp.array([24.0])

    @ctcs(nodes=(0, 5), grad_f_x=grad_x, grad_f_u=grad_u)
    def f(x, u):
        return base_func(x, u)

    # Ensure we got lambda wrappers
    assert isinstance(f, CTCSConstraint)
    assert isinstance(f.grad_f_x, types.LambdaType)
    assert isinstance(f.grad_f_u, types.LambdaType)

    x = jnp.array([3.0])
    u = jnp.array([1.0])
    node = 2

    # Call them; unwrap with .item() to get a Python scalar
    wrapped_fx = f.grad_f_x(x, u, node)
    wrapped_fu = f.grad_f_u(x, u, node)

    assert wrapped_fx.shape == (1,)
    assert wrapped_fu.shape == (1,)

    assert wrapped_fx.item() == pytest.approx(grad_x(x, u).item())
    assert wrapped_fu.item() == pytest.approx(grad_u(x, u).item())
