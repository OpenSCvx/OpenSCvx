import pytest
import jax.numpy as jnp
from openscvx.constraints.ctcs import CTCSConstraint, ctcs


def test_squared_relu_penalty_in_interval():
    # a func that returns [-1, 0, 2] → squared_relu ⇒ [0, 0, 4], sum = 4
    f = lambda x, u: jnp.array([-1.0, 0.0, 2.0])
    pen = lambda x: jnp.maximum(0, x) ** 2
    c = CTCSConstraint(func=f, penalty=pen, nodes=(0, 10))
    res = c(jnp.array([0]), jnp.array([0]), node=5)
    assert float(res) == 4.0


def test_squared_relu_outside_interval():
    # same func, but node == 10 is at the boundary (exclusive) → 0
    f = lambda x, u: jnp.array([1.0, 2.0])
    pen = lambda x: jnp.maximum(0, x) ** 2
    c = CTCSConstraint(func=f, penalty=pen, nodes=(0, 10))
    assert float(c(jnp.array([0]), jnp.array([0]), node=10)) == 0.0
    assert float(c(jnp.array([0]), jnp.array([0]), node=-1)) == 0.0


def test_huber_penalty_in_interval():
    # huber with delta=0.25 on [0.1, 0.3] → [0.5*0.1^2=0.005, 0.3-0.125=0.175], sum = 0.18
    @ctcs(nodes=(0, 5), penalty="huber")
    def f(x, u):
        return jnp.array([0.1, 0.3])

    res = f(jnp.array([0]), jnp.array([0]), node=2)
    assert pytest.approx(float(res), rel=1e-6) == 0.18


def test_unknown_penalty_raises():
    with pytest.raises(ValueError) as exc:
        ctcs(lambda x, u: x, penalty="not_a_real_penalty")
    assert "Unknown penalty not_a_real_penalty" in str(exc.value)


def test_decorator_returns_CTCSConstraint_and_attributes():
    @ctcs(nodes=(1, 3), idx=7)
    def my_constraint(x, u):
        return x + u

    # should be turned into a CTCSConstraint
    assert isinstance(my_constraint, CTCSConstraint)
    assert my_constraint.nodes == (1, 3)
    assert my_constraint.idx == 7
    # func should be the original function
    xs = jnp.array([2.0])
    us = jnp.array([3.0])
    # inside interval → (2+3)=5, squared_relu(5)=25
    out = my_constraint(xs, us, node=2)
    assert float(out) == 25.0


def test_ctcs_called_directly_without_parentheses():
    def raw_fn(x, u):
        return jnp.array([4.0, -1.0])

    c = ctcs(raw_fn)  # equivalent to @ctcs
    # defaults: penalty="squared_relu", nodes=None, idx=None
    assert isinstance(c, CTCSConstraint)
    assert c.func is raw_fn
    assert c.penalty is not None
    assert c.nodes is None
    assert c.idx is None

    # calling __call__ without nodes should error
    with pytest.raises(TypeError):
        _ = c(jnp.array([0]), jnp.array([0]), node=0)
