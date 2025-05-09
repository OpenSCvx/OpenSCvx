import pytest
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
