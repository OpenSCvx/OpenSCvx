import pytest
import jax.numpy as jnp

from openscvx.constraints.ctcs import CTCSConstraint
from openscvx.constraints.violation import (
    CTCSViolation,
    get_g_grad_x,
    get_g_grad_u,
    get_g_func,
    get_g_funcs,
)


def make_constraint(
    value: float,
    idx: int = None,
    nodes=(0, 10),
    grad_x: float = None,
    grad_u: float = None,
) -> CTCSConstraint:
    """Helper to create a CTCSConstraint whose func always returns `value`,
    and whose grad_f_x / grad_f_u return `grad_x` / `grad_u` if provided."""
    return CTCSConstraint(
        func=lambda x, u: jnp.array(value),
        penalty=lambda x: x,
        nodes=nodes,
        idx=idx,
        grad_f_x=(lambda x, u: jnp.array(grad_x)) if grad_x is not None else None,
        grad_f_u=(lambda x, u: jnp.array(grad_u)) if grad_u is not None else None,
    )


def test_get_g_func_empty():
    g = get_g_func([])
    # sum of zero constraints should be zero
    result = g(jnp.array([1.0]), jnp.array([2.0]), node=0)
    assert result == 0


def test_get_g_func_with_constraints():
    c1 = make_constraint(1.0, idx=0)
    c2 = make_constraint(2.0, idx=0)
    g = get_g_func([c1, c2])
    # value does not depend on x,u
    result = g(jnp.zeros(1), jnp.zeros(1), node=5)
    assert float(result) == pytest.approx(3.0)


def test_get_g_grad_x_no_grad():
    gdx = get_g_grad_x([])
    assert gdx(jnp.array([0.0]), jnp.array([0.0]), node=0) is None


def test_get_g_grad_x_with_grads():
    c1 = make_constraint(0.0, idx=0, grad_x=1.0)
    c2 = make_constraint(0.0, idx=0, grad_x=2.0)
    gdx = get_g_grad_x([c1, c2])
    out = gdx(jnp.zeros(1), jnp.zeros(1), node=0)
    assert isinstance(out, jnp.ndarray)
    assert float(out) == pytest.approx(3.0)


def test_get_g_grad_x_mixed():
    # one constraint has grad_x, one does not
    c1 = make_constraint(0.0, idx=0, grad_x=1.0)
    c2 = make_constraint(0.0, idx=0, grad_x=None)
    gdx = get_g_grad_x([c1, c2])
    out = gdx(jnp.zeros(1), jnp.zeros(1), node=0)
    # only c1 contributes
    assert float(out) == pytest.approx(1.0)


def test_get_g_grad_u_no_grad():
    gdu = get_g_grad_u([])
    assert gdu(jnp.array([0.0]), jnp.array([0.0]), node=0) is None


def test_get_g_grad_u_with_grads():
    c1 = make_constraint(0.0, idx=0, grad_u=3.0)
    c2 = make_constraint(0.0, idx=0, grad_u=4.0)
    gdu = get_g_grad_u([c1, c2])
    out = gdu(jnp.zeros(1), jnp.zeros(1), node=0)
    assert float(out) == pytest.approx(7.0)


def test_get_g_grad_u_mixed():
    c1 = make_constraint(0.0, idx=0, grad_u=3.0)
    c2 = make_constraint(0.0, idx=0, grad_u=None)
    gdu = get_g_grad_u([c1, c2])
    out = gdu(jnp.zeros(1), jnp.zeros(1), node=0)
    assert float(out) == pytest.approx(3.0)


def test_get_g_funcs_missing_idx():
    # any constraint without idx should error
    bad = make_constraint(1.0, idx=None)
    with pytest.raises(ValueError):
        get_g_funcs([bad])


def test_get_g_funcs_grouping_and_grad_flags():
    # two constraints in bucket 1, both have grad_x and grad_u
    c1 = make_constraint(1.0, idx=1, grad_x=1.0, grad_u=10.0)
    c2 = make_constraint(2.0, idx=1, grad_x=2.0, grad_u=20.0)
    # one constraint in bucket 2, no grads
    c3 = make_constraint(3.0, idx=2)

    violations = get_g_funcs([c1, c2, c3])
    # should produce one violation per distinct idx, in sorted order
    assert len(violations) == 2
    v1, v2 = violations

    # -- bucket idx=1 --
    # g sums func values 1 + 2 = 3
    assert float(v1.g(None, None, node=5)) == pytest.approx(3.0)

    # because all c1,c2 have grad_f_x, v1.g_grad_x is set (but note code swaps grad_u under the hood)
    assert callable(v1.g_grad_x)
    # get_g_grad_u on [c1,c2] sums their grad_u: 10 + 20 = 30
    assert float(v1.g_grad_x(None, None, node=5)) == pytest.approx(30.0)

    # because all c1,c2 have grad_f_u, v1.g_grad_u is set (but code uses get_g_grad_x internally)
    assert callable(v1.g_grad_u)
    # get_g_grad_x on [c1,c2] sums their grad_x: 1 + 2 = 3
    assert float(v1.g_grad_u(None, None, node=5)) == pytest.approx(3.0)

    # -- bucket idx=2 --
    # only c3 in bucket, g sums 3
    assert float(v2.g(None, None, node=1)) == pytest.approx(3.0)
    # c3 has no grads so both should be None
    assert v2.g_grad_x is None
    assert v2.g_grad_u is None
