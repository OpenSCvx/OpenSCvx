import pytest

import jax.numpy as jnp

from openscvx.constraints.violation import CTCSViolation
from openscvx.augmentation.state_augmentation import build_augmented_dynamics, get_augmented_dynamics
from openscvx.dynamics import dynamics

# base dynamics: ẋ = [ 2*x0, 3*x1 ] + [ 1*u0, 0*u1 ]
@dynamics(
    A=lambda x,u: jnp.diag(jnp.array([2.0,3.0])),
    B=lambda x,u: jnp.array([[1.0,0.0],[0.0,0.0]])
)
def f(x,u):
    return jnp.array([2*x[0], 3*x[1]]) + jnp.array([u[0], 0.0])

# one violation: g(x,u) = [ x0 + u0 ], with user grad_x = [1,0], grad_u=[1]
vio = CTCSViolation(
    g=lambda x,u,node: jnp.array([x[0] + u[0]]),
    g_grad_x=lambda x,u,node: jnp.array([[1.0,0.0]]),
    g_grad_u=lambda x,u,node: jnp.array([[1.0,0.0]])
)

def test_augmented_dynamics_stack():
    idx_x = slice(0,2)
    idx_u = slice(0,2)
    dyn_aug = get_augmented_dynamics(f.f, [vio], idx_x, idx_u)

    x = jnp.array([1.0,2.0,   0.0])   # last entry is “violation states”
    u = jnp.array([3.0,4.0,   0.0])

    # original ẋ = [2*1+3, 3*2+0] = [5,6]
    # violation = [1+3] = [4]
    out = dyn_aug(x,u, node=0)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(5)
    assert out[1] == pytest.approx(6)
    assert out[2] == pytest.approx(4)

def test_jacobians_with_custom_grads():
    idx_x = slice(0,2)
    idx_u = slice(0,2)
    dyn_aug = build_augmented_dynamics(f, [vio], idx_x, idx_u)

    x = jnp.array([1.0,2.0, 0.0])
    u = jnp.array([3.0,4.0, 0.0])
    A = dyn_aug.A(x,u,0)
    B = dyn_aug.B(x,u,0)

    # Top-left block is diag([2,3])
    assert A[0,0] == pytest.approx(2.0)
    assert A[1,1] == pytest.approx(3.0)

    # Top-right block (w.r.t violation-states) is zeros
    assert (A[:2,2:] == 0).all()

    # Violation-block: ∂g/∂x_true = [1,0], padded with zeros
    assert A[2,0] == pytest.approx(1.0)
    assert A[2,1] == pytest.approx(0.0)
    assert A[2,2] == pytest.approx(0.0)  # no cross-violation pad

    # B: top is custom B, bottom is grad_u = [1,0]
    assert B[0,0] == pytest.approx(1.0)
    assert B[0,1] == pytest.approx(0.0)
    assert B[1,0] == pytest.approx(0.0)
    assert B[2,0] == pytest.approx(1.0)  # ∂g/∂u0
    assert B[2,1] == pytest.approx(0.0)
