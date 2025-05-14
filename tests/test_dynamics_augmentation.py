import pytest

import jax.numpy as jnp

from openscvx.constraints.violation import CTCSViolation
from openscvx.augmentation.dynamics_augmentation import build_augmented_dynamics, get_augmented_dynamics, get_jacobians
from openscvx.dynamics import dynamics, Dynamics

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

def test_get_augmented_dynamics_no_violations():
    def f(x, u):
        # simple linear dynamics: x + 2*u
        return x + 2 * u

    dyn_fn = get_augmented_dynamics(f, [], slice(None), slice(None))
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0, 4.0])
    out = dyn_fn(x, u, node=0)

    # Expect x + 2*u = [1+6, 2+8] = [7, 10]
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([7.0, 10.0]))


def test_get_augmented_dynamics_with_violations():
    def f(x, u):
        # always zero
        return jnp.zeros_like(x)

    # violation 1: constant [1]
    vio1 = CTCSViolation(g=lambda x, u, node: jnp.array([1.0]))
    # violation 2: constant [2,3]
    vio2 = CTCSViolation(g=lambda x, u, node: jnp.array([2.0, 3.0]))

    dyn_fn = get_augmented_dynamics(f, [vio1, vio2], slice(None), slice(None))
    x = jnp.array([10.0])
    u = jnp.array([20.0])
    out = dyn_fn(x, u, node=5)

    # f → [0], then [1], then [2,3] ⇒ [0,1,2,3]
    assert out.shape == (4,)
    assert jnp.allclose(out, jnp.array([0.0, 1.0, 2.0, 3.0]))


def test_get_jacobians_no_grads():
    # f(x,u) = [2*x0 + 3*u0]
    def f(x, u):
        return jnp.array([2 * x[0] + 3 * u[0]])

    dyn_nonaug = Dynamics(f=f)  # A, B are None
    dyn_aug_fn = get_augmented_dynamics(f, [], slice(None), slice(None))
    A_fn, B_fn = get_jacobians(dyn_aug_fn, dyn_nonaug, [], slice(None), slice(None))

    x = jnp.array([5.0])
    u = jnp.array([7.0])

    A = A_fn(x, u, node=0)
    B = B_fn(x, u, node=0)

    # ∂/∂x[2*x + 3*u] = 2,  ∂/∂u = 3
    assert A.shape == (1, 1)
    assert B.shape == (1, 1)
    assert pytest.approx(A[0, 0]) == 2.0
    assert pytest.approx(B[0, 0]) == 3.0


def test_get_jacobians_with_violations_no_custom_grads():
    # f(x, u) = x + 2*u
    def f(x, u):
        return jnp.array([x[0] + 2 * u[0]])

    dyn_nonaug = Dynamics(f=f)
    # two violations: g1=[x], g2=[u]
    vio1 = CTCSViolation(g=lambda x, u, node: jnp.array([x[0]]))
    vio2 = CTCSViolation(g=lambda x, u, node: jnp.array([u[0]]))

    dyn_aug_fn = get_augmented_dynamics(f, [vio1, vio2], slice(None), slice(None))
    A_fn, B_fn = get_jacobians(
        dyn_aug_fn, dyn_nonaug, [vio1, vio2], slice(None), slice(None)
    )

    x = jnp.array([3.0])
    u = jnp.array([4.0])
    A = A_fn(x, u, node=0)
    B = B_fn(x, u, node=0)

    # dyn_aug = [x+2u, x, u] ⇒ A = [1,1,0], B = [2,0,1]
    assert A.shape == (3, 1)
    assert pytest.approx(A[:, 0].tolist()) == [1.0, 1.0, 0.0]
    assert B.shape == (3, 1)
    assert pytest.approx(B[:, 0].tolist()) == [2.0, 0.0, 1.0]


def test_get_jacobians_custom_dyn_grads_no_vio_grads():
    # f(x,u) = [x0 * u0]
    def f(x, u):
        return jnp.array([x[0] * u[0]])

    # supply custom A and B
    A_custom = lambda x, u: jnp.array([[10.0]])
    B_custom = lambda x, u: jnp.array([[20.0]])
    dyn_nonaug = Dynamics(f=f, A=A_custom, B=B_custom)

    # one violation, no custom grads ⇒ uses autodiff for g
    vio = CTCSViolation(g=lambda x, u, node: jnp.array([5.0]))

    dyn_aug_fn = get_augmented_dynamics(f, [vio], slice(None), slice(None))
    A_fn, B_fn = get_jacobians(
        dyn_aug_fn, dyn_nonaug, [vio], slice(None), slice(None)
    )

    x = jnp.array([2.0])
    u = jnp.array([3.0])
    A = A_fn(x, u, node=0)
    B = B_fn(x, u, node=0)

    # A_f = 10, pad zero ⇒ top row [10,0]; g_grad_x=0 ⇒ [0,0]
    assert A.shape == (2, 2)
    assert pytest.approx(A[0, 0]) == 10.0
    assert pytest.approx(A[0, 1]) == 0.0
    assert pytest.approx(A[1, 0]) == 0.0
    assert pytest.approx(A[1, 1]) == 0.0

    # B_f = 20 ⇒ [[20]]; g_grad_u=0 ⇒ [[0]]
    assert B.shape == (2, 1)
    assert pytest.approx(B[0, 0]) == 20.0
    assert pytest.approx(B[1, 0]) == 0.0


def test_get_jacobians_custom_vio_grads_no_dyn_grads():
    # f(x,u) = [x0 + u0]
    def f(x, u):
        return jnp.array([x[0] + u[0]])

    dyn_nonaug = Dynamics(f=f)  # no custom A/B

    # one violation with custom gradients
    vio = CTCSViolation(
        g=lambda x, u, node: jnp.array([x[0] + u[0]]),
        g_grad_x=lambda x, u, node: jnp.array([[7.0]]),
        g_grad_u=lambda x, u, node: jnp.array([[11.0]]),
    )

    dyn_aug_fn = get_augmented_dynamics(f, [vio], slice(None), slice(None))
    A_fn, B_fn = get_jacobians(
        dyn_aug_fn, dyn_nonaug, [vio], slice(None), slice(None)
    )

    x = jnp.array([1.0])
    u = jnp.array([2.0])
    A = A_fn(x, u, node=0)
    B = B_fn(x, u, node=0)

    # A_f = 1 ⇒ top [1,0]; then custom g_grad_x = 7 ⇒ [7,0]
    assert A.shape == (2, 2)
    assert pytest.approx(A[0, 0]) == 1.0
    assert pytest.approx(A[0, 1]) == 0.0
    assert pytest.approx(A[1, 0]) == 7.0
    assert pytest.approx(A[1, 1]) == 0.0

    # B_f = 1 ⇒ [[1]]; custom g_grad_u = 11 ⇒ [[11]]
    assert B.shape == (2, 1)
    assert pytest.approx(B[0, 0]) == 1.0
    assert pytest.approx(B[1, 0]) == 11.0


def test_build_augmented_dynamics_integration():
    # f(x,u) = [2*x0 + 3*u0]
    def f(x, u):
        return jnp.array([2 * x[0] + 3 * u[0]])

    dyn_nonaug = Dynamics(f=f)
    vio = CTCSViolation(g=lambda x, u, node: jnp.array([x[0] + 1.0]))

    idx_x = slice(None)
    idx_u = slice(None)
    dyn_aug = build_augmented_dynamics(dyn_nonaug, [vio], idx_x, idx_u)

    x = jnp.array([4.0])
    u = jnp.array([5.0])

    # f_aug = [8+15=23, g=5] ⇒ [23,5]
    out = dyn_aug.f(x, u, node=0)
    assert out.shape == (2,)
    assert pytest.approx(out[0]) == 23.0
    assert pytest.approx(out[1]) == 5.0

    A = dyn_aug.A(x, u, node=0)
    B = dyn_aug.B(x, u, node=0)

    # ∂/∂x = [2, 1],  ∂/∂u = [3, 0]
    assert A.shape == (2, 1)
    assert pytest.approx(A[:, 0].tolist()) == [2.0, 1.0]
    assert B.shape == (2, 1)
    assert pytest.approx(B[:, 0].tolist()) == [3.0, 0.0]
