from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp

from openscvx.constraints.violation import CTCSViolation
from openscvx.dynamics import Dynamics

def build_augmented_dynamics(
    dynamics_non_augmented: Dynamics,
    violations: List[CTCSViolation],
    idx_x_true: slice,
    idx_u_true: slice,
) -> Dynamics:
    dynamics_augmented = Dynamics(
        f=get_augmented_dynamics(
            dynamics_non_augmented.f, violations, idx_x_true, idx_u_true
        ),
    )
    A, B = get_jacobians(
        dynamics_augmented.f, dynamics_non_augmented, violations, idx_x_true, idx_u_true
    )
    dynamics_augmented.A = A
    dynamics_augmented.B = B
    return dynamics_augmented


def get_augmented_dynamics(
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    violations: List[CTCSViolation],
    idx_x_true: slice,
    idx_u_true: slice,
) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    def dynamics_augmented(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        x_dot = dynamics(x[idx_x_true], u[idx_u_true])

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for v in violations:
            x_dot = jnp.hstack([x_dot, v.g(x[idx_x_true], u[idx_u_true], node)])

        return x_dot

    return dynamics_augmented


def get_jacobians(
    dyn_augmented: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    dynamics_non_augmented: Dynamics,
    violations: List[CTCSViolation],
    idx_x_true: slice,
    idx_u_true: slice,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
]:
    # 1) Early return if absolutely no custom grads anywhere
    no_dyn_grads = dynamics_non_augmented.A is None and dynamics_non_augmented.B is None
    no_vio_grads = all(v.g_grad_x is None and v.g_grad_u is None for v in violations)

    if no_dyn_grads and no_vio_grads:
        return (
            jax.jacfwd(dyn_augmented, argnums=0),
            jax.jacfwd(dyn_augmented, argnums=1),
        )

    # 2) Build the *true‐state* Jacobians of f(x_true,u_true)
    f_fn = dynamics_non_augmented.f
    if dynamics_non_augmented.A is None:
        A_f = lambda x_true, u_true: jax.jacfwd(f_fn, argnums=0)(x_true, u_true)
    else:
        A_f = dynamics_non_augmented.A

    if dynamics_non_augmented.B is None:
        B_f = lambda x_true, u_true: jax.jacfwd(f_fn, argnums=1)(x_true, u_true)
    else:
        B_f = dynamics_non_augmented.B

    # 3) Build per-violation gradients
    def make_violation_grad_x(i: int) -> Callable:
        viol = violations[i]
        # use user‐provided if present, otherwise autodiff viol.g in argnum=0
        return viol.g_grad_x or jax.jacfwd(viol.g, argnums=0)

    def make_violation_grad_u(i: int) -> Callable:
        viol = violations[i]
        # use user‐provided if present, otherwise autodiff viol.g in argnum=0
        return viol.g_grad_u or jax.jacfwd(viol.g, argnums=1)

    # 4) Assemble A_aug, B_aug
    def A(x_aug, u_aug, node):
        # dynamics block + zero‐pad
        Af = A_f(x_aug[idx_x_true], u_aug[idx_u_true])  # (n_f, n_x_true)
        nv = sum(
            v.g(x_aug[idx_x_true], u_aug[idx_u_true], node).shape[0] for v in violations
        )  # total # rows of violations
        zero_pad = jnp.zeros((Af.shape[0], nv))  # (n_f, n_v)
        top = jnp.hstack([Af, zero_pad])  # (n_f, n_x_true + n_v)

        # violation blocks
        rows = [top]
        for i in range(len(violations)):
            dx_i = make_violation_grad_x(i)(
                x_aug[idx_x_true], u_aug[idx_u_true], node
            )  # (n_gi, n_x_true)
            pad_i = jnp.zeros((dx_i.shape[0], nv))  # (n_gi, n_v)
            rows.append(jnp.hstack([dx_i, pad_i]))  # (n_gi, n_x_true + n_v)

        return jnp.vstack(rows)

    def B(x_aug, u_aug, node):
        Bf = B_f(x_aug[idx_x_true], u_aug[idx_u_true])  # (n_f, n_u_true)
        rows = [Bf]
        for i in range(len(violations)):
            du_i = make_violation_grad_u(i)(
                x_aug[idx_x_true], u_aug[idx_u_true], node
            )  # (n_gi, n_u_true)
            rows.append(du_i)

        return jnp.vstack(rows)

    return A, B
