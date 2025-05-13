from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import jax
import jax.numpy as jnp

from openscvx.constraints.violation import CTCSViolation


@dataclass
class Dynamics:
    f: Callable
    A: Optional[Callable] = None
    B: Optional[Callable] = None


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
    A, B = get_jacobians(dynamics_augmented, dynamics_non_augmented, violations)
    dynamics_augmented.A = A
    dynamics_augmented.B = B
    return dynamics_augmented


def get_augmented_dynamics(
    dynamics: callable,
    violations: List[CTCSViolation],
    idx_x_true: slice,
    idx_u_true: slice,
) -> callable:
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
    violations: Optional[List[CTCSViolation]] = None,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
]:
    # Dynamics block — either user-supplied or autodiff
    if dynamics_non_augmented.A:
        A_dyn_fn = dynamics_non_augmented.A
    else:
        A_dyn_fn = lambda x, u, node: jax.jacfwd(
            lambda xx, uu: dyn_augmented(xx, uu, node), argnums=0
        )(x, u)

    if dynamics_non_augmented.B:
        B_dyn_fn = dynamics_non_augmented.B
    else:
        B_dyn_fn = lambda x, u, node: jax.jacfwd(
            lambda xx, uu: dyn_augmented(xx, uu, node), argnums=1
        )(x, u)

    # Handle violations
    violations = violations or []
    n_v = len(violations)

    def make_violation_grad_x(i: int) -> Callable:
        viol = violations[i]
        # use user‐provided if present, otherwise autodiff viol.g in argnum=0
        return viol.g_grad_x or jax.jacfwd(viol.g, argnums=0)

    def make_violation_grad_u(i: int) -> Callable:
        viol = violations[i]
        # use user‐provided if present, otherwise autodiff viol.g in argnum=1
        return viol.g_grad_u or jax.jacfwd(viol.g, argnums=1)

    # Assemble full A, B
    def A(x, u, node):
        rows = [A_dyn_fn(x, u, node)]
        for i in range(n_v):
            rows.append(make_violation_grad_x(i)(x, u, node))
        return jnp.vstack(rows)

    def B(x, u, node):
        rows = [B_dyn_fn(x, u, node)]
        for i in range(n_v):
            rows.append(make_violation_grad_u(i)(x, u, node))
        return jnp.vstack(rows)

    return A, B
