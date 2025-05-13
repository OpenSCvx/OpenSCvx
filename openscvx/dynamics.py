from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple

import jax
import jax.numpy as jnp

from openscvx.constraints.ctcs import CTCSConstraint


@dataclass
class Dynamics:
    f: Callable
    A: Optional[Callable] = None
    B: Optional[Callable] = None
    
@dataclass
class CTCSViolation:
    g: Callable
    g_grad_x: Optional[Callable] = None
    g_grad_u: Optional[Callable] = None


def get_augmented_dynamics(
    dynamics: callable,
    g_funcs: list[CTCSConstraint],
    idx_x_true: slice,
    idx_u_true: slice,
) -> callable:
    def dynamics_augmented(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        x_dot = dynamics(x[idx_x_true], u[idx_u_true])

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for g in g_funcs:
            x_dot = jnp.hstack([x_dot, g(x[idx_x_true], u[idx_u_true], node)])

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
        if viol.g_grad_x is not None:
            return viol.g_grad_x
        else:
            return lambda x, u, node: jax.jacfwd(
                lambda xx, uu: viol.g(xx, uu, node), argnums=0
            )(x, u, node)

    def make_violation_grad_u(i: int) -> Callable:
        viol = violations[i]
        if viol.g_grad_u is not None:
            return viol.g_grad_u
        else:
            return lambda x, u, node: jax.jacfwd(
                lambda xx, uu: viol.g(xx, uu, node), argnums=1
            )(x, u, node)

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
