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

    def __call__(self, x, u):
        return self.f(x, u)


def get_augmented_dynamics(
    dynamics: Dynamics,
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
    A_non_aug: Optional[Callable] = None,
    B_non_aug: Optional[Callable] = None,
    g_funcs: Optional[List[Callable]] = None,
    g_grads_x: Optional[List[Optional[Callable]]] = None,
    g_grads_u: Optional[List[Optional[Callable]]] = None,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
]:
    # Dynamics block
    if A_non_aug:
        A_dyn_fn = A_non_aug
    else:
        A_dyn_fn = lambda x, u, node: jax.jacfwd(
            lambda xx, uu: dyn_augmented(xx, uu, node), argnums=0
        )(x, u)

    if B_non_aug:
        B_dyn_fn = B_non_aug
    else:
        B_dyn_fn = lambda x, u, node: jax.jacfwd(
            lambda xx, uu: dyn_augmented(xx, uu, node), argnums=1
        )(x, u)

    # Per-constraint rows
    n_g = len(g_funcs or [])

    def make_gx(i):
        if g_grads_x and g_grads_x[i] is not None:
            return g_grads_x[i]
        else:
            return lambda x, u, node: jax.jacfwd(
                lambda xx, uu: g_funcs[i](xx, uu, node), argnums=0
            )(x, u, node)

    def make_gu(i):
        if g_grads_u and g_grads_u[i] is not None:
            return g_grads_u[i]
        else:
            return lambda x, u, node: jax.jacfwd(
                lambda xx, uu: g_funcs[i](xx, uu, node), argnums=1
            )(x, u, node)

    # Assemble full A, B
    def A(x, u, node):
        rows = [A_dyn_fn(x, u, node)]
        for i in range(n_g):
            rows.append(make_gx(i)(x, u, node))
        return jnp.vstack(rows)

    def B(x, u, node):
        rows = [B_dyn_fn(x, u, node)]
        for i in range(n_g):
            rows.append(make_gu(i)(x, u, node))
        return jnp.vstack(rows)

    return A, B
