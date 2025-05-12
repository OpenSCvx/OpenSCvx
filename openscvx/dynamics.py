from dataclasses import dataclass
from typing import Callable, Optional

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
    dynamics: Dynamics, g_funcs: list[CTCSConstraint], idx_x_true: slice, idx_u_true: slice
) -> callable:
    def dynamics_augmented(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        x_dot = dynamics(x[idx_x_true], u[idx_u_true])

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for g in g_funcs:
            x_dot = jnp.hstack([x_dot, g(x[idx_x_true], u[idx_u_true], node)])

        return x_dot

    return dynamics_augmented


def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B
