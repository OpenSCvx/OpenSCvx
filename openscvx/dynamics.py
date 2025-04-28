from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp


@dataclass
class Dynamics:
    def __init__(
        self,
        dynamics_augmented: callable,
    ):
        # Dynamics Functions
        self.state_dot = jax.vmap(dynamics_augmented)
        A_uncompiled, B_uncompiled = get_jacobians(dynamics_augmented)
        self.A = jax.jit(jax.vmap(A_uncompiled, in_axes=(0, 0)))
        self.B = jax.jit(jax.vmap(B_uncompiled, in_axes=(0, 0)))

def get_augmented_dynamics(dynamics: callable, g_func: callable):
    def dynamics_augmented(x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) handle varying lengths of x and u due to augmentation more elegantly
        x_dot = dynamics(x[:-1], u)
        t_dot = 1
        y_dot = g_func(x, u)
        return jnp.hstack([x_dot, t_dot, y_dot])
    return dynamics_augmented

def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B
