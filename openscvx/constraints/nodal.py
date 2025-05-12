from dataclasses import dataclass
from typing import Callable, Optional, List

import jax.numpy as jnp
from jax import jit, vmap, jacfwd


@dataclass
class NodalConstraint:
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    nodes: Optional[List[int]] = None
    convex: bool = False
    vectorized: bool = False
    grad_g_x: Optional[Callable] = None
    grad_g_u: Optional[Callable] = None

    def __post_init__(self):
        if not self.convex:
            # single-node but still using JAX
            self.g = self.func
            if self.grad_g_x is None:
                self.grad_g_x = jacfwd(self.func, argnums=0)
            if self.grad_g_u is None:
                self.grad_g_u = jacfwd(self.func, argnums=1)
            if not self.vectorized:
                self.g = vmap(self.g, in_axes=(0, 0))
                self.grad_g_x = vmap(self.grad_g_x, in_axes=(0, 0))
                self.grad_g_u = vmap(self.grad_g_u, in_axes=(0, 0))
        # if convex=True assume an external solver (e.g. CVX) will handle it

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray):
        return self.func(x, u)


def nodal(
    _func=None,
    *,
    nodes: Optional[List[int]] = None,
    convex: bool = False,
    vectorized: bool = False,
    grad_g_x: Optional[Callable] = None,
    grad_g_u: Optional[Callable] = None,
):
    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        return NodalConstraint(
            func=f,  # no wraps, just keep the original
            nodes=nodes,
            convex=convex,
            vectorized=vectorized,
            grad_g_x=grad_g_x,
            grad_g_u=grad_g_u,
        )

    return decorator if _func is None else decorator(_func)
