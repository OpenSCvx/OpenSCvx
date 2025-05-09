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

    def __post_init__(self):
        if not self.convex:
            # TODO: (haynec) switch to AOT instead of JIT
            if self.vectorized:
                # single-node but still using JAX
                self.g = jit(self.func)
                self.grad_g_x = jit(jacfwd(self.func, argnums=0))
                self.grad_g_u = jit(jacfwd(self.func, argnums=1))
            else:
                self.g = vmap(jit(self.func), in_axes=(0, 0))
                self.grad_g_x = jit(vmap(jacfwd(self.func, argnums=0), in_axes=(0, 0)))
                self.grad_g_u = jit(vmap(jacfwd(self.func, argnums=1), in_axes=(0, 0)))
        # if convex=True and inter_nodal=False, assume an external solver (e.g. CVX) will handle it

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray):
        return self.func(x, u)


def nodal(
    _func=None,
    *,
    nodes: Optional[List[int]] = None,
    convex: bool = False,
    vectorized: bool = False,
):
    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        return NodalConstraint(
            func=f,  # no wraps, just keep the original
            nodes=nodes,
            convex=convex,
            vectorized=vectorized,
        )

    return decorator if _func is None else decorator(_func)
