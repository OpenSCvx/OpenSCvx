from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Optional, List
import functools

from jax import jit, vmap, jacfwd
import jax.numpy as jnp


@dataclass
class CTCSConstraint:
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    penalty: Callable[[jnp.ndarray], jnp.ndarray]
    nodes: Sequence[Tuple[int, int]] | None
    idx: int | None

    def __call__(self, x, u):
        # slice x,u at the true-state indices upstream
        return jnp.sum(self.penalty(self.func(x, u)))


def ctcs(
    *,
    penalty: str = "squared_relu",
    nodes: Sequence[Tuple[int, int]] | None = None,
    idx: int | None = None,
):
    """Decorator to mark a function as a 'ctcs' constraint.
    
    Use as:
    @ctcs(nodes=[(0,10)], idx=2, penalty='huber')
    def my_constraint(x,u): ...
    """
    # prepare penalty function once
    if penalty == "squared_relu":

        def pen(x):
            return jnp.maximum(0, x) ** 2

    elif penalty == "huber":
        delta = 0.25

        def pen(x):
            r = jnp.maximum(0, x)
            return jnp.where(r < delta, 0.5 * r**2, r - 0.5 * delta)

    else:
        raise ValueError(penalty)

    def decorator(f: Callable):
        # wrap and return a CTCSConstraint instance
        # preserves f.__name__, __doc__ for introspection
        return CTCSConstraint(
            func=functools.wraps(f)(f),
            penalty=pen,
            nodes=nodes,
            idx=idx,
        )

    return decorator


@dataclass
class NodalConstraint:
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    nodes: Optional[List[int]] = None
    convex: bool = False
    inter_nodal: bool = False

    def __post_init__(self):
        if not self.convex:
        # TODO: (haynec) switch to AOT instead of JIT
            self.g = vmap(jit(self.func), in_axes=(0, 0))
            self.grad_g_x = jit(vmap(jacfwd(self.func, argnums=0), in_axes=(0, 0)))
            self.grad_g_u = jit(vmap(jacfwd(self.func, argnums=1), in_axes=(0, 0)))
        elif self.inter_nodal:
            # single-node but still using JAX
            self.g = jit(self.func)
            self.grad_g_x = jit(jacfwd(self.func, argnums=0))
            self.grad_g_u = jit(jacfwd(self.func, argnums=1))
        # if convex=True and inter_nodal=False, assume an external solver (e.g. CVX) will handle it

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray):
        idxs = self.nodes if self.nodes is not None else list(range(x.shape[0]))
        xs = x[idxs]
        us = u[idxs]
        return self.g(xs, us)


def nodal(
    *,
    nodes: Optional[List[int]] = None,
    convex: bool = False,
    inter_nodal: bool = False,
):
    """Decorator to mark a function as a 'nodal' constraint."""
    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        wrapped = functools.wraps(f)(f)
        return NodalConstraint(
            func=wrapped,
            nodes=nodes,
            convex=convex,
            inter_nodal=inter_nodal,
        )
    return decorator