from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Optional
import functools
import types

from jax.lax import cond
import jax.numpy as jnp


@dataclass
class CTCSConstraint:
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    penalty: Callable[[jnp.ndarray], jnp.ndarray]
    nodes: Optional[Tuple[int, int]] = None
    idx: Optional[int] = None
    grad_f_x: Optional[Callable] = None
    grad_f_u: Optional[Callable] = None

    def __post_init__(self):
        if self.grad_f_x is not None:
            _grad_f_x = self.grad_f_x
            self.grad_f_x = lambda x, u, nodes: _grad_f_x(x, u)
        if self.grad_f_u is not None:
            _grad_f_u = self.grad_f_u
            self.grad_f_u = lambda x, u, nodes: _grad_f_u(x, u)

    def __call__(self, x, u, node):
        return cond(
            jnp.all((self.nodes[0] <= node) & (node < self.nodes[1])),
            lambda _: jnp.sum(self.penalty(self.func(x, u))),
            lambda _: 0.0,
            operand=None,
        )


def ctcs(
    _func=None,
    *,
    penalty: str = "squared_relu",
    nodes: Optional[Sequence[Tuple[int, int]]] = None,
    idx: Optional[int] = None,
    grad_f_x: Optional[Callable] = None,
    grad_f_u: Optional[Callable] = None,
):
    """Decorator to mark a function as a 'ctcs' constraint.

    Use as:
    @ctcs(nodes=[(0,10)], idx=2, penalty='huber')
    def my_constraint(x,u): ...
    """
    # prepare penalty function once
    if penalty == "squared_relu":
        pen = lambda x: jnp.maximum(0, x) ** 2
    elif penalty == "huber":
        delta = 0.25

        def pen(x):
            r = jnp.maximum(0, x)
            return jnp.where(r < delta, 0.5 * r**2, r - 0.5 * delta)
    elif penalty == "smooth_relu":
        c = 1e-8
        pen = lambda x: (jnp.maximum(0, x) ** 2 + c**2) ** 0.5 - c
    elif isinstance(penalty, types.LambdaType):
        pen = penalty
    else:
        raise ValueError(f"Unknown penalty {penalty}")

    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        # wrap so name, doc, signature stay on f
        wrapped = functools.wraps(f)(f)
        return CTCSConstraint(
            func=wrapped,
            penalty=pen,
            nodes=nodes,
            idx=idx,
            grad_f_x=grad_f_x,
            grad_f_u=grad_f_u,
        )

    # if called as @ctcs or @ctcs(...), _func will be None and we return decorator
    if _func is None:
        return decorator
    # if called as ctcs(func), we immediately decorate
    else:
        return decorator(_func)
