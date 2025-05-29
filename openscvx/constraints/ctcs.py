from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Union
import functools
import types

from jax.lax import cond
import jax.numpy as jnp

# TODO: (norrisg) Unclear if should specify behavior for `idx`, `jacfwd` behavior for Jacobians, etc. since that logic is handled elsewhere and could change
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
from openscvx.backend.expr import State, Control, Parameter, Variable
import jax.numpy as jnp
from jax.lax import cond
import functools
import types



@dataclass
class CTCSConstraint:
    func: Callable[..., jnp.ndarray]  # takes (x_expr, u_expr, *param_exprs)
    penalty: Callable[[jnp.ndarray], jnp.ndarray]
    nodes: Optional[Tuple[int, int]] = None
    idx: Optional[int] = None
    grad_f_x: Optional[Callable[..., jnp.ndarray]] = None
    grad_f_u: Optional[Callable[..., jnp.ndarray]] = None

    def __call__(self, x, u, node: int, *params):
        x_expr = x.expr if isinstance(x, (State, Variable)) else x
        u_expr = u.expr if isinstance(u, (Control, Variable)) else u
        param_exprs = [p.expr if isinstance(p, Parameter) else p for p in params]

        return cond(
            jnp.all((self.nodes[0] <= node) & (node < self.nodes[1]))
            if self.nodes is not None else True,
            lambda _: jnp.sum(self.penalty(self.func(x_expr, u_expr, *param_exprs))),
            lambda _: 0.0,
            operand=None,
        )


def ctcs(
    _func: Optional[Callable[..., jnp.ndarray]] = None,
    *,
    penalty: Union[str, Callable[[jnp.ndarray], jnp.ndarray]] = "squared_relu",
    nodes: Optional[Tuple[int, int]] = None,
    idx: Optional[int] = None,
    grad_f_x: Optional[Callable[..., jnp.ndarray]] = None,
    grad_f_u: Optional[Callable[..., jnp.ndarray]] = None,
) -> Union[Callable, CTCSConstraint]:
    if penalty == "squared_relu":
        pen = lambda x: jnp.maximum(0, x) ** 2
    elif penalty == "huber":
        delta = 0.25
        def pen(x): return jnp.where(x < delta, 0.5 * x**2, x - 0.5 * delta)
    elif penalty == "smooth_relu":
        c = 1e-8
        pen = lambda x: (jnp.maximum(0, x) ** 2 + c**2) ** 0.5 - c
    elif callable(penalty):
        pen = penalty
    else:
        raise ValueError(f"Unknown penalty {penalty}")

    def decorator(user_func: Callable[..., jnp.ndarray]) -> CTCSConstraint:
        return CTCSConstraint(
            func=user_func,
            penalty=pen,
            nodes=nodes,
            idx=idx,
            grad_f_x=grad_f_x,
            grad_f_u=grad_f_u,
        )

    return decorator if _func is None else decorator(_func)
