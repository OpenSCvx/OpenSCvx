from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from jax.lax import cond
import functools
import types

from openscvx.backend.state import State, Variable
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter

# TODO: (norrisg) Unclear if should specify behavior for `idx`, `jacfwd` behavior for Jacobians, etc. since that logic is handled elsewhere and could change

@dataclass
class CTCSConstraint:
    """
    Dataclass for continuous-time constraint satisfaction (CTCS) constraints over a trajectory interval.
    A `CTCSConstraint` wraps a residual function `func(x, u)`, applies a
    pointwise `penalty` to its outputs, and accumulates the penalized sum
    only within a specified node interval [nodes[0], nodes[1]).
    Note: the user is intended to instantiate `CTCSConstraint`s using the `@ctcs` decorator

    Args:
        func (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): 
            Function computing constraint residuals g(x, u).
        penalty (Callable[[jnp.ndarray], jnp.ndarray]): 
            Penalty function applied elementwise to g's output. Used to calculate and penalize
            constraint violation during state augmentation
        nodes (Optional[Tuple[int, int]]):
            Half-open interval (start, end) of node indices where this constraint is active.
            If None, the penalty applies at every node.
        idx (Optional[int]): 
            Optional index used to group CTCS constraints. Used during automatic state augmentation.
            All CTCS constraints with the same index must be active over the same `nodes` interval
        grad_f_x (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            User-supplied gradient of `func` w.r.t. state `x`, signature (x, u) -> jacobian.
            If None, computed via `jax.jacfwd(func, argnums=0)` during state augmentation.
        grad_f_u (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            User-supplied gradient of `func` w.r.t. input `u`, signature (x, u) -> jacobian.
            If None, computed via `jax.jacfwd(func, argnums=1)` during state augmentation.
    """
    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # takes (x_expr, u_expr, *param_exprs)
    penalty: Callable[[jnp.ndarray], jnp.ndarray]
    nodes: Optional[Tuple[int, int]] = None
    idx: Optional[int] = None
    grad_f_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    grad_f_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

    def __call__(self, x, u, node: int, *params):
        """
        Evaluate the penalized constraint at a given node index.
        The penalty is summed only if `node` lies within the active interval.
        Args:
            x (jnp.ndarray): State vector at this node.
            u (jnp.ndarray): Input vector at this node.
            node (int): Trajectory time-step index.
            *params: parameters
        Returns:
            jnp.ndarray or float:
                The total penalty (sum over selected residuals) if inside interval,
                otherwise zero.
        """
        x_expr = x.expr if isinstance(x, (State, Variable)) else x
        u_expr = u.expr if isinstance(u, (Control, Variable)) else u
        param_exprs = [p.expr if isinstance(p, Parameter) else p for p in params]

        # check if within [start, end)
        return cond(
            jnp.all((self.nodes[0] <= node) & (node < self.nodes[1]))
            if self.nodes is not None else True,
            lambda _: jnp.sum(self.penalty(self.func(x_expr, u_expr, *param_exprs))),
            lambda _: 0.0,
            operand=None,
        )


def ctcs(
    _func: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    *,
    penalty: Union[str, Callable[[jnp.ndarray], jnp.ndarray]] = "squared_relu",
    nodes: Optional[Tuple[int, int]] = None,
    idx: Optional[int] = None,
    grad_f_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    grad_f_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
) -> Union[Callable, CTCSConstraint]:
    """
    Decorator to build a CTCSConstraint from a raw constraint function.

    Supports built-in penalties by name or a custom penalty function.

    Usage examples:

    ```python
    @ctcs
    def g(x, u):
        return jnp.maximum(0, x - 1)
    ```
    ```python
    @ctcs("huber", nodes=[(0, 10)], idx=2)
    def g2(x, u):
        return jnp.sin(x) + u
    ```

    Or can directly wrap a function if a more lambda-function interface is desired:

    ```python
    constraint = ctcs(lambda x, u: ...)
    ```

    Args:
        _func (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            The function to wrap; provided automatically when using bare @ctcs.
        penalty (Union[str, Callable[[jnp.ndarray], jnp.ndarray]]):
            Name of a built-in penalty ('squared_relu', 'huber', 'smooth_relu')
            or a custom elementwise penalty function.
        nodes (Optional[Tuple[int, int]]):
            Half-open interval (start, end) of node indices where this constraint is active.
            If None, the penalty applies at every node.
        idx (Optional[int]):
            Optional index used to group CTCS constraints. Used during automatic state augmentation.
            All CTCS constraints with the same index must be active over the same `nodes` interval
        grad_f_x (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            User-supplied gradient of `func` w.r.t state `x`.
            If None, computed via `jax.jacfwd(func, argnums=0)` during state augmentation.
        grad_f_u (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            User-supplied gradient of `func` w.r.t input `u`.
            If None, computed via `jax.jacfwd(func, argnums=1)` during state augmentation.

    Returns:
        Union[Callable, CTCSConstraint]
            A decorator if called without a function, or a CTCSConstraint instance
            when applied to a function.

    Raises:
        ValueError: If `penalty` string is not one of the supported names.
    """
    # prepare penalty function once
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

    def decorator(user_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> CTCSConstraint:
        # wrap so name, doc, signature stay on f
        wrapped = functools.wraps(user_func)(user_func)
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
