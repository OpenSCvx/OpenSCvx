from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Union
import functools
import types

from jax.lax import cond
import jax.numpy as jnp


@dataclass
class CTCSConstraint:
    """
    Dataclass for continuous-time constraint satisfaction (CTCS) constraints over a trajectory interval.

    A `CTCSConstraint` wraps a residual function `func(x, u)`, applies a
    pointwise `penalty` to its outputs, and accumulates the penalized sum
    only within a specified node interval [nodes[0], nodes[1]).

    Note: the user is intended to instantiate `CTCSConstraint`s using the `@ctcs` decorator

    Args:
        func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            Function computing constraint residuals g(x, u).
        penalty: Callable[[jnp.ndarray], jnp.ndarray]
            Penalty function applied elementwise to g's output. Used to calculate and penalize
            constraint violation during state augmentation
        nodes: Optional[Tuple[int, int]]
            Half-open interval (start, end) of node indices where this constraint is active.
            If None, the penalty applies at every node.
        idx: Optional[int]
            Optional index used to group CTCS constraints. Used during automatic state augmentation.
            All CTCS constraints with the same index must be active over the same `nodes` interval
        grad_f_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` w.r.t. state `x`, signature (x, u) -> jacobian.
            If None, computed via `jax.jacfwd(func, argnums=0)` during state augmentation.
        grad_f_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` w.r.t. input `u`, signature (x, u) -> jacobian.
            If None, computed via `jax.jacfwd(func, argnums=1)` during state augmentation.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    penalty: Callable[[jnp.ndarray], jnp.ndarray]
    nodes: Optional[Tuple[int, int]] = None
    idx: Optional[int] = None
    grad_f_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    grad_f_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

    def __post_init__(self):
        """
        Adapt user-provided gradients to the three-argument signature (x, u, node).

        If `grad_f_x` or `grad_f_u` are given as functions of (x, u), wrap them
        so they accept the extra `node` argument to match `__call__`.
        """
        if self.grad_f_x is not None:
            _grad_f_x = self.grad_f_x
            self.grad_f_x = lambda x, u, nodes: _grad_f_x(x, u)
        if self.grad_f_u is not None:
            _grad_f_u = self.grad_f_u
            self.grad_f_u = lambda x, u, nodes: _grad_f_u(x, u)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray, node: int):
        """
        Evaluate the penalized constraint at a given node index.

        The penalty is summed only if `node` lies within the active interval.

        Args:
            x: jnp.ndarray
                State vector at this node.
            u: jnp.ndarray
                Input vector at this node.
            node: int
                Trajectory time-step index.

        Returns:
            jnp.ndarray or float:
                The total penalty (sum over selected residuals) if inside interval,
                otherwise zero.
        """
        # check if within [start, end)
        return cond(
            jnp.all((self.nodes[0] <= node) & (node < self.nodes[1])),
            lambda _: jnp.sum(self.penalty(self.func(x, u))),
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
        _func: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            The function to wrap; provided automatically when using bare @ctcs.
        penalty: Union[str, Callable[[jnp.ndarray], jnp.ndarray]]
            Name of a built-in penalty ('squared_relu', 'huber', 'smooth_relu')
            or a custom elementwise penalty function.
        nodes: Optional[Tuple[int, int]]
            Half-open interval (start, end) of node indices where this constraint is active.
            If None, the penalty applies at every node.
        idx: Optional[int]
            Optional index used to group CTCS constraints. Used during automatic state augmentation.
            All CTCS constraints with the same index must be active over the same `nodes` interval
        grad_f_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` w.r.t state `x`.
            If None, computed via `jax.jacfwd(func, argnums=0)` during state augmentation.
        grad_f_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
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
