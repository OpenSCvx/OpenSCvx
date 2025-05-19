from dataclasses import dataclass
from typing import Callable, Optional, List, Union

import jax.numpy as jnp
from jax import vmap, jacfwd


@dataclass
class NodalConstraint:
    """
    Encapsulates a constraint function applied at specific trajectory nodes.

    A `NodalConstraint` wraps a function `g(x, u)` that computes constraint residuals
    for given state `x` and input `u`. It can optionally apply only at
    a subset of trajectory nodes, support vectorized evaluation across nodes,
    and integrate with convex solvers when `convex=True`.

    Args:
        func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            User-supplied constraint function g(x, u).
        nodes: Optional[List[int]]
            Specific node indices where this constraint applies. If None, applies at all nodes.
        convex: bool
            If True, indicates the constraint should be handled by an external
            convex solver (e.g., CVX).
            Note that the constraint must be defined using cvxpy if this flag is set
        vectorized: bool
            If False, automatically vectorizes `func` and its jacobians over
            the node dimension using `jax.vmap`.
        grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` wrt `x`. If None, computed via
            `jax.jacfwd(func, argnums=0)`.
        grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` wrt `u`. If None, computed via
            `jax.jacfwd(func, argnums=1)`.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    nodes: Optional[List[int]] = None
    convex: bool = False
    vectorized: bool = False
    grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

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
    _func: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    *,
    nodes: Optional[List[int]] = None,
    convex: bool = False,
    vectorized: bool = False,
    grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
) -> Union[Callable, NodalConstraint]:
    """
    Decorator to build a `NodalConstraint` from a constraint function.

    Usage examples:

    ```python
    @nodal
    def g(x, u):
        ...
    ```
    ```python
    @nodal(nodes=[0, -1], convex=True, vectorized=False)
    def g(x, u):
        ...
    ```

    Or can directly wrap a function if a more lambda-function interface is desired:

    ```python
    constraint = nodal(lambda x, u: ...)
    ```

    Args:
        _func: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            The function to wrap; populated automatically when using bare @nodal.
        nodes: Optional[List[int]]
            Node indices where the constraint applies; default None applies to all.
        convex: bool
            If True, skip automatic jacobian/vectorization and assume external solver.
            Note that the constraint must be defined using cvxpy if this flag is set
        vectorized: bool
            If False, auto-vectorize over nodes using `jax.vmap`.
        grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` wrt `x`. If None, computed via
            `jax.jacfwd(func, argnums=0)`.
        grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]
            User-supplied gradient of `func` wrt `u`. If None, computed via
            `jax.jacfwd(func, argnums=1)`.

    Returns:
        Union[Callable, NodalConstraint]
            A decorator if called without a function, or a `NodalConstraint` dataclass
            instance bundling nodal constraint function and Jacobians when applied to a function
    """

    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        return NodalConstraint(
            func=f,
            nodes=nodes,
            convex=convex,
            vectorized=vectorized,
            grad_g_x=grad_g_x,
            grad_g_u=grad_g_u,
        )

    return decorator if _func is None else decorator(_func)
