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

    **Expected input types:**

    | Case                        | x, u type/shape                                 |
    |-----------------------------|-------------------------------------------------|
    | convex=False, vectorized=False | 1D arrays, shape (n_x,), (n_u,) (single node)   |
    | convex=False, vectorized=True  | 2D arrays, shape (N, n_x), (N, n_u) (all nodes) |
    | convex=True, vectorized=False  | list of cvxpy variables, one per node           |
    | convex=True, vectorized=True   | list of cvxpy variables, one per node           |

    **Expected output:**

    | Case                        | Output type                                      |
    |-----------------------------|--------------------------------------------------|
    | convex=False, vectorized=False | float (single node)                              |
    | convex=False, vectorized=True  | float array (per node)                           |
    | convex=True, vectorized=False  | cvxpy expression (single node)                   |
    | convex=True, vectorized=True   | list of cvxpy expressions (one per node)         |

    Nonconvex examples:
    
    ```python
    @nodal
    def g(x_, u_):
        return 1 - x_[0] <= 0
    ```
    ```python
    @nodal(nodes=[0, 3])
    def g(x_, u_):
        return jnp.linalg.norm(x_) - 1.0 
    ```

    Or can directly wrap a function if a more lambda-function interface is desired:

    ```python
    constraint = nodal(lambda x_, u_: 1 - x_[0])
    ```

    Convex Examples:

    ```python
    @nodal(convex=True)
    def g(x_, u_):
        return cp.norm(x_) <= 1.0  # cvxpy expression following DPP rules
    ```

    Args:
        func (Callable):
            The user-supplied constraint function. The expected input and output types depend on the values of `convex` and `vectorized`:

            | Case                          | x, u type/shape                                   | Output type                                      |
            |-------------------------------|---------------------------------------------------|--------------------------------------------------|
            | convex=False, vectorized=False | 1D arrays, shape (n_x,), (n_u,) (single node)     | float (single node)                              |
            | convex=False, vectorized=True  | 2D arrays, shape (N, n_x), (N, n_u) (all nodes)   | float array (per node)                           |
            | convex=True, vectorized=False  | list of cvxpy variables, one per node             | cvxpy expression (single node)                   |
            | convex=True, vectorized=True   | list of cvxpy variables, one per node             | list of cvxpy expressions (one per node)         |

            Additional parameters: always passed as keyword arguments with names matching the parameter name plus an underscore (e.g., `g_` for `Parameter('g')`).
            For nonconvex constraints, the function should return constraint residuals (g(x,u) <= 0). For convex constraints, the function should return a cvxpy expression.
        nodes (Optional[List[int]]):
            Specific node indices where this constraint applies. If None, applies at all nodes.
        convex (bool):
            If True, the provided cvxpy.expression is directly passed to the cvxpy.problem.
        vectorized (bool):
            If False, automatically vectorizes `func` and its jacobians over
            the node dimension using `jax.vmap`. If True, assumes `func` already
            handles vectorization.
        grad_g_x (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            User-supplied gradient of `func` wrt `x`. If None, computed via
            `jax.jacfwd(func, argnums=0)`.
        grad_g_u (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]): 
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
        """Initialize gradients and vectorization after instance creation.
        
        If the constraint is not convex, this method:
        1. Sets up the constraint function
        2. Computes gradients if not provided
        3. Vectorizes the functions if needed
        """
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
        """Evaluate the constraint function at the given state and control.
        
        Args:
            x (jnp.ndarray): The state vector.
            u (jnp.ndarray): The control vector.
            
        Returns:
            jnp.ndarray: The constraint violation values.
        """
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
        _func (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            The function to wrap; populated automatically when using bare @nodal.
            When `convex=False`, this is a standard function g(x, u) that should
            return constraint residuals (g(x,u) <= 0). When `convex=True`, this
            must be a cvxpy expression following the DPP ruleset.
        nodes (Optional[List[int]]):
            Node indices where the constraint applies; default None applies to all.
        convex (bool):
            If True, the provided cvxpy.expression is directly passed to the cvxpy.problem.
        vectorized (bool):
            If False, auto-vectorize over nodes using `jax.vmap`. If True, assumes
            the function already handles vectorization.
    """
    def decorator(f: Callable):
        return NodalConstraint(
            func=f,
            nodes=nodes,
            convex=convex,
            vectorized=vectorized,
            grad_g_x=grad_g_x,
            grad_g_u=grad_g_u,
        )

    if _func is None:
        # Called with arguments, e.g., @nodal(nodes=[0, 1])
        return decorator
    else:
        # Called as a bare decorator, e.g., @nodal
        return decorator(_func)