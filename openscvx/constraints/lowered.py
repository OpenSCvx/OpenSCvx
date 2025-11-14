from dataclasses import dataclass
from typing import Callable, List, Optional

import jax.numpy as jnp


# TODO: (norrisg) remove this as soon as it is not necessary as a drop in replacement for
# `NodalConstraint`
@dataclass
class LoweredNodalConstraint:
    """
    Dataclass to hold a lowered symbolic constraint function and its jacobians.

    This is a simplified drop-in replacement for NodalConstraint that holds
    only the essential lowered JAX functions and their jacobians, without
    the complexity of convex/vectorized flags or post-initialization logic.

    Designed for use with symbolic expressions that have been lowered to JAX
    and will be linearized for sequential convex programming.

    Args:
        func (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
            The lowered constraint function g(x, u, ...params) that returns
            constraint residuals. Should follow g(x, u) <= 0 convention.
            - x: 1D array (state at a single node), shape (n_x,)
            - u: 1D array (control at a single node), shape (n_u,)
            - Additional parameters: passed as keyword arguments

        grad_g_x (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of g w.r.t. x. If None, should be computed using jax.jacfwd.

        grad_g_u (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of g w.r.t. u. If None, should be computed using jax.jacfwd.

        nodes (Optional[List[int]]): List of node indices where this constraint applies.
            Set after lowering from NodalConstraint.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    grad_g_x: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    grad_g_u: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    nodes: Optional[List[int]] = None
