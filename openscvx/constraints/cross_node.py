"""Cross-node constraint handling for OpenSCVX.

This module provides lowered constraint representations for constraints that
relate variables across different trajectory nodes, such as rate limits and
multi-step dependencies.
"""

from dataclasses import dataclass
from typing import Callable, List

import jax.numpy as jnp


@dataclass
class CrossNodeConstraintLowered:
    """Lowered cross-node constraint with trajectory-level evaluation.

    Unlike regular LoweredNodalConstraint which operates on single-node vectors
    and is vmapped across the trajectory, CrossNodeConstraintLowered operates
    on full trajectory arrays to relate multiple nodes simultaneously.

    This is necessary for constraints like:
    - Rate limits: x[k] - x[k-1] <= max_rate
    - Multi-step dependencies: x[k] = 2*x[k-1] - x[k-2]
    - Periodic boundaries: x[0] = x[N-1]

    The function signatures differ from LoweredNodalConstraint:
    - Regular: f(x, u, node, params) -> scalar (vmapped to handle (N, n_x))
    - Cross-node: f(X, U, params) -> scalar (single constraint with fixed node indices)

    Attributes:
        func: Function (X, U, params) -> scalar residual
            where X: (N, n_x), U: (N, n_u)
            Returns constraint residual following g(X, U) <= 0 convention
            The constraint references fixed trajectory nodes (e.g., X[5] - X[4])
        grad_g_X: Function (X, U, params) -> (N, n_x) Jacobian wrt full state trajectory
            This is typically sparse - most constraints only couple nearby nodes
        grad_g_U: Function (X, U, params) -> (N, n_u) Jacobian wrt full control trajectory
            Often zero or very sparse for cross-node state constraints

    Example:
        For rate constraint x[5] - x[4] <= r:

            func(X, U, params) -> scalar residual
            grad_g_X(X, U, params) -> (N, n_x) sparse Jacobian
                where grad_g_X[5, :] = ∂g/∂x[5] (derivative wrt node 5)
                and grad_g_X[4, :] = ∂g/∂x[4] (derivative wrt node 4)
                all other entries are zero

    Performance Note - Dense Jacobian Storage:
        The Jacobian matrices grad_g_X and grad_g_U are stored as DENSE arrays with
        shape (N, n_x) and (N, n_u), but most cross-node constraints only couple a
        small number of nearby nodes, making these matrices extremely sparse.

        For example, a rate limit constraint x[k] - x[k-1] <= r only has non-zero
        Jacobian entries at positions [k, :] and [k-1, :]. All other N-2 rows are
        zero but still stored in memory.

        Memory impact for large problems:
        - A single constraint with N=100 nodes, n_x=10 states requires ~8KB for
          grad_g_X (compared to ~160 bytes if sparse with 2 non-zero rows)
        - Multiple cross-node constraints multiply this overhead
        - May cause issues for N > 1000 with many constraints

        Performance impact:
        - Slower autodiff (computes many zero gradients)
        - Inefficient constraint linearization in the SCP solver
        - Potential GPU memory limitations for very large problems

        The current implementation prioritizes simplicity and compatibility with
        JAX's autodiff over memory efficiency. Future versions may support sparse
        Jacobian formats (COO, CSR, or custom sparse representations) if this
        becomes a bottleneck in practice.
    """

    func: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
    grad_g_X: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
    grad_g_U: Callable[[jnp.ndarray, jnp.ndarray, dict], jnp.ndarray]
