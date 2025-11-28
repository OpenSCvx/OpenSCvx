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
    - Cross-node: f(X, U, params) -> (M,) residuals (already handles full trajectory)

    Attributes:
        func: Function (X, U, params) -> (M,) residuals
            where X: (N, n_x), U: (N, n_u), M = number of evaluation points
            Returns constraint residuals following g(X, U) <= 0 convention
        grad_g_X: Function (X, U, params) -> (M, N, n_x) Jacobian wrt full state trajectory
            This is typically sparse - most constraints only couple nearby nodes
        grad_g_U: Function (X, U, params) -> (M, N, n_u) Jacobian wrt full control trajectory
            Often zero or very sparse for cross-node state constraints
        eval_nodes: List of node indices where constraint is evaluated (length M)

    Example:
        For rate constraint (x[k] - x[k-1] <= r) at k in 1..N-1:

            func(X, U, params) -> (N-1,) residuals
            grad_g_X(X, U, params) -> (N-1, N, n_x) sparse Jacobian
                where grad_g_X[i, i, :] = ∂g_i/∂x[i] (derivative wrt current)
                and grad_g_X[i, i-1, :] = ∂g_i/∂x[i-1] (derivative wrt previous)
                all other entries are zero
            eval_nodes = [1, 2, 3, ..., N-1]

    Performance Note - Dense Jacobian Storage:
        The Jacobian matrices grad_g_X and grad_g_U are stored as DENSE arrays with
        shape (M, N, n_x) and (M, N, n_u), but most cross-node constraints only
        couple a small number of nearby nodes, making these matrices extremely sparse.

        For example, a rate limit constraint x[k] - x[k-1] <= r only has non-zero
        Jacobian entries at positions [i, k, :] and [i, k-1, :] for each constraint i.
        All other N-2 entries per row are zero but still stored in memory.

        Memory impact for large problems:
        - A single constraint with M=50 evaluations, N=100 nodes, n_x=10 states
          requires ~400KB for grad_g_X alone (instead of ~8KB if sparse)
        - Multiple cross-node constraints multiply this overhead
        - May cause issues for N > 100 with many constraints

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
    eval_nodes: List[int]
