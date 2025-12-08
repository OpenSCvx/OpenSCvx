"""Convex subproblem solvers for trajectory optimization.

This module provides implementations of convex subproblem solvers used within
SCvX algorithms. At each iteration of a successive convexification algorithm,
the non-convex problem is approximated by a convex subproblem, which is then
solved using one of these solver backends.

Current Implementations:
    CVXPy Solver: The default solver backend using CVXPy's modeling language
        with support for multiple backend solvers (ECOS, SCS, MOSEK, etc.).
        Includes optional code generation via cvxpygen for improved performance.

Core Functions:
    optimal_control_problem: Builds and returns a CVXPy Problem object from
        a LoweredProblem, incorporating dynamics constraints, trust regions,
        virtual controls, and user-defined constraints.

Planned Architecture (ABC-based):
    A base class will be introduced to enable pluggable solver implementations.
    Future solvers will implement the ConvexSolver interface:

    .. code-block:: python

        # solvers/base.py (planned):
        class ConvexSolver(ABC):
            @abstractmethod
            def build_subproblem(self, state: SolverState, lowered: LoweredProblem):
                '''Build the convex subproblem from current state.'''
                ...

            @abstractmethod
            def solve(self) -> OptimizationResults:
                '''Solve the convex subproblem and return results.'''
                ...

    This will enable users to implement custom solver backends such as:
    - Direct Clarabel solver (Rust-based, GPU-capable)
    - QPAX (JAX-based QP solver for end-to-end differentiability)
    - OSQP direct interface (specialized for QP structure)
    - Custom embedded solvers for real-time applications
    - Research solvers with specialized structure exploitation

Note:
    The solver backend is independent of the SCvX algorithm choice. Users will
    be able to mix and match algorithms (PTR, GuSTO, etc.) with solvers (CVXPy,
    Clarabel, etc.) based on their performance and capability requirements.
"""

from .cvxpy import optimal_control_problem

__all__ = [
    "optimal_control_problem",
]
