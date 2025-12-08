"""Convex subproblem solvers for trajectory optimization.

This module provides implementations of convex subproblem solvers used within
SCvx algorithms. At each iteration of a successive convexification algorithm,
the non-convex problem is approximated by a convex subproblem, which is then
solved using one of these solver backends.

Current Implementations:
    CVXPy Solver: The default solver backend using CVXPy's modeling language
        with support for multiple backend solvers (CLARABEL, etc.).
        Includes optional code generation via cvxpygen for improved performance.

Planned Architecture (ABC-based):

A base class will be introduced to enable pluggable solver implementations.
This will enable users to implement custom solver backends such as:

- Direct Clarabel solver (Rust-based, GPU-capable)
- QPAX (JAX-based QP solver for end-to-end differentiability)
- OSQP direct interface (specialized for QP structure)
- Custom embedded solvers for real-time applications
- Research solvers with specialized structure exploitation

This should also make the solver choice independent of the algorithm choice

Future solvers will implement the ConvexSolver interface:

```python
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
```
"""

from .cvxpy import optimal_control_problem

__all__ = [
    "optimal_control_problem",
]
