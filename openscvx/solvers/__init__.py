"""Convex subproblem solvers for trajectory optimization.

This module provides implementations of convex subproblem solvers used within
SCvx algorithms. At each iteration of a successive convexification algorithm,
the non-convex problem is approximated by a convex subproblem, which is then
solved using one of these solver backends.

All solvers inherit from :class:`ConvexSolver`, enabling pluggable solver
implementations and custom backends:

```python
class ConvexSolver(ABC):
    @abstractmethod
    def initialize(self, lowered, settings) -> None:
        '''Build the convex subproblem structure (called once).'''
        ...

    @abstractmethod
    def solve(self, state, params, settings) -> Any:
        '''Update parameters and solve (called each iteration).'''
        ...
```

This architecture enables users to implement custom solver backends such as:

- Direct Clarabel solver (Rust-based, GPU-capable)
- QPAX (JAX-based QP solver for end-to-end differentiability)
- OSQP direct interface (specialized for QP structure)
- Custom embedded solvers for real-time applications
- Research solvers with specialized structure exploitation

Note:
    The current implementation is CVXPy-centric. :class:`LoweredProblem` contains
    CVXPy-specific objects (``ocp_vars``, ``cvxpy_constraints``). See the
    architecture note in :mod:`openscvx.solvers.base` for planned refactoring
    to support backend-agnostic problem lowering.

Note:
    CVXPyGen setup logic is currently in :class:`Problem`. When solvers are
    refactored to use the ``ConvexSolver`` base class, this setup will move here.
"""

from .base import ConvexSolver
from .cvxpy import CVXPySolver

__all__ = [
    # Base class
    "ConvexSolver",
    # CVXPy implementation
    "CVXPySolver",
]
