"""Successive convexification algorithms for trajectory optimization.

This module provides implementations of SCvx (Successive Convexification) algorithms
for solving non-convex trajectory optimization problems through iterative convex
approximation.

Current Implementations:
    PTR (Penalized Trust Region): The default SCvx algorithm using trust region
        methods with penalty-based constraint handling. Includes adaptive parameter
        tuning and virtual control relaxation.

Planned Architecture (ABC-based):

A base class will be introduced to enable pluggable algorithm implementations.
This will enable users to implement custom SCvx variants or research algorithms.
Future algorithms will implement the SCvxAlgorithm interface:

```python
# algorithms/base.py (planned):
class SCvxAlgorithm(ABC):
    @abstractmethod
    def initialize(self, lowered: LoweredProblem) -> SolverState:
        '''Initialize solver state from a lowered problem.'''
        ...

    @abstractmethod
    def step(self, state: SolverState, solver: ConvexSolver) -> SolverState:
        '''Execute one iteration of the algorithm.'''
        ...
```
"""

from .optimization_results import OptimizationResults
from .ptr import PTR_init, PTR_step, format_result
from .solver_state import SolverState

__all__ = [
    # Core state and results
    "SolverState",
    "OptimizationResults",
    # PTR algorithm
    "PTR_init",
    "PTR_step",
    "format_result",
]
