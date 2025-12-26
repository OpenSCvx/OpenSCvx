"""Successive convexification algorithms for trajectory optimization.

This module provides implementations of SCvx (Successive Convexification) algorithms
for solving non-convex trajectory optimization problems through iterative convex
approximation.

All algorithms inherit from :class:`Algorithm`, enabling pluggable algorithm
implementations and custom SCvx variants:

```python
class Algorithm(ABC):
    @abstractmethod
    def initialize(self, params, ocp, discretization_solver,
                   settings, jax_constraints) -> Any:
        '''Initialize algorithm and return algorithm-specific data.'''
        ...

    @abstractmethod
    def step(self, params, settings, state, ocp, discretization_solver,
             init_data, emitter_function, jax_constraints) -> bool:
        '''Execute one iteration of the algorithm.'''
        ...
```

Current Implementations:
    - :class:`PenalizedTrustRegion`: Penalized Trust Region (PTR) algorithm
"""

from .base import Algorithm
from .optimization_results import OptimizationResults
from .penalized_trust_region import PenalizedTrustRegion, PTR_init, PTR_step
from .solver_state import SolverState

__all__ = [
    # Base class
    "Algorithm",
    # Core state and results
    "SolverState",
    "OptimizationResults",
    # PTR algorithm
    "PenalizedTrustRegion",
    "PTR_init",
    "PTR_step",
]
