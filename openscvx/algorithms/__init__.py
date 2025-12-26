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

:class:`AlgorithmState` holds mutable state during SCP iterations. Algorithms
that require additional state can subclass it:

```python
@dataclass
class MyAlgorithmState(AlgorithmState):
    my_custom_field: float = 0.0
```

Current Implementations:
    - :class:`PenalizedTrustRegion`: Penalized Trust Region (PTR) algorithm
"""

from .base import Algorithm, AlgorithmState
from .optimization_results import OptimizationResults
from .penalized_trust_region import PenalizedTrustRegion

__all__ = [
    # Base class
    "Algorithm",
    "AlgorithmState",
    # Core results
    "OptimizationResults",
    # PTR algorithm
    "PenalizedTrustRegion",
]
