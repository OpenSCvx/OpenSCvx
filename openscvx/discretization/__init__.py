"""Discretization methods for trajectory optimization.

This module provides implementations of discretization schemes that convert
continuous-time optimal control problems into discrete-time approximations
suitable for numerical optimization. Discretization is a critical step in
trajectory optimization that linearizes the nonlinear dynamics around a
reference trajectory.

Planned Architecture (ABC-based):

A base class will be introduced to enable pluggable discretization methods.
This will enable users to implement custom discretization methods.
Future discretizers will implement the Discretizer interface:

```python
# discretization/base.py (planned):
class Discretizer(ABC):
    def __init__(self, integrator: Integrator):
        '''Initialize with a numerical integrator.'''
        self.integrator = integrator

    @abstractmethod
    def discretize(self, dynamics, x, u, dt) -> tuple[A_d, B_d, C_d]:
        '''Discretize continuous dynamics around trajectory (x, u).

        Args:
            dynamics: Continuous-time dynamics object
            x: State trajectory
            u: Control trajectory
            dt: Time step

        Returns:
            A_d: Discretized state transition matrix
            B_d: Discretized control influence matrix (current node)
            C_d: Discretized control influence matrix (next node)
        '''
        ...
```
"""

from .discretization import calculate_discretization, dVdt, get_discretization_solver

__all__ = [
    "calculate_discretization",
    "get_discretization_solver",
    "dVdt",
]
