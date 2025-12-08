"""Trajectory propagation for trajectory optimization.

This module provides implementations of trajectory propagation methods that
simulate the nonlinear system dynamics forward in time. Propagation is used
to evaluate solution quality, verify constraint satisfaction, and generate
high-fidelity trajectories from optimized control sequences.

Current Implementations:
    Forward Simulation: The default propagation method that integrates the
        nonlinear dynamics forward in time using adaptive or fixed-step
        numerical integration (via Diffrax). Supports both ZOH and FOH
        control interpolation schemes.

Planned Architecture (ABC-based):

A base class will be introduced to enable pluggable propagation methods.
This will enable users to implement custom propagation methods.
Future propagators will implement the Propagator interface:

```python
# propagation/base.py (planned):
class Propagator(ABC):
    def __init__(self, integrator: Integrator):
        '''Initialize with a numerical integrator.'''
        self.integrator = integrator

    @abstractmethod
    def propagate(self, dynamics, x0, u_traj, time_grid) -> Array:
        '''Propagate trajectory forward in time.

        Args:
            dynamics: Continuous-time dynamics object
            x0: Initial state
            u_traj: Control trajectory
            time_grid: Time points for dense output

        Returns:
            State trajectory evaluated at time_grid points
        '''
        ...
```
"""

from .post_processing import propagate_trajectory_results
from .propagation import (
    get_propagation_solver,
    prop_aug_dy,
    s_to_t,
    simulate_nonlinear_time,
    t_to_tau,
)

__all__ = [
    "get_propagation_solver",
    "simulate_nonlinear_time",
    "prop_aug_dy",
    "s_to_t",
    "t_to_tau",
    "propagate_trajectory_results",
]
