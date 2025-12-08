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

Core Functions:
    get_propagation_solver: Factory function that creates a propagation solver
        for given dynamics and configuration. Returns a callable that propagates
        the system state forward through time.

    simulate_nonlinear_time: High-level function that simulates the nonlinear
        system dynamics over a specified time grid using the optimal control
        sequence. Handles segmented integration and control interpolation.

    prop_aug_dy: Computes the augmented dynamics for propagation, including
        time scaling and control interpolation based on discretization type.

    s_to_t: Converts normalized time s to real time t based on time dilation
        factors and discretization type (ZOH or FOH).

    t_to_tau: Converts real time t to normalized time tau and interpolates
        control inputs accordingly using the specified discretization type.

Technical Details:
    The propagation process handles:
    - Time dilation effects from normalized to real time coordinates
    - Control interpolation (ZOH assumes piecewise constant, FOH uses linear)
    - Segmented integration across the trajectory nodes
    - Adaptive time stepping for accuracy and efficiency
    - Dense output at arbitrary time points within each segment

    Propagation uses the integrators from the `integrators/` module and
    respects the discretization type (ZOH/FOH) configured in the settings.

Planned Architecture (ABC-based):
    A base class will be introduced to enable pluggable propagation methods.
    Future propagators will implement the Propagator interface:

    .. code-block:: python

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

    This will enable users to implement custom propagation methods such as:
    - Parallel shooting (propagate multiple segments in parallel)
    - Multiple shooting (propagate with intermediate state corrections)
    - Event-based propagation (handle discontinuous dynamics, impacts, etc.)
    - Custom research propagation schemes

Note:
    Propagation methods depend on integrators from the `integrators/` module.
    The propagation scheme is independent of the SCvX algorithm, solver, and
    discretization choices, though it should match the discretization type
    (ZOH/FOH) used during optimization for consistency.
"""

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
]
