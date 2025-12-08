"""Numerical integration schemes for trajectory optimization.

This module provides implementations of numerical integrators used for simulating
continuous-time dynamics and discretizing optimal control problems. Integrators
are foundational components used by both the discretization and propagation modules.

Current Implementations:
    RK45 Integration: Explicit Runge-Kutta-Fehlberg method (4th/5th order)
        with both fixed-step and adaptive implementations via Diffrax.
        Supports a variety of explicit and implicit ODE solvers through the
        Diffrax backend (Dopri5/8, Tsit5, KenCarp3/4/5, etc.).

Core Functions:
    rk45_step: Perform a single RK45 integration step with Dorman-Prince coefficients.
    solve_ivp_rk45: Solve an initial-value problem using fixed-step RK45 integration.
    solve_ivp_diffrax: Solve an IVP using Diffrax's adaptive solvers with error control.
    solve_ivp_diffrax_prop: Diffrax-based IVP solver specialized for trajectory
        optimization with dense output and masking support.

Core Constants:
    SOLVER_MAP: Dictionary mapping solver names to Diffrax solver classes
        (Tsit5, Euler, Heun, Dopri5/8, KenCarp3/4/5, etc.).

Planned Architecture (ABC-based):
    A base class will be introduced to enable pluggable integrator implementations.
    Future integrators will implement the Integrator interface:

    .. code-block:: python

        # integrators/base.py (planned):
        class Integrator(ABC):
            @abstractmethod
            def step(self, f: Callable, x: Array, u: Array, t: float, dt: float) -> Array:
                '''Take one integration step from state x at time t with step dt.'''
                ...

            @abstractmethod
            def integrate(self, f: Callable, x0: Array, u_traj: Array,
                         t_span: tuple[float, float], num_steps: int) -> Array:
                '''Integrate over a time span with given control trajectory.'''
                ...

    This will enable users to implement custom integrators such as:
    - Explicit methods: Euler, Midpoint, Ralston, Heun
    - Runge-Kutta variants: RK4, RK45 (Dorman-Prince), Tsit5, Dopri5/8
    - Custom embedded methods: For specialized applications or research
    - _etc._

Note:
    Integrators are foundational components shared by both discretization
    (for linearizing dynamics) and propagation (for forward simulation) modules.
    This separation ensures integrators remain minimal and reusable across
    different trajectory optimization contexts.
"""

from .rk4 import (
    SOLVER_MAP,
    rk45_step,
    solve_ivp_diffrax,
    solve_ivp_diffrax_prop,
    solve_ivp_rk45,
)

__all__ = [
    "SOLVER_MAP",
    "rk45_step",
    "solve_ivp_rk45",
    "solve_ivp_diffrax",
    "solve_ivp_diffrax_prop",
]
