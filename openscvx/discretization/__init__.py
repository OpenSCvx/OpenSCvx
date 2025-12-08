"""OCP discretization methods for trajectory optimization.

This module provides implementations of discretization schemes that convert
continuous-time optimal control problems into discrete-time approximations
suitable for numerical optimization. Discretization is a critical step in
trajectory optimization that linearizes the nonlinear dynamics around a
reference trajectory.

Current Implementations:
    ZOH (Zero-Order Hold): Assumes control inputs are piecewise constant
        over each time interval. This is the most common discretization
        method for trajectory optimization.

    FOH (First-Order Hold): Assumes control inputs vary linearly between
        nodes, providing higher accuracy for smooth control profiles at
        the cost of additional computation.

Core Functions:
    get_discretization_solver: Factory function that creates a discretization
        solver for a given dynamics object and configuration. Returns a
        callable that computes discretized system matrices.

    calculate_discretization: Computes the discretized system matrices
        (A_bar, B_bar, C_bar) and defect vector using numerical integration
        of augmented state-transition dynamics.

    dVdt: Computes time derivatives of the augmented state vector including
        state, state-transition matrix, and control influence matrices. Used
        internally by the discretization integrators.

Technical Details:
    The discretization process integrates an augmented system that includes:
    - State trajectory: x(t)
    - State transition matrix
    - Control influence matrices: B_cur(t) and B_next(t)

    The resulting discrete-time system approximates:
        x[k+1] H A_bar[k] @ x[k] + B_bar[k] @ u[k] + C_bar[k] @ u[k+1]

    where A_bar, B_bar, and C_bar are computed via numerical integration
    using either custom RK45 or Diffrax adaptive solvers.

Planned Architecture (ABC-based):
    A base class will be introduced to enable pluggable discretization methods.
    Future discretizers will implement the Discretizer interface:

    .. code-block:: python

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

    This will enable users to implement custom discretization methods such as:
    - Direct collocation (Hermite-Simpson, trapezoidal)
    - Multiple shooting variants
    - Pseudospectral methods (Chebyshev, Legendre)
    - Custom research discretization schemes

Note:
    Discretization methods depend on integrators from the `integrators/` module.
    The discretization scheme is independent of the SCvX algorithm and solver
    choices, allowing flexible combinations based on problem characteristics.
"""

from .discretization import calculate_discretization, dVdt, get_discretization_solver

__all__ = [
    "calculate_discretization",
    "get_discretization_solver",
    "dVdt",
]
