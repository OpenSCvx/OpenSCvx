"""Base class for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
must follow. The design is intentionally minimal and functional - algorithms
define initialization and step methods without storing significant state.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cvxpy as cp

    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints

    from .solver_state import SolverState


class Algorithm(ABC):
    """Abstract base class for successive convexification algorithms.

    This class defines the interface for SCP algorithms used in trajectory
    optimization. Implementations should remain minimal and functional,
    delegating state management to the SolverState dataclass.

    The two core methods mirror the SCP workflow:
    - initialize: Prepare algorithm-specific solver state (e.g., warm-start data)
    - step: Execute one convex subproblem iteration

    Example:
        Implementing a custom algorithm::

            class MyAlgorithm(Algorithm):
                def initialize(self, params, ocp, discretization_solver,
                               settings, jax_constraints):
                    # Setup and return any algorithm-specific data
                    return my_init_data

                def step(self, params, settings, state, ocp, discretization_solver,
                         init_data, emitter_function, jax_constraints):
                    # Run one iteration, mutate state, return convergence
                    return converged
    """

    @abstractmethod
    def initialize(
        self,
        params: dict,
        ocp: "cp.Problem",
        discretization_solver: callable,
        settings: "Config",
        jax_constraints: "LoweredJaxConstraints",
    ) -> Any:
        """Initialize the algorithm and return algorithm-specific data.

        This method performs any setup required before the SCP loop begins,
        such as warm-starting solvers or pre-compiling solver code.

        Args:
            params: Problem parameters dictionary (for JAX/CVXPy)
            ocp: The CVXPy optimal control problem
            discretization_solver: Compiled discretization solver function
            settings: Configuration object with SCP, simulation, and solver settings
            jax_constraints: JIT-compiled JAX constraint functions

        Returns:
            Algorithm-specific initialization data to be passed to step().
            For example, PTR returns a cpg_solve handle for CVXPyGen.
        """
        ...

    @abstractmethod
    def step(
        self,
        params: dict,
        settings: "Config",
        state: "SolverState",
        ocp: "cp.Problem",
        discretization_solver: callable,
        init_data: Any,
        emitter_function: callable,
        jax_constraints: "LoweredJaxConstraints",
    ) -> bool:
        """Execute one iteration of the SCP algorithm.

        This method solves a single convex subproblem, updates the solver
        state in place, and returns whether convergence criteria are met.

        Args:
            params: Problem parameters dictionary
            settings: Configuration object
            state: Mutable solver state (modified in place)
            ocp: The CVXPy optimal control problem
            discretization_solver: Compiled discretization solver function
            init_data: Algorithm-specific data returned from initialize()
            emitter_function: Callback for emitting iteration progress data
            jax_constraints: JIT-compiled JAX constraint functions

        Returns:
            True if convergence criteria are satisfied, False otherwise.
        """
        ...
