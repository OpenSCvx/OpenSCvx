"""Base class for successive convexification algorithms.

This module defines the abstract interface that all SCP algorithm implementations
must follow, along with the AlgorithmState dataclass that holds mutable state
during SCP iterations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np

if TYPE_CHECKING:
    import cvxpy as cp

    from openscvx.config import Config
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints


@dataclass
class AlgorithmState:
    """Mutable state for SCP iterations.

    This dataclass holds all state that changes during the solve process.
    It stores only the evolving trajectory arrays, not the full State/Control
    objects which contain immutable configuration metadata.

    Trajectory arrays are stored in history lists, with the current guess
    accessed via properties that return the latest entry.

    A fresh instance is created for each solve, enabling easy reset functionality.

    Attributes:
        k: Current iteration number (starts at 1)
        J_tr: Current trust region cost
        J_vb: Current virtual buffer cost
        J_vc: Current virtual control cost
        w_tr: Current trust region weight (may adapt during solve)
        lam_cost: Current cost weight (may relax during solve)
        lam_vc: Current virtual control penalty weight
        lam_vb: Current virtual buffer penalty weight
        n_x: Number of states (for unpacking V vectors)
        n_u: Number of controls (for unpacking V vectors)
        N: Number of trajectory nodes (for unpacking V vectors)
        X: List of state trajectory iterates
        U: List of control trajectory iterates
        V_history: List of discretization history
    """

    k: int
    J_tr: float
    J_vb: float
    J_vc: float
    w_tr: float
    lam_cost: float
    lam_vc: Union[float, np.ndarray]
    lam_vb: float
    n_x: int
    n_u: int
    N: int
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    V_history: List[np.ndarray] = field(default_factory=list)
    VC_history: List[np.ndarray] = field(default_factory=list)
    TR_history: List[np.ndarray] = field(default_factory=list)

    @property
    def x(self) -> np.ndarray:
        """Get current state trajectory array.

        Returns:
            Current state trajectory guess (latest entry in history), shape (N, n_states)
        """
        return self.X[-1]

    @property
    def u(self) -> np.ndarray:
        """Get current control trajectory array.

        Returns:
            Current control trajectory guess (latest entry in history), shape (N, n_controls)
        """
        return self.U[-1]

    @property
    def x_prop(self) -> np.ndarray:
        """Extract propagated state trajectory from latest V.

        Returns:
            Propagated state trajectory x_prop with shape (N-1, n_x), or None if no V_history

        Example:
            After running an iteration, access the propagated states::

                problem.step()
                x_prop = problem.state.x_prop  # Shape (N-1, n_x)
        """
        if not self.V_history:
            return None

        # V_history contains Vmulti from discretization
        # Shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
        V = self.V_history[-1]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract propagated state (first n_x elements of each row)
        return V_final[:, : self.n_x]

    @property
    def A_d(self) -> np.ndarray:
        """Extract discretized state transition matrix from latest V.

        Returns:
            Discretized state Jacobian A_d with shape (N-1, n_x, n_x), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                A_d = problem.state.A_d  # Shape (N-1, n_x, n_x)
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i1 = self.n_x
        i2 = i1 + self.n_x * self.n_x

        # V_history contains Vmulti from discretization
        # Shape: (flattened_size, n_timesteps) where flattened_size = (N-1) * i4
        V = self.V_history[-1]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape A_d matrix
        return V_final[:, i1:i2].reshape(self.N - 1, self.n_x, self.n_x)

    @property
    def B_d(self) -> np.ndarray:
        """Extract discretized control influence matrix (current node) from latest V.

        Returns:
            Discretized control Jacobian B_d with shape (N-1, n_x, n_u), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                B_d = problem.state.B_d  # Shape (N-1, n_x, n_u)
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i1 = self.n_x
        i2 = i1 + self.n_x * self.n_x
        i3 = i2 + self.n_x * self.n_u

        # V_history contains Vmulti from discretization
        V = self.V_history[-1]

        # Take final timestep and reshape to (N-1, i4)
        i4 = self.n_x + self.n_x * self.n_x + 2 * self.n_x * self.n_u
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape B_d matrix
        return V_final[:, i2:i3].reshape(self.N - 1, self.n_x, self.n_u)

    @property
    def C_d(self) -> np.ndarray:
        """Extract discretized control influence matrix (next node) from latest V.

        Returns:
            Discretized control Jacobian C_d with shape (N-1, n_x, n_u), or None if no V_history

        Example:
            After running an iteration, access linearization matrices::

                problem.step()
                C_d = problem.state.C_d  # Shape (N-1, n_x, n_u)
        """
        if not self.V_history:
            return None

        # Extract indices for unpacking V vector
        i2 = self.n_x + self.n_x * self.n_x
        i3 = i2 + self.n_x * self.n_u
        i4 = i3 + self.n_x * self.n_u

        # V_history contains Vmulti from discretization
        V = self.V_history[-1]

        # Take final timestep and reshape to (N-1, i4)
        V_final = V[:, -1].reshape(-1, i4)

        # Extract and reshape C_d matrix
        return V_final[:, i3:i4].reshape(self.N - 1, self.n_x, self.n_u)

    @classmethod
    def from_settings(cls, settings: "Config") -> "AlgorithmState":
        """Create initial algorithm state from configuration.

        Copies only the trajectory arrays from settings, leaving all metadata
        (bounds, boundary conditions, etc.) in the original settings object.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters

        Returns:
            Fresh AlgorithmState initialized from settings with copied arrays
        """
        return cls(
            k=1,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            w_tr=settings.scp.w_tr,
            lam_cost=settings.scp.lam_cost,
            lam_vc=settings.scp.lam_vc,
            lam_vb=settings.scp.lam_vb,
            n_x=settings.sim.n_states,
            n_u=settings.sim.n_controls,
            N=settings.scp.n,
            X=[settings.sim.x.guess.copy()],
            U=[settings.sim.u.guess.copy()],
            V_history=[],
            VC_history=[],
            TR_history=[],
        )


class Algorithm(ABC):
    """Abstract base class for successive convexification algorithms.

    This class defines the interface for SCP algorithms used in trajectory
    optimization. Implementations should remain minimal and functional,
    delegating state management to the AlgorithmState dataclass.

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
        state: AlgorithmState,
        ocp: "cp.Problem",
        discretization_solver: callable,
        init_data: Any,
        emitter_function: callable,
        jax_constraints: "LoweredJaxConstraints",
    ) -> bool:
        """Execute one iteration of the SCP algorithm.

        This method solves a single convex subproblem, updates the algorithm
        state in place, and returns whether convergence criteria are met.

        Args:
            params: Problem parameters dictionary
            settings: Configuration object
            state: Mutable algorithm state (modified in place)
            ocp: The CVXPy optimal control problem
            discretization_solver: Compiled discretization solver function
            init_data: Algorithm-specific data returned from initialize()
            emitter_function: Callback for emitting iteration progress data
            jax_constraints: JIT-compiled JAX constraint functions

        Returns:
            True if convergence criteria are satisfied, False otherwise.
        """
        ...
