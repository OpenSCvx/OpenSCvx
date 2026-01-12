"""Base class for convex subproblem solvers.

This module defines the abstract interface that all convex solver implementations
must follow for use within successive convexification algorithms.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import cvxpy as cp

    from openscvx.algorithms import AlgorithmState
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem


class ConvexSolver(ABC):
    """Abstract base class for convex subproblem solvers.

    This class defines the interface for solvers that handle the convex
    subproblems generated at each iteration of a successive convexification
    algorithm.

    The two core methods mirror the SCP workflow:

    - initialize: Build the problem structure once (with Parameters)
    - solve: Update parameter values and solve each iteration

    Example:
        Implementing a custom solver::

            class MySolver(ConvexSolver):
                def initialize(self, lowered, settings):
                    # Build problem structure with CVXPy Parameters
                    self._prob = build_my_problem(lowered, settings)

                def solve(self, state, params, settings) -> cp.Problem:
                    # Update parameters and solve
                    update_params(self._prob, state, params)
                    self._prob.solve()
                    return self._prob
    """

    @abstractmethod
    def initialize(
        self,
        lowered: "LoweredProblem",
        settings: "Config",
    ) -> None:
        """Build the convex subproblem structure.

        This method constructs the optimization problem once, using CVXPy
        Parameters (or equivalent) for values that change each iteration.
        Called once during problem setup, not at each SCP iteration.

        Args:
            lowered: Lowered problem with CVXPy variables and constraints
            settings: Configuration object with solver settings
        """
        ...

    @abstractmethod
    def solve(
        self,
        state: "AlgorithmState",
        params: dict,
        settings: "Config",
    ) -> "cp.Problem":
        """Update parameters and solve the convex subproblem.

        Called at each SCP iteration. Updates the parameter values from the
        current linearization point and solves the problem.

        Args:
            state: Current algorithm state containing linearization point
            params: Problem parameters dictionary
            settings: Configuration object with solver settings

        Returns:
            The solved CVXPy problem (or equivalent representation).
        """
        ...

    @abstractmethod
    def citation(self) -> List[str]:
        """Return BibTeX citations for this solver.

        Implementations should return a list of BibTeX entry strings for the
        papers that should be cited when using this solver.

        Returns:
            List of BibTeX citation strings.
        """
        ...
