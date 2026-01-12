"""Base class for convex subproblem solvers.

This module defines the abstract interface that all convex solver implementations
must follow for use within successive convexification algorithms.

Architecture Note:
    The current implementation is CVXPy-centric. :class:`LoweredProblem` contains
    ``ocp_vars`` (:class:`CVXPyVariables`) and ``cvxpy_constraints``, which are
    CVXPy-specific. This couples the lowering layer to CVXPy.

    To support alternative backends (QPAX, Clarabel, COCO, etc.), a future refactor
    should:

    1. Move ``CVXPyVariables`` creation from ``lower_symbolic_problem()`` into
       ``CVXPySolver.initialize()``, so each solver owns its problem representation.

    2. Keep ``LoweredProblem`` backend-agnostic, containing only:
       - Dynamics (JAX functions)
       - JAX constraints (for linearization)
       - Unified state/control interfaces
       - Problem dimensions and constraint metadata

    3. Add alternative constraint lowering paths (symbolic -> target format) or
       have solvers convert from a common intermediate representation.

    For now, non-CVXPy solvers would need to either:
    - Extract data from ``CVXPyVariables`` and convert to their format
    - Work at a lower level, bypassing symbolic constraints entirely
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

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

    Note:
        The current interface is CVXPy-centric. ``LoweredProblem`` provides
        ``ocp_vars`` (CVXPy Variables/Parameters) and ``solve()`` returns a
        ``cp.Problem``. Future backends may require interface changes - see
        the module-level architecture note for planned refactoring.

    Example:
        Implementing a custom solver::

            class MySolver(ConvexSolver):
                def initialize(self, lowered, settings):
                    # Build problem structure using lowered.ocp_vars
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

        The solver should store its problem representation on ``self`` for use
        in subsequent ``solve()`` calls.

        Args:
            lowered: Lowered problem containing:
                - ``ocp_vars``: CVXPy Variables and Parameters (CVXPy-specific)
                - ``cvxpy_constraints``: Lowered convex constraints (CVXPy-specific)
                - ``jax_constraints``: JAX constraint functions (backend-agnostic)
                - ``x_unified``, ``u_unified``: State/control interfaces (backend-agnostic)
            settings: Configuration object with solver settings
        """
        ...

    @abstractmethod
    def solve(
        self,
        state: "AlgorithmState",
        params: dict,
        settings: "Config",
    ) -> Any:
        """Update parameters and solve the convex subproblem.

        Called at each SCP iteration. Updates the parameter values from the
        current linearization point and solves the problem.

        Args:
            state: Current algorithm state containing linearization point
                (provides ``x``, ``u`` for linearization via properties)
            params: Problem parameters dictionary
            settings: Configuration object with solver settings

        Returns:
            The solved problem. Currently returns ``cp.Problem`` for CVXPy-based
            solvers. Future backends may return different types (e.g., solution
            arrays, result objects). The return type will be refined when the
            interface is generalized.
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
