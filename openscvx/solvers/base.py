"""Base class for convex subproblem solvers.

This module defines the abstract interface that all convex solver implementations
must follow for use within successive convexification algorithms.

The solver lifecycle follows three phases:

1. **create_variables**: Create backend-specific optimization variables
2. **initialize**: Build the complete optimization problem structure
3. **solve**: Update parameters and solve (called each SCP iteration)

This separation allows the lowering process to:
- Call ``create_variables()`` to get backend-specific variables
- Lower convex constraints using those variables
- Call ``initialize()`` with the lowered constraints

.. note:: Interface subject to change

    This interface is under active development. Solvers now own their optimization
    variables via ``create_variables()``, and ``LoweredProblem`` no longer contains
    ``ocp_vars``.

    TODO(norrisg): Continue refactoring toward backend-agnostic LoweredProblem:

    1. [DONE] Integrate ``create_variables()`` into the lowering flow
    2. [DONE] Have ``lower_symbolic_problem()`` call ``solver.create_variables()``
    3. [DONE] Remove ``ocp_vars`` from ``LoweredProblem``
    4. Remove ``cvxpy_constraints``, ``cvxpy_params`` from ``LoweredProblem``
       (move constraint lowering into solver)
    5. Add ``lower_convex_constraints()`` abstract method so each backend
       can provide its own constraint lowering path (QPAX, Clarabel, etc.)

    The goal is for ``LoweredProblem`` to contain only backend-agnostic data:
    dynamics (JAX), JAX constraints, unified state/control interfaces.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    import cvxpy as cp

    from openscvx.algorithms import AlgorithmState
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState


class ConvexSolver(ABC):
    """Abstract base class for convex subproblem solvers.

    This class defines the interface for solvers that handle the convex
    subproblems generated at each iteration of a successive convexification
    algorithm.

    The solver lifecycle has three phases:

    - create_variables: Create backend-specific variables (called once)
    - initialize: Build the problem structure using lowered constraints (called once)
    - solve: Update parameter values and solve (called each SCP iteration)

    This separation allows the lowering process to create variables first,
    then lower convex constraints using those variables, before building
    the complete problem.

    Example:
        Implementing a custom solver::

            class MySolver(ConvexSolver):
                def create_variables(self, N, x_unified, u_unified, jax_constraints):
                    # Create backend-specific variables
                    self._vars = create_my_variables(N, x_unified, ...)

                def initialize(self, lowered, settings):
                    # Build problem structure using self._vars
                    self._prob = build_my_problem(self._vars, lowered, settings)

                def solve(self, state, params, settings) -> cp.Problem:
                    # Update parameters and solve
                    update_params(self._prob, state, params)
                    self._prob.solve()
                    return self._prob
    """

    @abstractmethod
    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
    ) -> None:
        """Create backend-specific optimization variables.

        This method creates the optimization variables (decision variables and
        parameters) for this solver's backend. Called once during problem setup,
        before constraint lowering.

        The solver should store its variables on ``self`` for use in subsequent
        ``initialize()`` and ``solve()`` calls.

        Args:
            N: Number of discretization nodes
            x_unified: Unified state interface with dimensions and scaling bounds
            u_unified: Unified control interface with dimensions and scaling bounds
            jax_constraints: Lowered JAX constraints (for sizing linearization params)
        """
        ...

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
