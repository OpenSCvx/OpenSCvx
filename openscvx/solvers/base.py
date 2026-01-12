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

!!! note

    Solvers own their optimization variables via ``create_variables()``.
    Convex constraint lowering remains in ``lower.py`` but uses the solver's
    variables. If non-CVXPy backends are needed in the future, the solver
    could own the constraint lowering as well (via a ``lower_convex_constraints()``
    method), while keeping the orchestration in ``lower_symbolic_problem()``:

    ```python
    @abstractmethod
    def lower_convex_constraints(self, constraints: ConstraintSet, parameters: dict) -> None:
        '''Lower symbolic convex constraints using created variables.'''
        ...
    ```
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
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

                def solve(self):
                    # Parameters already updated by algorithm
                    self._prob.solve()
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
                - ``cvxpy_constraints``: Lowered convex constraints
                - ``jax_constraints``: JAX constraint functions
                - ``x_unified``, ``u_unified``: State/control interfaces
            settings: Configuration object with solver settings
        """
        ...

    @abstractmethod
    def solve(self) -> None:
        """Solve the convex subproblem.

        Called at each SCP iteration after the algorithm has updated parameter
        values. The algorithm is responsible for setting linearization parameters
        before calling this method.

        Note:
            Results are accessed via backend-specific attributes after solving
            (e.g., ``solver.problem.var_dict["x"].value`` for CVXPy).
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
