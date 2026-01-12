"""CVXPy-based convex subproblem solver.

This module provides the default solver backend using CVXPy's modeling language
with support for multiple backend solvers (CLARABEL, etc.). Includes optional
code generation via cvxpygen for improved performance.
"""

import os
from typing import TYPE_CHECKING, Any, List

import cvxpy as cp
import numpy as np

from openscvx.config import Config

from .base import ConvexSolver

if TYPE_CHECKING:
    from openscvx.algorithms import AlgorithmState
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.cvxpy_variables import CVXPyVariables
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState

# Optional cvxpygen import
try:
    from cvxpygen import cpg

    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False
    cpg = None


class CVXPySolver(ConvexSolver):
    """CVXPy-based convex subproblem solver.

    This solver uses CVXPy's modeling language to construct and solve the convex
    subproblems generated at each SCP iteration. It supports multiple backend
    solvers (CLARABEL, ECOS, MOSEK, etc.) and optional code generation via
    cvxpygen for improved performance.

    The solver builds the problem structure once during ``initialize()``, using
    CVXPy Parameters for values that change each iteration. The ``solve()``
    method then updates these parameters and solves the problem.

    Note:
        Parameter updates (linearization matrices, constraint gradients, etc.)
        are currently performed by the algorithm before calling ``solve()``.
        A future refactor may move this logic into the solver for better
        encapsulation.

    Example:
        Using CVXPySolver with the SCP framework::

            solver = CVXPySolver()
            solver.initialize(lowered, settings)

            # Each iteration (parameter updates done by algorithm):
            prob = solver.solve(state, params, settings)
            x_sol = prob.var_dict["x"].value

    Attributes:
        problem: The CVXPy Problem object (available after initialize())
    """

    def __init__(self):
        """Initialize CVXPySolver with unset problem.

        Call create_variables() then initialize() to build the problem structure.
        """
        self._ocp_vars: "CVXPyVariables" = None
        self._problem: cp.Problem = None
        self._solve_fn: callable = None

    @property
    def problem(self) -> cp.Problem:
        """The CVXPy Problem object.

        Returns:
            The constructed CVXPy problem, or None if not initialized.
        """
        return self._problem

    @property
    def ocp_vars(self) -> "CVXPyVariables":
        """The CVXPy variables and parameters.

        Returns:
            The CVXPyVariables dataclass, or None if create_variables() not called.
        """
        return self._ocp_vars

    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
    ) -> None:
        """Create CVXPy optimization variables.

        Creates all CVXPy Variable and Parameter objects needed for the optimal
        control problem. This includes state/control variables, dynamics parameters,
        constraint linearization parameters, and scaling matrices.

        Args:
            N: Number of discretization nodes
            x_unified: Unified state interface with dimensions and scaling bounds
            u_unified: Unified control interface with dimensions and scaling bounds
            jax_constraints: Lowered JAX constraints (for sizing linearization params)
        """
        from openscvx.config import get_affine_scaling_matrices
        from openscvx.symbolic.lower import create_cvxpy_variables

        n_states = len(x_unified.max)
        n_controls = len(u_unified.max)

        # Compute scaling matrices from unified object bounds
        if x_unified.scaling_min is not None:
            lower_x = np.array(x_unified.scaling_min, dtype=float)
        else:
            lower_x = np.array(x_unified.min, dtype=float)

        if x_unified.scaling_max is not None:
            upper_x = np.array(x_unified.scaling_max, dtype=float)
        else:
            upper_x = np.array(x_unified.max, dtype=float)

        S_x, c_x = get_affine_scaling_matrices(n_states, lower_x, upper_x)

        if u_unified.scaling_min is not None:
            lower_u = np.array(u_unified.scaling_min, dtype=float)
        else:
            lower_u = np.array(u_unified.min, dtype=float)

        if u_unified.scaling_max is not None:
            upper_u = np.array(u_unified.scaling_max, dtype=float)
        else:
            upper_u = np.array(u_unified.max, dtype=float)

        S_u, c_u = get_affine_scaling_matrices(n_controls, lower_u, upper_u)

        # Create all CVXPy variables for the OCP
        self._ocp_vars = create_cvxpy_variables(
            N=N,
            n_states=n_states,
            n_controls=n_controls,
            S_x=S_x,
            c_x=c_x,
            S_u=S_u,
            c_u=c_u,
            n_nodal_constraints=len(jax_constraints.nodal),
            n_cross_node_constraints=len(jax_constraints.cross_node),
        )

    def initialize(
        self,
        lowered: "LoweredProblem",
        settings: "Config",
    ) -> None:
        """Build the CVXPy optimal control problem.

        Constructs the complete optimization problem with all constraints,
        using CVXPy Parameters for values that change each SCP iteration
        (linearization matrices, constraint gradients, penalty weights, etc.).

        If cvxpygen is enabled in settings, generates compiled solver code
        for improved performance.

        Note:
            ``create_variables()`` must be called before this method.

        Args:
            lowered: Lowered problem containing:
                - ``cvxpy_constraints``: Lowered convex constraints
                - ``jax_constraints``: JAX constraint functions (for structure)
            settings: Configuration object with solver settings

        Raises:
            RuntimeError: If create_variables() has not been called.
        """
        if self._ocp_vars is None:
            raise RuntimeError(
                "CVXPySolver.initialize() called before create_variables(). "
                "Call create_variables() first to create optimization variables."
            )
        self._problem = _build_optimal_control_problem(settings, lowered, self._ocp_vars)
        self._setup_solve_function(settings)

    def _setup_solve_function(self, settings: "Config") -> None:
        """Configure the solve function based on settings.

        Sets up either cvxpygen-based solving or standard CVXPy solving
        based on the configuration.

        Args:
            settings: Configuration object with solver settings
        """
        if settings.cvx.cvxpygen:
            try:
                import pickle

                from solver.cpg_solver import cpg_solve

                with open("solver/problem.pickle", "rb") as f:
                    pickle.load(f)
                self._problem.register_solve("CPG", cpg_solve)
                solver_args = settings.cvx.solver_args
                self._solve_fn = lambda: self._problem.solve(method="CPG", **solver_args)
            except ImportError:
                raise ImportError(
                    "cvxpygen solver not found. Make sure cvxpygen is installed and code "
                    "generation has been run. Install with: pip install openscvx[cvxpygen]"
                )
        else:
            solver = settings.cvx.solver
            solver_args = settings.cvx.solver_args
            self._solve_fn = lambda: self._problem.solve(solver=solver, **solver_args)

    def solve(
        self,
        state: "AlgorithmState",
        params: dict,
        settings: "Config",
    ) -> cp.Problem:
        """Solve the convex subproblem.

        Note:
            Parameter updates (x_bar, u_bar, A_d, B_d, constraint linearizations,
            etc.) are currently performed by the algorithm before calling this
            method. This method simply invokes the configured solver.

            A future refactor may move parameter update logic here for better
            encapsulation, at which point the full ``state`` and ``params``
            arguments will be used.

        Args:
            state: Current algorithm state (reserved for future use)
            params: Problem parameters dictionary (reserved for future use)
            settings: Configuration object (reserved for future use)

        Returns:
            The solved CVXPy problem. Access solution via:
            - ``problem.var_dict["x"].value`` for state trajectory
            - ``problem.var_dict["u"].value`` for control trajectory
            - ``problem.value`` for optimal cost
            - ``problem.status`` for solver status

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._problem is None:
            raise RuntimeError(
                "CVXPySolver.solve() called before initialize(). "
                "Call initialize() first to build the problem structure."
            )

        self._solve_fn()
        return self._problem

    def citation(self) -> List[str]:
        """Return BibTeX citations for CVXPy.

        Returns:
            List containing BibTeX entries for CVXPy and DCCP papers.
        """
        return [
            r"""@article{diamond2016cvxpy,
  title={CVXPY: A Python-embedded modeling language for convex optimization},
  author={Diamond, Steven and Boyd, Stephen},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={83},
  pages={1--5},
  year={2016}
}""",
            r"""@article{agrawal2018rewriting,
  title={A rewriting system for convex optimization problems},
  author={Agrawal, Akshay and Verschueren, Robin and Diamond, Steven and Boyd, Stephen},
  journal={Journal of Control and Decision},
  volume={5},
  number={1},
  pages={42--60},
  year={2018},
  publisher={Taylor \& Francis}
}""",
        ]


def _build_optimal_control_problem(
    settings: Config,
    lowered: "LoweredProblem",
    ocp_vars: "CVXPyVariables",
) -> cp.Problem:
    """Build the complete optimal control problem with all constraints.

    This is the internal implementation that constructs the CVXPy problem
    structure. It is called by CVXPySolver.initialize().

    Args:
        settings: Configuration settings for the optimization problem
        lowered: LoweredProblem containing lowered constraints
        ocp_vars: CVXPy variables and parameters (owned by solver)

    Returns:
        The constructed CVXPy Problem object.
    """

    # Extract variables from the dataclass for easier access
    w_tr = ocp_vars.w_tr
    lam_cost = ocp_vars.lam_cost
    lam_vc = ocp_vars.lam_vc
    lam_vb = ocp_vars.lam_vb
    x = ocp_vars.x
    dx = ocp_vars.dx
    x_bar = ocp_vars.x_bar
    x_init = ocp_vars.x_init
    x_term = ocp_vars.x_term
    u = ocp_vars.u
    du = ocp_vars.du
    u_bar = ocp_vars.u_bar
    A_d = ocp_vars.A_d
    B_d = ocp_vars.B_d
    C_d = ocp_vars.C_d
    x_prop = ocp_vars.x_prop
    nu = ocp_vars.nu
    g = ocp_vars.g
    grad_g_x = ocp_vars.grad_g_x
    grad_g_u = ocp_vars.grad_g_u
    nu_vb = ocp_vars.nu_vb
    g_cross = ocp_vars.g_cross
    grad_g_X_cross = ocp_vars.grad_g_X_cross
    grad_g_U_cross = ocp_vars.grad_g_U_cross
    nu_vb_cross = ocp_vars.nu_vb_cross
    S_x = ocp_vars.S_x
    c_x = ocp_vars.c_x
    S_u = ocp_vars.S_u
    c_u = ocp_vars.c_u
    x_nonscaled = ocp_vars.x_nonscaled
    u_nonscaled = ocp_vars.u_nonscaled
    dx_nonscaled = ocp_vars.dx_nonscaled
    du_nonscaled = ocp_vars.du_nonscaled

    # Extract lowered constraints
    jax_constraints = lowered.jax_constraints
    cvxpy_constraints = lowered.cvxpy_constraints

    constr = []
    cost = lam_cost * 0
    cost += lam_vb * 0

    #############
    # CONSTRAINTS
    #############

    # Linearized nodal constraints (from JAX-lowered non-convex)
    idx_ncvx = 0
    if jax_constraints.nodal:
        for constraint in jax_constraints.nodal:
            # nodes should already be validated and normalized in preprocessing
            nodes = constraint.nodes
            constr += [
                (
                    g[idx_ncvx][node]
                    + grad_g_x[idx_ncvx][node] @ dx[node]
                    + grad_g_u[idx_ncvx][node] @ du[node]
                )
                == nu_vb[idx_ncvx][node]
                for node in nodes
            ]
            idx_ncvx += 1

    # Linearized cross-node constraints (from JAX-lowered non-convex)
    idx_cross = 0
    if jax_constraints.cross_node:
        for constraint in jax_constraints.cross_node:
            # Linearization: g(X_bar, U_bar) + ∇g_X @ dX + ∇g_U @ dU == nu_vb
            # Sum over all trajectory nodes to couple multiple nodes
            residual = g_cross[idx_cross]
            for k in range(settings.scp.n):
                # Contribution from state at node k
                residual += grad_g_X_cross[idx_cross][k, :] @ dx[k]
                # Contribution from control at node k
                residual += grad_g_U_cross[idx_cross][k, :] @ du[k]
            # Add constraint: residual == slack variable
            constr += [residual == nu_vb_cross[idx_cross]]
            idx_cross += 1

    # Convex constraints (already lowered to CVXPy)
    if cvxpy_constraints.constraints:
        constr += cvxpy_constraints.constraints

    for i in range(settings.sim.true_state_slice.start, settings.sim.true_state_slice.stop):
        if settings.sim.x.initial_type[i] == "Fix":
            constr += [x_nonscaled[0][i] == x_init[i]]  # Initial Boundary Conditions
        if settings.sim.x.final_type[i] == "Fix":
            constr += [x_nonscaled[-1][i] == x_term[i]]  # Final Boundary Conditions
        if settings.sim.x.initial_type[i] == "Minimize":
            cost += lam_cost * x_nonscaled[0][i]
        if settings.sim.x.final_type[i] == "Minimize":
            cost += lam_cost * x_nonscaled[-1][i]
        if settings.sim.x.initial_type[i] == "Maximize":
            cost -= lam_cost * x_nonscaled[0][i]
        if settings.sim.x.final_type[i] == "Maximize":
            cost -= lam_cost * x_nonscaled[-1][i]

    if settings.scp.uniform_time_grid:
        constr += [
            u_nonscaled[i][settings.sim.time_dilation_slice]
            == u_nonscaled[i - 1][settings.sim.time_dilation_slice]
            for i in range(1, settings.scp.n)
        ]

    constr += [
        (x[i] - np.linalg.inv(S_x) @ (x_bar[i] - c_x) - dx[i]) == 0 for i in range(settings.scp.n)
    ]  # State Error
    constr += [
        (u[i] - np.linalg.inv(S_u) @ (u_bar[i] - c_u) - du[i]) == 0 for i in range(settings.scp.n)
    ]  # Control Error

    constr += [
        x_nonscaled[i]
        == A_d[i - 1] @ dx_nonscaled[i - 1]
        + B_d[i - 1] @ du_nonscaled[i - 1]
        + C_d[i - 1] @ du_nonscaled[i]
        + x_prop[i - 1]
        + nu[i - 1]
        for i in range(1, settings.scp.n)
    ]  # Dynamics Constraint

    constr += [u_nonscaled[i] <= settings.sim.u.max for i in range(settings.scp.n)]
    constr += [
        u_nonscaled[i] >= settings.sim.u.min for i in range(settings.scp.n)
    ]  # Control Constraints

    # TODO: (norrisg) formalize this
    constr += [x_nonscaled[i][:] <= settings.sim.x.max for i in range(settings.scp.n)]
    constr += [
        x_nonscaled[i][:] >= settings.sim.x.min for i in range(settings.scp.n)
    ]  # State Constraints (Also implemented in CTCS but included for numerical stability)

    ########
    # COSTS
    ########

    cost += sum(
        w_tr * cp.sum_squares(cp.hstack((dx[i], du[i]))) for i in range(settings.scp.n)
    )  # Trust Region Cost
    cost += sum(
        cp.sum(lam_vc[i - 1] * cp.abs(nu[i - 1])) for i in range(1, settings.scp.n)
    )  # Virtual Control Slack

    idx_ncvx = 0
    if jax_constraints.nodal:
        for constraint in jax_constraints.nodal:
            cost += lam_vb * cp.sum(cp.pos(nu_vb[idx_ncvx]))
            idx_ncvx += 1

    # Virtual slack penalty for cross-node constraints
    idx_cross = 0
    if jax_constraints.cross_node:
        for constraint in jax_constraints.cross_node:
            cost += lam_vb * cp.pos(nu_vb_cross[idx_cross])
            idx_cross += 1

    for idx, nodes in zip(
        np.arange(settings.sim.ctcs_slice.start, settings.sim.ctcs_slice.stop),
        settings.sim.ctcs_node_intervals,
    ):
        start_idx = 1 if nodes[0] == 0 else nodes[0]
        constr += [
            cp.abs(x_nonscaled[i][idx] - x_nonscaled[i - 1][idx]) <= settings.sim.x.max[idx]
            for i in range(start_idx, nodes[1])
        ]
        constr += [x_nonscaled[0][idx] == 0]

    #########
    # PROBLEM
    #########
    prob = cp.Problem(cp.Minimize(cost), constr)

    # Handle cvxpygen code generation
    if settings.cvx.cvxpygen:
        if not CVXPYGEN_AVAILABLE:
            raise ImportError(
                "cvxpygen is required for code generation but not installed. "
                "Install it with: pip install openscvx[cvxpygen] or pip install cvxpygen"
            )
        # Check to see if solver directory exists
        if not os.path.exists("solver"):
            cpg.generate_code(prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True)
        else:
            # Prompt the user to indicate if they wish to overwrite the solver
            # directory or use the existing compiled solver
            if settings.cvx.cvxpygen_override:
                cpg.generate_code(prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True)
            else:
                overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
                if overwrite.lower() == "y":
                    cpg.generate_code(
                        prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True
                    )

    return prob
