import os
from typing import Dict, List

import cvxpy as cp
import numpy as np

from openscvx.config import Config

# Optional cvxpygen import
try:
    from cvxpygen import cpg

    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False
    cpg = None


def create_cvxpy_variables(settings: Config) -> Dict:
    """Phase 1: Create CVXPy variables and parameters for the optimal control problem."""
    ########################
    # VARIABLES & PARAMETERS
    ########################

    # Parameters
    w_tr = cp.Parameter(nonneg=True, name="w_tr")
    lam_cost = cp.Parameter(nonneg=True, name="lam_cost")
    lam_vc = cp.Parameter((settings.scp.n - 1, settings.sim.n_states), nonneg=True, name="lam_vc")

    # State
    x = cp.Variable((settings.scp.n, settings.sim.n_states), name="x")  # Current State
    dx = cp.Variable((settings.scp.n, settings.sim.n_states), name="dx")  # State Error
    x_bar = cp.Parameter(
        (settings.scp.n, settings.sim.n_states), name="x_bar"
    )  # Previous SCP State
    x_init = cp.Parameter(settings.sim.n_states, name="x_init")  # Initial State
    x_term = cp.Parameter(settings.sim.n_states, name="x_term")  # Final State

    # Affine Scaling for State
    S_x = settings.sim.S_x
    inv_S_x = settings.sim.inv_S_x
    c_x = settings.sim.c_x

    # Control
    u = cp.Variable((settings.scp.n, settings.sim.n_controls), name="u")  # Current Control
    du = cp.Variable((settings.scp.n, settings.sim.n_controls), name="du")  # Control Error
    u_bar = cp.Parameter(
        (settings.scp.n, settings.sim.n_controls), name="u_bar"
    )  # Previous SCP Control

    # Affine Scaling for Control
    S_u = settings.sim.S_u
    inv_S_u = settings.sim.inv_S_u
    c_u = settings.sim.c_u

    # Discretized Augmented Dynamics Constraints
    A_d = cp.Parameter(
        (settings.scp.n - 1, (settings.sim.n_states) * (settings.sim.n_states)), name="A_d"
    )
    B_d = cp.Parameter(
        (settings.scp.n - 1, settings.sim.n_states * settings.sim.n_controls), name="B_d"
    )
    C_d = cp.Parameter(
        (settings.scp.n - 1, settings.sim.n_states * settings.sim.n_controls), name="C_d"
    )
    x_prop = cp.Parameter((settings.scp.n - 1, settings.sim.n_states), name="x_prop")
    nu = cp.Variable((settings.scp.n - 1, settings.sim.n_states), name="nu")  # Virtual Control

    # Linearized Nonconvex Nodal Constraints
    g = []
    grad_g_x = []
    grad_g_u = []
    nu_vb = []
    if settings.sim.constraints_nodal:
        for idx_ncvx, constraint in enumerate(settings.sim.constraints_nodal):
            g.append(cp.Parameter(settings.scp.n, name="g_" + str(idx_ncvx)))
            grad_g_x.append(
                cp.Parameter(
                    (settings.scp.n, settings.sim.n_states), name="grad_g_x_" + str(idx_ncvx)
                )
            )
            grad_g_u.append(
                cp.Parameter(
                    (settings.scp.n, settings.sim.n_controls), name="grad_g_u_" + str(idx_ncvx)
                )
            )
            nu_vb.append(
                cp.Variable(settings.scp.n, name="nu_vb_" + str(idx_ncvx))
            )  # Virtual Control for VB

    # Applying the affine scaling to state and control
    x_nonscaled = []
    u_nonscaled = []
    dx_nonscaled = []
    du_nonscaled = []
    for k in range(settings.scp.n):
        x_nonscaled.append(S_x @ x[k] + c_x)
        u_nonscaled.append(S_u @ u[k] + c_u)
        dx_nonscaled.append(S_x @ dx[k])
        du_nonscaled.append(S_u @ du[k])

    return {
        "w_tr": w_tr,
        "lam_cost": lam_cost,
        "lam_vc": lam_vc,
        "x": x,
        "dx": dx,
        "x_bar": x_bar,
        "x_init": x_init,
        "x_term": x_term,
        "u": u,
        "du": du,
        "u_bar": u_bar,
        "A_d": A_d,
        "B_d": B_d,
        "C_d": C_d,
        "x_prop": x_prop,
        "nu": nu,
        "g": g,
        "grad_g_x": grad_g_x,
        "grad_g_u": grad_g_u,
        "nu_vb": nu_vb,
        "S_x": S_x,
        "inv_S_x": inv_S_x,
        "c_x": c_x,
        "S_u": S_u,
        "inv_S_u": inv_S_u,
        "c_u": c_u,
        "x_nonscaled": x_nonscaled,
        "u_nonscaled": u_nonscaled,
        "dx_nonscaled": dx_nonscaled,
        "du_nonscaled": du_nonscaled,
    }


def lower_convex_constraints(
    constraints_nodal_convex, ocp_vars: Dict, params: Dict = None
) -> tuple[List[cp.Constraint], Dict[str, cp.Parameter]]:
    """Phase 2: Lower symbolic convex constraints to CVXPy constraints with node-awareness.

    Note: One symbolic constraint applied at N nodes becomes N CVXPy constraints.
    The CVXPy variables x and u are already (n_nodes, n_states/n_controls) shaped,
    so we apply constraints at specific nodes using x[k] and u[k].

    Args:
        constraints_nodal_convex: List of convex constraints to lower
        ocp_vars: Dictionary of CVXPy variables
        params: Optional dictionary of parameter values to override the defaults

    Returns:
        Tuple of (list of CVXPy constraints, dict of CVXPy Parameter objects)
    """
    from openscvx.symbolic.expr import Parameter, traverse
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State
    from openscvx.symbolic.lowerers.cvxpy import lower_to_cvxpy

    if not constraints_nodal_convex:
        return [], {}

    # Get the full trajectory CVXPy variables (n_nodes, n_states/n_controls)
    x_nonscaled = ocp_vars["x_nonscaled"]  # List of x_nonscaled[k] for each node k
    u_nonscaled = ocp_vars["u_nonscaled"]  # List of u_nonscaled[k] for each node k

    # Collect all unique Parameters across all constraints and create cp.Parameter objects
    all_params = {}

    def collect_params(expr):
        if isinstance(expr, Parameter):
            if expr.name not in all_params:
                # Use value from params dict if provided, otherwise use Parameter's initial value
                if params and expr.name in params:
                    param_value = params[expr.name]
                else:
                    param_value = expr.value

                cvx_param = cp.Parameter(expr.shape, value=param_value, name=expr.name)
                all_params[expr.name] = cvx_param

    # Collect all parameters from all constraints
    for constraint in constraints_nodal_convex:
        traverse(constraint.constraint, collect_params)

    cvxpy_constraints = []

    for constraint in constraints_nodal_convex:
        # nodes should already be validated and normalized in preprocessing
        nodes = constraint.nodes

        # Collect all State and Control variables referenced in the constraint
        state_vars = {}
        control_vars = {}

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars[expr.name] = expr
            elif isinstance(expr, Control):
                control_vars[expr.name] = expr

        traverse(constraint.constraint, collect_vars)

        # Apply the constraint at each specified node
        for node in nodes:
            # Create simplified variable map for this specific node
            # The CVXPy lowerer will use _slice attributes to extract the right portions
            variable_map = {}

            # Only add state vector if we have state variables in this constraint
            if state_vars:
                variable_map["x"] = x_nonscaled[node]  # Full state vector at this node

            # Only add control vector if we have control variables in this constraint
            if control_vars:
                variable_map["u"] = u_nonscaled[node]  # Full control vector at this node

            # Add all CVXPy Parameter objects to the variable map
            variable_map.update(all_params)

            # Verify all variables have slices (should be guaranteed by preprocessing)
            for state_name, state_var in state_vars.items():
                if state_var._slice is None:
                    raise ValueError(
                        f"State variable '{state_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline - "
                        f"collect_and_assign_slices should have assigned slices to all variables."
                    )

            for control_name, control_var in control_vars.items():
                if control_var._slice is None:
                    raise ValueError(
                        f"Control variable '{control_name}' has no slice assigned. "
                        f"This indicates a bug in the preprocessing pipeline - "
                        f"collect_and_assign_slices should have assigned slices to all variables."
                    )

            # Lower the constraint to CVXPy using existing infrastructure
            # This creates one CVXPy constraint for this specific node
            cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
            cvxpy_constraints.append(cvxpy_constraint)

    return cvxpy_constraints, all_params


def OptimalControlProblem(settings: Config, ocp_vars: Dict):
    """Phase 3: Build the complete optimal control problem with all constraints."""
    # Extract variables from the dict for easier access
    w_tr = ocp_vars["w_tr"]
    lam_cost = ocp_vars["lam_cost"]
    lam_vc = ocp_vars["lam_vc"]
    x = ocp_vars["x"]
    dx = ocp_vars["dx"]
    x_bar = ocp_vars["x_bar"]
    x_init = ocp_vars["x_init"]
    x_term = ocp_vars["x_term"]
    u = ocp_vars["u"]
    du = ocp_vars["du"]
    u_bar = ocp_vars["u_bar"]
    A_d = ocp_vars["A_d"]
    B_d = ocp_vars["B_d"]
    C_d = ocp_vars["C_d"]
    x_prop = ocp_vars["x_prop"]
    nu = ocp_vars["nu"]
    g = ocp_vars["g"]
    grad_g_x = ocp_vars["grad_g_x"]
    grad_g_u = ocp_vars["grad_g_u"]
    nu_vb = ocp_vars["nu_vb"]
    S_x = ocp_vars["S_x"]
    c_x = ocp_vars["c_x"]
    S_u = ocp_vars["S_u"]
    c_u = ocp_vars["c_u"]
    x_nonscaled = ocp_vars["x_nonscaled"]
    u_nonscaled = ocp_vars["u_nonscaled"]
    dx_nonscaled = ocp_vars["dx_nonscaled"]
    du_nonscaled = ocp_vars["du_nonscaled"]

    constr = []
    cost = lam_cost * 0

    #############
    # CONSTRAINTS
    #############

    # Linearized nodal constraints
    idx_ncvx = 0
    if settings.sim.constraints_nodal:
        for constraint in settings.sim.constraints_nodal:
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

    # Convex nodal constraints (already lowered to CVXPy in trajoptproblem)
    if settings.sim.constraints_nodal_convex:
        constr += settings.sim.constraints_nodal_convex

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
        == cp.reshape(A_d[i - 1], (settings.sim.n_states, settings.sim.n_states))
        @ dx_nonscaled[i - 1]
        + cp.reshape(B_d[i - 1], (settings.sim.n_states, settings.sim.n_controls))
        @ du_nonscaled[i - 1]
        + cp.reshape(C_d[i - 1], (settings.sim.n_states, settings.sim.n_controls)) @ du_nonscaled[i]
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
    if settings.sim.constraints_nodal:
        for constraint in settings.sim.constraints_nodal:
            # if not constraint.convex:
            cost += settings.scp.lam_vb * cp.sum(cp.pos(nu_vb[idx_ncvx]))
            idx_ncvx += 1

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
            # Prompt the use to indicate if they wish to overwrite the solver
            # directory or use the existing compiled solver
            if settings.cvx.cvxpygen_override:
                cpg.generate_code(prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True)
            else:
                overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
                if overwrite.lower() == "y":
                    cpg.generate_code(
                        prob, solver=settings.cvx.solver, code_dir="solver", wrapper=True
                    )
                else:
                    pass
    return prob
