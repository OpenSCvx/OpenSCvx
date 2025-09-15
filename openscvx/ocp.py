import os
from typing import Dict, List

import cvxpy as cp
import numpy as np
import numpy.linalg as la
from numpy import block

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
    z_d = cp.Parameter((settings.scp.n - 1, settings.sim.n_states), name="z_d")
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
    for k in range(settings.scp.n):
        x_nonscaled.append(S_x @ x[k] + c_x)
        u_nonscaled.append(S_u @ u[k] + c_u)

    return {
        "w_tr": w_tr,
        "lam_cost": lam_cost,
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
        "z_d": z_d,
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
    }


def lower_convex_constraints(constraints_nodal_convex, ocp_vars: Dict) -> List[cp.Constraint]:
    """Phase 2: Lower symbolic convex constraints to CVXPy constraints with node-awareness.

    Note: One symbolic constraint applied at N nodes becomes N CVXPy constraints.
    The CVXPy variables x and u are already (n_nodes, n_states/n_controls) shaped,
    so we apply constraints at specific nodes using x[k] and u[k].
    """
    from openscvx.backend.control import Control
    from openscvx.backend.expr import traverse
    from openscvx.backend.lowerers.cvxpy import lower_to_cvxpy
    from openscvx.backend.state import State

    if not constraints_nodal_convex:
        return []

    # TODO: (norrisg) This does not work. Fix. :(
    x_nonscaled = ocp_vars["x_nonscaled"]  # List of x_nonscaled[k] for each node k
    u_nonscaled = ocp_vars["u_nonscaled"]  # List of u_nonscaled[k] for each node k

    cvxpy_constraints = []

    for constraint in constraints_nodal_convex:
        # nodes should already be validated and normalized in preprocessing
        nodes = constraint.nodes

        # Collect all State and Control variables referenced in the constraint
        state_vars = set()
        control_vars = set()

        def collect_vars(expr):
            if isinstance(expr, State):
                state_vars.add(expr.name)
            elif isinstance(expr, Control):
                control_vars.add(expr.name)

        traverse(constraint.constraint, collect_vars)

        # Apply the constraint at each specified node
        for node in nodes:
            # Create variable map for this specific node (row k of the trajectory)
            variable_map = {}

            # Map state variables to x_nonscaled[node] (the state at node k)
            # The CVXPy lowerer will handle _slice attributes automatically
            for state_name in state_vars:
                variable_map[state_name] = x_nonscaled[node]  # x_nonscaled[k, :]

            # Map control variables to u_nonscaled[node] (the control at node k)
            # The CVXPy lowerer will handle _slice attributes automatically
            for control_name in control_vars:
                variable_map[control_name] = u_nonscaled[node]  # u_nonscaled[k, :]

            # Lower the constraint to CVXPy using existing infrastructure
            # This creates one CVXPy constraint for this specific node
            cvxpy_constraint = lower_to_cvxpy(constraint.constraint, variable_map)
            cvxpy_constraints.append(cvxpy_constraint)

    return cvxpy_constraints


def OptimalControlProblem(settings: Config, ocp_vars: Dict):
    """Phase 3: Build the complete optimal control problem with all constraints."""
    # Extract variables from the dict for easier access
    w_tr = ocp_vars["w_tr"]
    lam_cost = ocp_vars["lam_cost"]
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
    z_d = ocp_vars["z_d"]
    nu = ocp_vars["nu"]
    g = ocp_vars["g"]
    grad_g_x = ocp_vars["grad_g_x"]
    grad_g_u = ocp_vars["grad_g_u"]
    nu_vb = ocp_vars["nu_vb"]
    S_x = ocp_vars["S_x"]
    inv_S_x = ocp_vars["inv_S_x"]
    c_x = ocp_vars["c_x"]
    S_u = ocp_vars["S_u"]
    inv_S_u = ocp_vars["inv_S_u"]
    c_u = ocp_vars["c_u"]
    x_nonscaled = ocp_vars["x_nonscaled"]
    u_nonscaled = ocp_vars["u_nonscaled"]

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

    for i in range(settings.sim.idx_x_true.start, settings.sim.idx_x_true.stop):
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
            u_nonscaled[i][settings.sim.idx_s] == u_nonscaled[i - 1][settings.sim.idx_s]
            for i in range(1, settings.scp.n)
        ]

    constr += [
        la.inv(S_x) @ (x_nonscaled[i] - x_bar[i] - dx[i]) == 0 for i in range(settings.scp.n)
    ]  # State Error
    constr += [
        la.inv(S_u) @ (u_nonscaled[i] - u_bar[i] - du[i]) == 0 for i in range(settings.scp.n)
    ]  # Control Error

    constr += [
        x_nonscaled[i]
        == cp.reshape(A_d[i - 1], (settings.sim.n_states, settings.sim.n_states))
        @ x_nonscaled[i - 1]
        + cp.reshape(B_d[i - 1], (settings.sim.n_states, settings.sim.n_controls))
        @ u_nonscaled[i - 1]
        + cp.reshape(C_d[i - 1], (settings.sim.n_states, settings.sim.n_controls)) @ u_nonscaled[i]
        + z_d[i - 1]
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

    inv = block(
        [
            [inv_S_x, np.zeros((S_x.shape[0], S_u.shape[1]))],
            [np.zeros((S_u.shape[0], S_x.shape[1])), inv_S_u],
        ]
    )
    cost += sum(
        w_tr * cp.sum_squares(inv @ cp.hstack((dx[i], du[i]))) for i in range(settings.scp.n)
    )  # Trust Region Cost
    cost += sum(
        settings.scp.lam_vc * cp.sum(cp.abs(nu[i - 1])) for i in range(1, settings.scp.n)
    )  # Virtual Control Slack

    idx_ncvx = 0
    if settings.sim.constraints_nodal:
        for constraint in settings.sim.constraints_nodal:
            # if not constraint.convex:
            cost += settings.scp.lam_vb * cp.sum(cp.pos(nu_vb[idx_ncvx]))
            idx_ncvx += 1

    for idx, nodes in zip(
        np.arange(settings.sim.idx_y.start, settings.sim.idx_y.stop),
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
