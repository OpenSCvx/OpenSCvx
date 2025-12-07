import os
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from openscvx.config import Config

if TYPE_CHECKING:
    from openscvx.lowered import LoweredProblem

# Optional cvxpygen import
try:
    from cvxpygen import cpg

    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False
    cpg = None


def optimal_control_problem(settings: Config, lowered: "LoweredProblem"):
    """Build the complete optimal control problem with all constraints.

    Args:
        settings: Configuration settings for the optimization problem
        lowered: LoweredProblem containing ocp_vars and lowered constraints
    """
    # Extract typed CVXPy variables from LoweredProblem
    ocp_vars = lowered.ocp_vars

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
