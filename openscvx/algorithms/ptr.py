import pickle
import time
import warnings
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import numpy.linalg as la

from openscvx.config import Config

from .autotuning import update_scp_weights
from .optimization_results import OptimizationResults
from .solver_state import SolverState

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

warnings.filterwarnings("ignore")


def PTR_init(
    params,
    ocp: cp.Problem,
    discretization_solver: callable,
    settings: Config,
    jax_constraints: "LoweredJaxConstraints",
):
    if settings.cvx.cvxpygen:
        try:
            from solver.cpg_solver import cpg_solve

            with open("solver/problem.pickle", "rb") as f:
                pickle.load(f)
        except ImportError:
            raise ImportError(
                "cvxpygen solver not found. Make sure cvxpygen is installed and code generation has"
                " been run. Install with: pip install openscvx[cvxpygen]"
            )
    else:
        cpg_solve = None

    if "x_init" in ocp.param_dict:
        ocp.param_dict["x_init"].value = settings.sim.x.initial

    if "x_term" in ocp.param_dict:
        ocp.param_dict["x_term"].value = settings.sim.x.final

    # Create temporary state for initialization solve
    init_state = SolverState.from_settings(settings)

    # Solve a dumb problem to initialize DPP and JAX jacobians
    _ = PTR_subproblem(
        params.items(),
        cpg_solve,
        init_state,
        discretization_solver,
        ocp,
        settings,
        jax_constraints,
    )

    return cpg_solve


def format_result(problem, state: "SolverState", converged: bool) -> OptimizationResults:
    """Formats the solver state as an OptimizationResults object.

    Directly passes trajectory arrays from solver state to results - no object
    construction needed. Results store pure arrays, settings store metadata.

    Args:
        problem: The Problem instance (for symbolic metadata and settings).
        state: The SolverState to extract results from.
        converged: Whether the optimization converged.

    Returns:
        OptimizationResults containing the solution data.
    """
    # Build nodes dictionary with all states and controls
    nodes_dict = {}

    # Add all states (user-defined and augmented)
    for sym_state in problem.symbolic.states:
        nodes_dict[sym_state.name] = state.x[:, sym_state._slice]

    # Add all controls (user-defined and augmented)
    for control in problem.symbolic.controls:
        nodes_dict[control.name] = state.u[:, control._slice]

    return OptimizationResults(
        converged=converged,
        t_final=state.x[:, problem.settings.sim.time_slice][-1],
        nodes=nodes_dict,
        trajectory={},  # Populated by post_process
        _states=problem.symbolic.states_prop,  # Use propagation states for trajectory dict
        _controls=problem.symbolic.controls,
        X=state.X,  # Single source of truth - x and u are properties
        U=state.U,
        discretization_history=state.V_history,
        J_tr_history=state.J_tr,
        J_vb_history=state.J_vb,
        J_vc_history=state.J_vc,
        TR_history=state.TR_history,
        VC_history=state.VC_history,
    )


def PTR_step(
    params,
    settings: Config,
    state: SolverState,
    prob: cp.Problem,
    discretization_solver: callable,
    cpg_solve,
    emitter_function,
    jax_constraints: "LoweredJaxConstraints",
) -> bool:
    """Performs a single SCP iteration.

    Args:
        params: Problem parameters
        settings: Configuration object
        state: Solver state (mutated in place)
        prob: CVXPy problem
        discretization_solver: Discretization solver function
        cpg_solve: CVXPyGen solver (if enabled)
        emitter_function: Function to emit iteration data
        jax_constraints: JAX-lowered non-convex constraints

    Returns:
        bool: True if converged, False otherwise
    """
    # Run the subproblem
    (
        x_sol,
        u_sol,
        cost,
        J_total,
        J_vb_vec,
        J_vc_vec,
        J_tr_vec,
        prob_stat,
        V_multi_shoot,
        subprop_time,
        dis_time,
        vc_mat,
        tr_mat,
    ) = PTR_subproblem(
        params.items(),
        cpg_solve,
        state,
        discretization_solver,
        prob,
        settings,
        jax_constraints,
    )

    # Update state in place by appending to history
    # The x_guess/u_guess properties will automatically return the latest entry
    state.V_history.append(V_multi_shoot)
    state.X.append(x_sol)
    state.U.append(u_sol)
    state.VC_history.append(vc_mat)
    state.TR_history.append(tr_mat)

    state.J_tr = np.sum(np.array(J_tr_vec))
    state.J_vb = np.sum(np.array(J_vb_vec))
    state.J_vc = np.sum(np.array(J_vc_vec))

    # Update weights in state
    update_scp_weights(state, settings, state.k)

    # Emit data
    emitter_function(
        {
            "iter": state.k,
            "dis_time": dis_time * 1000.0,
            "subprop_time": subprop_time * 1000.0,
            "J_total": J_total,
            "J_tr": state.J_tr,
            "J_vb": state.J_vb,
            "J_vc": state.J_vc,
            "cost": cost[-1],
            "prob_stat": prob_stat,
        }
    )

    # Increment iteration counter
    state.k += 1

    # Return convergence status
    return (
        (state.J_tr < settings.scp.ep_tr)
        and (state.J_vb < settings.scp.ep_vb)
        and (state.J_vc < settings.scp.ep_vc)
    )


def PTR_subproblem(
    params,
    cpg_solve,
    state: SolverState,
    aug_dy,
    prob,
    settings: Config,
    jax_constraints: "LoweredJaxConstraints",
):
    prob.param_dict["x_bar"].value = state.x
    prob.param_dict["u_bar"].value = state.u

    # Convert parameters to dictionary
    param_dict = dict(params)

    t0 = time.time()
    A_bar, B_bar, C_bar, x_prop, V_multi_shoot = aug_dy.call(
        state.x, state.u.astype(float), param_dict
    )

    prob.param_dict["A_d"].value = A_bar.__array__()
    prob.param_dict["B_d"].value = B_bar.__array__()
    prob.param_dict["C_d"].value = C_bar.__array__()
    prob.param_dict["x_prop"].value = x_prop.__array__()
    dis_time = time.time() - t0

    # Update nodal constraint linearization parameters
    # TODO: (norrisg) investigate why we are passing `0` for the node here
    if jax_constraints.nodal:
        for g_id, constraint in enumerate(jax_constraints.nodal):
            prob.param_dict["g_" + str(g_id)].value = np.asarray(
                constraint.func(state.x, state.u, 0, param_dict)
            )
            prob.param_dict["grad_g_x_" + str(g_id)].value = np.asarray(
                constraint.grad_g_x(state.x, state.u, 0, param_dict)
            )
            prob.param_dict["grad_g_u_" + str(g_id)].value = np.asarray(
                constraint.grad_g_u(state.x, state.u, 0, param_dict)
            )

    # Update cross-node constraint linearization parameters
    if jax_constraints.cross_node:
        for g_id, constraint in enumerate(jax_constraints.cross_node):
            # Cross-node constraints take (X, U, params) not (x, u, node, params)
            prob.param_dict["g_cross_" + str(g_id)].value = np.asarray(
                constraint.func(state.x, state.u, param_dict)
            )
            prob.param_dict["grad_g_X_cross_" + str(g_id)].value = np.asarray(
                constraint.grad_g_X(state.x, state.u, param_dict)
            )
            prob.param_dict["grad_g_U_cross_" + str(g_id)].value = np.asarray(
                constraint.grad_g_U(state.x, state.u, param_dict)
            )

    # Convex constraints are already lowered and handled in the OCP, no action needed here

    # Initialize lam_vc as matrix if it's still a scalar in state
    if isinstance(state.lam_vc, (int, float)):
        # Convert scalar to matrix: (N-1, n_states)
        state.lam_vc = np.ones((settings.scp.n - 1, settings.sim.n_states)) * state.lam_vc

    # Update CVXPy parameters from state
    prob.param_dict["w_tr"].value = state.w_tr
    prob.param_dict["lam_cost"].value = state.lam_cost
    prob.param_dict["lam_vc"].value = state.lam_vc
    prob.param_dict["lam_vb"].value = state.lam_vb

    if settings.cvx.cvxpygen:
        t0 = time.time()
        prob.register_solve("CPG", cpg_solve)
        prob.solve(method="CPG", **settings.cvx.solver_args)
        subprop_time = time.time() - t0
    else:
        t0 = time.time()
        prob.solve(solver=settings.cvx.solver, **settings.cvx.solver_args)
        subprop_time = time.time() - t0

    x_new_guess = (
        settings.sim.S_x @ prob.var_dict["x"].value.T + np.expand_dims(settings.sim.c_x, axis=1)
    ).T
    u_new_guess = (
        settings.sim.S_u @ prob.var_dict["u"].value.T + np.expand_dims(settings.sim.c_u, axis=1)
    ).T

    # Calculate costs from boundary conditions using utility function
    # Note: The original code only considered final_type, but the utility handles both
    # Here we maintain backward compatibility by only using final_type
    costs = [0]
    for i, bc_type in enumerate(settings.sim.x.final_type):
        if bc_type == "Minimize":
            costs += x_new_guess[:, i]
        elif bc_type == "Maximize":
            costs -= x_new_guess[:, i]

    # Create the block diagonal matrix using jax.numpy.block
    inv_block_diag = np.block(
        [
            [
                settings.sim.inv_S_x,
                np.zeros((settings.sim.inv_S_x.shape[0], settings.sim.inv_S_u.shape[1])),
            ],
            [
                np.zeros((settings.sim.inv_S_u.shape[0], settings.sim.inv_S_x.shape[1])),
                settings.sim.inv_S_u,
            ],
        ]
    )

    # Calculate J_tr_vec using the JAX-compatible block diagonal matrix
    tr_mat = inv_block_diag @ np.hstack((x_new_guess - state.x, u_new_guess - state.u)).T
    J_tr_vec = la.norm(tr_mat, axis=0) ** 2
    vc_mat = np.abs(prob.var_dict["nu"].value)
    J_vc_vec = np.sum(vc_mat, axis=1)

    id_ncvx = 0
    J_vb_vec = 0
    if jax_constraints.nodal:
        for constraint in jax_constraints.nodal:
            J_vb_vec += np.maximum(0, prob.var_dict["nu_vb_" + str(id_ncvx)].value)
            id_ncvx += 1

    # Add cross-node constraint violations
    id_cross = 0
    if jax_constraints.cross_node:
        for constraint in jax_constraints.cross_node:
            J_vb_vec += np.maximum(0, prob.var_dict["nu_vb_cross_" + str(id_cross)].value)
            id_cross += 1

    # Convex constraints are already handled in the OCP, no processing needed here
    return (
        x_new_guess,
        u_new_guess,
        costs,
        prob.value,
        J_vb_vec,
        J_vc_vec,
        J_tr_vec,
        prob.status,
        V_multi_shoot,
        subprop_time,
        dis_time,
        vc_mat,
        abs(tr_mat),
    )
