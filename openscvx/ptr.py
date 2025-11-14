import pickle
import time
import warnings

import cvxpy as cp
import numpy as np
import numpy.linalg as la

from openscvx.autotuning import update_scp_weights
from openscvx.config import Config
from openscvx.results import OptimizationResults

warnings.filterwarnings("ignore")


def PTR_init(params, ocp: cp.Problem, discretization_solver: callable, settings: Config):
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

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(
        params.items(),
        cpg_solve,
        settings.sim.x,
        settings.sim.u,
        discretization_solver,
        ocp,
        settings,
    )

    return cpg_solve


def format_result(problem, converged: bool) -> OptimizationResults:
    """Formats the final result as an OptimizationResults object from the problem's state."""
    # Build nodes dictionary with all states and controls
    nodes_dict = {}

    # Add all states (user-defined and augmented)
    for state in problem.states:
        nodes_dict[state.name] = problem.settings.sim.x.guess[:, state._slice]

    # Add all controls (user-defined and augmented)
    for control in problem.controls:
        nodes_dict[control.name] = problem.settings.sim.u.guess[:, control._slice]

    return OptimizationResults(
        converged=converged,
        t_final=problem.settings.sim.x.guess[:, problem.settings.sim.time_slice][-1],
        u=problem.settings.sim.u,
        x=problem.settings.sim.x,
        nodes=nodes_dict,
        trajectory={},  # Populated by post_process
        _states=problem.states_prop,  # Use propagation states for trajectory dict
        _controls=problem.controls,
        x_history=problem.scp_trajs,
        u_history=problem.scp_controls,
        discretization_history=problem.scp_V_multi_shoot_traj,
        J_tr_history=problem.scp_J_tr,
        J_vb_history=problem.scp_J_vb,
        J_vc_history=problem.scp_J_vc,
    )


def PTR_step(
    params,
    settings: Config,
    prob: cp.Problem,
    discretization_solver: callable,
    cpg_solve,
    emitter_function,
    scp_k: int,
    scp_J_tr: float,
    scp_J_vb: float,
    scp_J_vc: float,
    scp_trajs: list,
    scp_controls: list,
    scp_V_multi_shoot_traj: list,
) -> dict:
    """Performs a single SCP iteration.

    Args:
        params: Problem parameters
        settings: Configuration object
        prob: CVXPy problem
        discretization_solver: Discretization solver function
        cpg_solve: CVXPyGen solver (if enabled)
        emitter_function: Function to emit iteration data
        scp_k: Current iteration number
        scp_J_tr: Current trust region cost
        scp_J_vb: Current virtual buffer cost
        scp_J_vc: Current virtual control cost
        scp_trajs: List of trajectory history
        scp_controls: List of control history
        scp_V_multi_shoot_traj: List of discretization history

    Returns:
        dict: Updated SCP state and convergence information
    """
    x = settings.sim.x
    u = settings.sim.u

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
    ) = PTR_subproblem(
        params.items(),
        cpg_solve,
        x,
        u,
        discretization_solver,
        prob,
        settings,
    )

    # Update state
    scp_V_multi_shoot_traj.append(V_multi_shoot)
    x.guess = x_sol
    u.guess = u_sol
    scp_trajs.append(x.guess)
    scp_controls.append(u.guess)

    scp_J_tr = np.sum(np.array(J_tr_vec))
    scp_J_vb = np.sum(np.array(J_vb_vec))
    scp_J_vc = np.sum(np.array(J_vc_vec))

    # Update weights
    update_scp_weights(settings, scp_k)

    # Emit data
    emitter_function(
        {
            "iter": scp_k,
            "dis_time": dis_time * 1000.0,
            "subprop_time": subprop_time * 1000.0,
            "J_total": J_total,
            "J_tr": scp_J_tr,
            "J_vb": scp_J_vb,
            "J_vc": scp_J_vc,
            "cost": cost[-1],
            "prob_stat": prob_stat,
        }
    )

    # Return updated state and convergence info
    return {
        "converged": (
            (scp_J_tr < settings.scp.ep_tr)
            and (scp_J_vb < settings.scp.ep_vb)
            and (scp_J_vc < settings.scp.ep_vc)
        ),
        "scp_k": scp_k + 1,
        "scp_J_tr": scp_J_tr,
        "scp_J_vb": scp_J_vb,
        "scp_J_vc": scp_J_vc,
        "u": u,
        "x": x,
        "V_multi_shoot": V_multi_shoot,
    }


def PTR_main(
    params, settings: Config, prob: cp.Problem, aug_dy: callable, cpg_solve, emitter_function
) -> OptimizationResults:
    x = settings.sim.x
    u = settings.sim.u

    if "x_init" in prob.param_dict:
        prob.param_dict["x_init"].value = settings.sim.x.initial

    if "x_term" in prob.param_dict:
        prob.param_dict["x_term"].value = settings.sim.x.final

    # Initialize SCP state
    scp_k = 1
    scp_J_tr = 1e2
    scp_J_vb = 1e2
    scp_J_vc = 1e2
    scp_trajs = [x.guess]
    scp_controls = [u.guess]
    scp_V_multi_shoot_traj = []

    while scp_k <= settings.scp.k_max and (
        (scp_J_tr >= settings.scp.ep_tr)
        or (scp_J_vb >= settings.scp.ep_vb)
        or (scp_J_vc >= settings.scp.ep_vc)
    ):
        result = PTR_step(
            params,
            settings,
            prob,
            aug_dy,
            cpg_solve,
            emitter_function,
            scp_k,
            scp_J_tr,
            scp_J_vb,
            scp_J_vc,
            scp_trajs,
            scp_controls,
            scp_V_multi_shoot_traj,
        )

        # Update state from result
        scp_k = result["scp_k"]
        scp_J_tr = result["scp_J_tr"]
        scp_J_vb = result["scp_J_vb"]
        scp_J_vc = result["scp_J_vc"]

    # Use the final vectors for the result (from the last iteration)
    final_J_tr_vec = [scp_J_tr]
    final_J_vb_vec = [scp_J_vb]
    final_J_vc_vec = [scp_J_vc]

    result = OptimizationResults(
        converged=scp_k <= settings.scp.k_max,
        t_final=x.guess[:, settings.sim.time_slice][-1],
        u=u,
        x=x,
        x_history=scp_trajs,
        u_history=scp_controls,
        discretization_history=scp_V_multi_shoot_traj,
        J_tr_history=final_J_tr_vec,
        J_vb_history=final_J_vb_vec,
        J_vc_history=final_J_vc_vec,
    )

    return result


def PTR_subproblem(params, cpg_solve, x, u, aug_dy, prob, settings: Config):
    prob.param_dict["x_bar"].value = x.guess
    prob.param_dict["u_bar"].value = u.guess

    # Convert parameters to dictionary
    param_dict = dict(params)

    t0 = time.time()
    A_bar, B_bar, C_bar, x_prop, V_multi_shoot = aug_dy.call(
        x.guess, u.guess.astype(float), param_dict
    )

    prob.param_dict["A_d"].value = A_bar.__array__()
    prob.param_dict["B_d"].value = B_bar.__array__()
    prob.param_dict["C_d"].value = C_bar.__array__()
    prob.param_dict["x_prop"].value = x_prop.__array__()
    dis_time = time.time() - t0

    # TODO: (norrisg) investigate why we are passing `0` for the node here
    if settings.sim.constraints_nodal:
        for g_id, constraint in enumerate(settings.sim.constraints_nodal):
            prob.param_dict["g_" + str(g_id)].value = np.asarray(
                constraint.func(x.guess, u.guess, 0, param_dict)
            )
            prob.param_dict["grad_g_x_" + str(g_id)].value = np.asarray(
                constraint.grad_g_x(x.guess, u.guess, 0, param_dict)
            )
            prob.param_dict["grad_g_u_" + str(g_id)].value = np.asarray(
                constraint.grad_g_u(x.guess, u.guess, 0, param_dict)
            )

    # Convex constraints are already lowered and handled in the OCP, no action needed here

    # Initialize lam_vc as matrix if it's still a scalar in settings
    if isinstance(settings.scp.lam_vc, (int, float)):
        # Convert scalar to matrix: (N-1, n_states)
        lam_vc_matrix = np.ones((settings.scp.n - 1, settings.sim.n_states)) * settings.scp.lam_vc
        settings.scp.lam_vc = lam_vc_matrix

    # Update CVXPy parameter
    prob.param_dict["w_tr"].value = settings.scp.w_tr
    prob.param_dict["lam_cost"].value = settings.scp.lam_cost
    prob.param_dict["lam_vc"].value = settings.scp.lam_vc

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

    i = 0
    costs = [0]
    for type in x.final_type:
        if type == "Minimize":
            costs += x_new_guess[:, i]
        if type == "Maximize":
            costs -= x_new_guess[:, i]
        i += 1

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
    J_tr_vec = (
        la.norm(
            inv_block_diag @ np.hstack((x_new_guess - x.guess, u_new_guess - u.guess)).T, axis=0
        )
        ** 2
    )
    J_vc_vec = np.sum(np.abs(prob.var_dict["nu"].value), axis=1)

    id_ncvx = 0
    J_vb_vec = 0
    for constraint in settings.sim.constraints_nodal:
        J_vb_vec += np.maximum(0, prob.var_dict["nu_vb_" + str(id_ncvx)].value)
        id_ncvx += 1
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
    )
