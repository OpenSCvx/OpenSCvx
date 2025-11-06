import pickle
import time
import warnings

import cvxpy as cp
import numpy as np
import numpy.linalg as la

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

    # Solve a dumb problem to initialize DPP and JAX jacobians
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
    return OptimizationResults(
        converged=converged,
        t_final=problem.settings.sim.x.guess[:, problem.settings.sim.idx_t][-1],
        u=problem.settings.sim.u,
        x=problem.settings.sim.x,
        x_history=problem.scp_trajs,
        u_history=problem.scp_controls,
        discretization_history=problem.scp_V_multi_shoot_traj,
        J_tr_history=problem.scp_J_tr,
        J_vb_history=problem.scp_J_vb,
        J_vc_history=problem.scp_J_vc,
    )


def PTR_subproblem(params, cpg_solve, x, u, aug_dy, prob, settings: Config):
    prob.param_dict["x_bar"].value = x.guess
    prob.param_dict["u_bar"].value = u.guess

    # Make a tuple from list of parameter values
    param_values = tuple([param.value for _, param in params])

    t0 = time.time()
    A_bar, B_bar, C_bar, x_prop, V_multi_shoot = aug_dy.call(
        x.guess, u.guess.astype(float), *param_values
    )

    prob.param_dict["x_prop"].value = x_prop.__array__()
    prob.param_dict["A_d"].value = A_bar.__array__()
    prob.param_dict["B_d"].value = B_bar.__array__()
    prob.param_dict["C_d"].value = C_bar.__array__()
    dis_time = time.time() - t0

    if settings.sim.constraints_nodal:
        for g_id, constraint in enumerate(settings.sim.constraints_nodal):
            if not constraint.convex:
                prob.param_dict["g_" + str(g_id)].value = np.asarray(constraint.g(x.guess, u.guess))
                prob.param_dict["grad_g_x_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_x(x.guess, u.guess)
                )
                prob.param_dict["grad_g_u_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_u(x.guess, u.guess)
                )

    prob.param_dict["w_tr"].value = settings.scp.w_tr
    prob.param_dict["lam_cost"].value = settings.scp.lam_cost

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
    for final_type in x.final_type:
        if final_type == "Minimize":
            costs += x_new_guess[:, i]
        if final_type == "Maximize":
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
        if not constraint.convex:
            J_vb_vec += np.maximum(0, prob.var_dict["nu_vb_" + str(id_ncvx)].value)
            id_ncvx += 1
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
