import numpy as np
import numpy.linalg as la
import cvxpy as cp
import pickle
import time

from openscvx.backend.parameter import Parameter
from openscvx.config import Config
from openscvx.results import OptimizationResults

import warnings
warnings.filterwarnings("ignore")

def PTR_init(params, ocp: cp.Problem, discretization_solver: callable, settings: Config):
    if settings.cvx.cvxpygen:
        try:
            from solver.cpg_solver import cpg_solve
            with open('solver/problem.pickle', 'rb') as f:
                prob = pickle.load(f)
        except ImportError:
            raise ImportError(
                "cvxpygen solver not found. Make sure cvxpygen is installed and code generation has been run. "
                "Install with: pip install openscvx[cvxpygen]"
            )
    else:
        cpg_solve = None

    if 'x_init' in ocp.param_dict:
        ocp.param_dict['x_init'].value = settings.sim.x.initial
    
    if 'x_term' in ocp.param_dict:
        ocp.param_dict['x_term'].value = settings.sim.x.final

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(params.items(), cpg_solve, settings.sim.x, settings.sim.u, discretization_solver, ocp, settings)

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

def PTR_main(params, settings: Config, prob: cp.Problem, aug_dy: callable, cpg_solve, emitter_function) -> OptimizationResults:
    J_vb = 1E2
    J_vc = 1E2
    J_tr = 1E2

    x = settings.sim.x
    u = settings.sim.u

    if 'x_init' in prob.param_dict:
        prob.param_dict['x_init'].value = settings.sim.x.initial
    
    if 'x_term' in prob.param_dict:
        prob.param_dict['x_term'].value = settings.sim.x.final


    scp_trajs = [x.guess]
    scp_controls = [u.guess]
    V_multi_shoot_traj = []

    k = 1

    while k <= settings.scp.k_max and ((J_tr >= settings.scp.ep_tr) or (J_vb >= settings.scp.ep_vb) or (J_vc >= settings.scp.ep_vc)):
        x_sol, u_sol, cost, J_total, J_vb_vec, J_vc_vec, J_tr_vec, prob_stat, V_multi_shoot, subprop_time, dis_time = PTR_subproblem(params.items(), cpg_solve, x, u, aug_dy, prob, settings)

        V_multi_shoot_traj.append(V_multi_shoot)

        x.guess = x_sol
        u.guess = u_sol

        J_tr = np.sum(np.array(J_tr_vec))
        J_vb = np.sum(np.array(J_vb_vec))
        J_vc = np.sum(np.array(J_vc_vec))
        scp_trajs.append(x.guess)
        scp_controls.append(u.guess)

        settings.scp.w_tr = min(settings.scp.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)
        if k > settings.scp.cost_drop:
            settings.scp.lam_cost = settings.scp.lam_cost * settings.scp.cost_relax

        emitter_function(
            {
                "iter": k,
                "dis_time": dis_time * 1000.0,
                "subprop_time": subprop_time * 1000.0,
                "J_total": J_total,
                "J_tr": J_tr,
                "J_vb": J_vb,
                "J_vc": J_vc,
                "cost": cost[-1],
                "prob_stat": prob_stat,
            }
        )

        k += 1

    result = OptimizationResults(
        converged=k <= settings.scp.k_max,
        t_final=x.guess[:,settings.sim.idx_t][-1],
        u=u,
        x=x,
        x_history=scp_trajs,
        u_history=scp_controls,
        discretization_history=V_multi_shoot_traj,
        J_tr_history=J_tr_vec,
        J_vb_history=J_vb_vec,
        J_vc_history=J_vc_vec,
    )
    
    return result

def PTR_subproblem(params, cpg_solve, x, u, aug_dy, prob, settings: Config):
    prob.param_dict['x_bar'].value = x.guess
    prob.param_dict['u_bar'].value = u.guess
    
    # Make a tuple from list of parameter values
    param_values = tuple([param.value for _, param in params])

    t0 = time.time()
    A_bar, B_bar, C_bar, z_bar, V_multi_shoot = aug_dy.call(x.guess, u.guess.astype(float), *param_values)

    prob.param_dict['A_d'].value = A_bar.__array__()
    prob.param_dict['B_d'].value = B_bar.__array__()
    prob.param_dict['C_d'].value = C_bar.__array__()
    prob.param_dict['z_d'].value = z_bar.__array__()
    dis_time = time.time() - t0

    if settings.sim.constraints_nodal:
        for g_id, constraint in enumerate(settings.sim.constraints_nodal):
            if not constraint.convex:
                prob.param_dict['g_' + str(g_id)].value = np.asarray(constraint.g(x.guess, u.guess))
                prob.param_dict['grad_g_x_' + str(g_id)].value = np.asarray(constraint.grad_g_x(x.guess, u.guess))
                prob.param_dict['grad_g_u_' + str(g_id)].value = np.asarray(constraint.grad_g_u(x.guess, u.guess))
    
    prob.param_dict['w_tr'].value = settings.scp.w_tr
    prob.param_dict['lam_cost'].value = settings.scp.lam_cost

    if settings.cvx.cvxpygen:
        t0 = time.time()
        prob.register_solve('CPG', cpg_solve)
        prob.solve(method = 'CPG', **settings.cvx.solver_args)
        subprop_time = time.time() - t0
    else:
        t0 = time.time()
        prob.solve(solver = settings.cvx.solver, **settings.cvx.solver_args)
        subprop_time = time.time() - t0

    x_new_guess = (settings.sim.S_x @ prob.var_dict['x'].value.T + np.expand_dims(settings.sim.c_x, axis = 1)).T
    u_new_guess = (settings.sim.S_u @ prob.var_dict['u'].value.T + np.expand_dims(settings.sim.c_u, axis = 1)).T

    i = 0
    costs = [0]
    for type in x.final_type:
        if type == 'Minimize':
            costs += x_new_guess[:,i]
        if type == 'Maximize':
            costs -= x_new_guess[:,i]
        i += 1

    # Create the block diagonal matrix using jax.numpy.block
    inv_block_diag = np.block([
        [settings.sim.inv_S_x, np.zeros((settings.sim.inv_S_x.shape[0], settings.sim.inv_S_u.shape[1]))],
        [np.zeros((settings.sim.inv_S_u.shape[0], settings.sim.inv_S_x.shape[1])), settings.sim.inv_S_u]
    ])

    # Calculate J_tr_vec using the JAX-compatible block diagonal matrix
    J_tr_vec = la.norm(inv_block_diag @ np.hstack((x_new_guess - x.guess, u_new_guess - u.guess)).T, axis=0)**2
    J_vc_vec = np.sum(np.abs(prob.var_dict['nu'].value), axis = 1)
    
    id_ncvx = 0
    J_vb_vec = 0
    for constraint in settings.sim.constraints_nodal:
        if constraint.convex == False:
            J_vb_vec += np.maximum(0, prob.var_dict['nu_vb_' + str(id_ncvx)].value)
            id_ncvx += 1
    return x_new_guess, u_new_guess, costs, prob.value, J_vb_vec, J_vc_vec, J_tr_vec, prob.status, V_multi_shoot, subprop_time, dis_time
