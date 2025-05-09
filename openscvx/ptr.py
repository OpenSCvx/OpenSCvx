import numpy as np
import numpy.linalg as la
import cvxpy as cp
import pickle
import time

from openscvx.config import Config

import warnings
warnings.filterwarnings("ignore")

def PTR_init(ocp: cp.Problem, discretization_solver: callable, params: Config):
    if params.cvx.cvxpygen:
        from solver.cpg_solver import cpg_solve
        with open('solver/problem.pickle', 'rb') as f:
            prob = pickle.load(f)
    else:
        cpg_solve = None

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(cpg_solve, params.sim.x_bar, params.sim.u_bar, discretization_solver, ocp, params)

    return cpg_solve

def PTR_main(params: Config, prob: cp.Problem, aug_dy: callable, cpg_solve, emitter_function) -> dict:
    J_vb = 1E2
    J_vc = 1E2
    J_tr = 1E2

    x_bar = params.sim.x_bar
    u_bar = params.sim.u_bar

    scp_trajs = [x_bar]
    scp_controls = [u_bar]
    V_multi_shoot_traj = []

    k = 1

    while k <= params.scp.k_max and ((J_tr >= params.scp.ep_tr) or (J_vb >= params.scp.ep_vb) or (J_vc >= params.scp.ep_vc)):
        x, u, t, J_total, J_vb_vec, J_vc_vec, J_tr_vec, prob_stat, V_multi_shoot, subprop_time, dis_time = PTR_subproblem(cpg_solve, x_bar, u_bar, aug_dy, prob, params)

        V_multi_shoot_traj.append(V_multi_shoot)

        x_bar = x
        u_bar = u

        J_tr = np.sum(np.array(J_tr_vec))
        J_vb = np.sum(np.array(J_vb_vec))
        J_vc = np.sum(np.array(J_vc_vec))
        scp_trajs.append(x)
        scp_controls.append(u)

        params.scp.w_tr = min(params.scp.w_tr * params.scp.w_tr_adapt, params.scp.w_tr_max)
        if k > params.scp.cost_drop:
            params.scp.lam_cost = params.scp.lam_cost * params.scp.cost_relax

        emitter_function(
            {
                "iter": k,
                "dis_time": dis_time * 1000.0,
                "subprop_time": subprop_time * 1000.0,
                "J_total": J_total,
                "J_tr": J_tr,
                "J_vb": J_vb,
                "J_vc": J_vc,
                "cost": t[-1],
                "prob_stat": prob_stat,
            }
        )

        k += 1

    result = dict(
        converged = k <= params.scp.k_max,
        t_final = x[:,params.sim.idx_t][-1],
        u = u,
        x = x,
        x_history = scp_trajs,
        u_history = scp_controls,
        discretization_history = V_multi_shoot_traj,
        J_tr_history = J_tr_vec,
        J_vb_history = J_vb_vec,
        J_vc_history = J_vc_vec,
    )
    return result


def PTR_subproblem(cpg_solve, x_bar, u_bar, aug_dy, prob, params: Config):
    prob.param_dict['x_bar'].value = x_bar
    prob.param_dict['u_bar'].value = u_bar
    
    t0 = time.time()
    A_bar, B_bar, C_bar, z_bar, V_multi_shoot = aug_dy(x_bar, u_bar.astype(float))

    prob.param_dict['A_d'].value = A_bar.__array__()
    prob.param_dict['B_d'].value = B_bar.__array__()
    prob.param_dict['C_d'].value = C_bar.__array__()
    prob.param_dict['z_d'].value = z_bar.__array__()
    dis_time = time.time() - t0

    if params.sim.constraints_nodal:
        for g_id, constraint in enumerate(params.sim.constraints_nodal):
            if not constraint.convex:
                prob.param_dict['g_' + str(g_id)].value = np.asarray(constraint.g(x_bar, u_bar))
                prob.param_dict['grad_g_x_' + str(g_id)].value = np.asarray(constraint.grad_g_x(x_bar, u_bar))
                prob.param_dict['grad_g_u_' + str(g_id)].value = np.asarray(constraint.grad_g_u(x_bar, u_bar))
    
    prob.param_dict['w_tr'].value = params.scp.w_tr
    prob.param_dict['lam_cost'].value = params.scp.lam_cost

    if params.cvx.cvxpygen:
        t0 = time.time()
        prob.register_solve('CPG', cpg_solve)
        prob.solve(method = 'CPG', **params.cvx.solver_args)
        subprop_time = time.time() - t0
    else:
        t0 = time.time()
        prob.solve(solver = params.cvx.solver, enforce_dpp = True, **params.cvx.solver_args)
        subprop_time = time.time() - t0

    x = (params.sim.S_x @ prob.var_dict['x'].value.T + np.expand_dims(params.sim.c_x, axis = 1)).T
    u = (params.sim.S_u @ prob.var_dict['u'].value.T + np.expand_dims(params.sim.c_u, axis = 1)).T

    i = 0
    costs = [0]
    for type in params.sim.final_state.type:
        if type == 'Minimize':
            costs += x[:,i]
        if type == 'Maximize':
            costs -= x[:,i]
        i += 1

    # Create the block diagonal matrix using jax.numpy.block
    inv_block_diag = np.block([
        [params.sim.inv_S_x, np.zeros((params.sim.inv_S_x.shape[0], params.sim.inv_S_u.shape[1]))],
        [np.zeros((params.sim.inv_S_u.shape[0], params.sim.inv_S_x.shape[1])), params.sim.inv_S_u]
    ])

    # Calculate J_tr_vec using the JAX-compatible block diagonal matrix
    J_tr_vec = la.norm(inv_block_diag @ np.hstack((x - x_bar, u - u_bar)).T, axis=0)**2
    J_vc_vec = np.sum(np.abs(prob.var_dict['nu'].value), axis = 1)
    
    id_ncvx = 0
    J_vb_vec = 0
    for constraint in params.sim.constraints_nodal:
        if constraint.convex == False:
            J_vb_vec += np.maximum(0, prob.var_dict['nu_vb_' + str(id_ncvx)].value)
            id_ncvx += 1
    return x, u, costs, prob.value, J_vb_vec, J_vc_vec, J_tr_vec, prob.status, V_multi_shoot, subprop_time, dis_time
