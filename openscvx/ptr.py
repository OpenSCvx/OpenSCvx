import numpy as np
import numpy.linalg as la
import cvxpy as cp
import pickle
import time
import threading
import queue

import sys
from termcolor import colored

from openscvx.propagation import s_to_t, t_to_tau, simulate_nonlinear_time
from openscvx.config import Config
from openscvx.ocp import OptimalControlProblem
from openscvx import io

import warnings
warnings.filterwarnings("ignore")

def PTR_init(ocp: cp.Problem, discretization_solver: callable, params: Config, ) -> tuple[cp.Problem, callable]:
    io.intro()

    t_0_while = time.time()

    if params.cvx.cvxpygen:
        from solver.cpg_solver import cpg_solve
        with open('solver/problem.pickle', 'rb') as f:
            prob = pickle.load(f)
    else:
        cpg_solve = None

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(cpg_solve, params.sim.x_bar, params.sim.u_bar, discretization_solver, ocp, params)

    t_f_while = time.time()
    print("Total Initialization Time: ", t_f_while - t_0_while)
    return cpg_solve

def PTR_main(params: Config, prob: cp.Problem, aug_dy: callable, cpg_solve) -> dict:
    J_vb = 1E2
    J_vc = 1E2
    J_tr = 1E2

    x_bar = params.sim.x_bar
    u_bar = params.sim.u_bar

    scp_trajs = [x_bar]
    scp_controls = [u_bar]
    V_multi_shoot_traj = []

    print_queue = queue.Queue()
    threading.Thread(target=io.intermediate, args=(print_queue, params), daemon=True).start()
    io.header()

    k = 1

    if params.dev.profiling:
        import cProfile
        pr = cProfile.Profile()
        
        # Enable the profiler
        pr.enable()

    log_data = []
    # print_queue = []

    t_0_while = time.time()
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

        log_data.append({
                "iter": k,
                "dis_time": dis_time * 1000.0,
                "subprop_time": subprop_time * 1000.0,
                "J_total": J_total,
                "J_tr": J_tr,
                "J_vb": J_vb,
                "J_vc": J_vc,
                "cost": t[-1],
                "prob_stat": prob_stat
            })
        print_queue.put(log_data[-1])

        # if params.dev.debug_printing:
        #     io.intermediate(print_queue, params)
            
        k += 1

    t_f_while = time.time()
    # Disable the profiler
    if params.dev.profiling:
        pr.disable()
        
        # Save results so it can be viusualized with snakeviz
        pr.dump_stats('profiling_results.prof')
    

    # Print logged data once
    # if not params.dev.debug_printing:  
    #     # print(colored("---------------------------------------------------------------------------------------------------------"))
    #     # print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
    #     #     "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))

    #     io.intermediate(print_queue, params)
    time.sleep(0.1)

    io.footer(t_f_while - t_0_while)

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

def PTR_post(params: Config, result: dict, propagation_solver: callable) -> dict:
    t_0_post = time.time()
    x = result["x"]
    u = result["u"]

    t = np.array(s_to_t(u, params))

    t_full = np.arange(0, t[-1], params.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, u, t, params)

    x_full = simulate_nonlinear_time(x[0], u, tau_vals, t, params, propagation_solver)

    print("Total CTCS Constraint Violation:", x_full[-1, params.sim.idx_y])
    i = 0
    cost = np.zeros_like(x[-1, i])
    for type in params.sim.initial_state.type:
        if type == 'Minimize':
            cost += x[0, i]
        i +=1
    i = 0
    for type in params.sim.final_state.type:
        if type == 'Minimize':
            cost += x[-1, i]
        i +=1
    print("Cost: ", cost)

    more_result = dict(
        t_full = t_full,
        x_full = x_full,
        u_full = u_full
    )

    t_f_post = time.time()
    print("Total Post Processing Time: ", t_f_post - t_0_post)
    result.update(more_result)
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
            costs = x[:,i]
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
