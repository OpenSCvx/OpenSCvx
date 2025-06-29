import os
import numpy.linalg as la
from numpy import block
import numpy as np
import cvxpy as cp
from openscvx.config import Config

# Optional cvxpygen import
try:
    from cvxpygen import cpg
    CVXPYGEN_AVAILABLE = True
except ImportError:
    CVXPYGEN_AVAILABLE = False
    cpg = None


def OptimalControlProblem(settings: Config):
    ########################
    # VARIABLES & PARAMETERS
    ########################

    # Parameters
    w_tr = cp.Parameter(nonneg = True, name='w_tr')
    lam_cost = cp.Parameter(nonneg=True, name='lam_cost')

    # State
    x = cp.Variable((settings.scp.n, settings.sim.n_states), name='x') # Current State
    dx = cp.Variable((settings.scp.n, settings.sim.n_states), name='dx') # State Error
    x_bar = cp.Parameter((settings.scp.n, settings.sim.n_states), name='x_bar') # Previous SCP State
    x_init = cp.Parameter(settings.sim.n_states, name='x_init') # Initial State
    x_term = cp.Parameter(settings.sim.n_states, name='x_term') # Final State

    # Affine Scaling for State
    S_x = settings.sim.S_x
    inv_S_x = settings.sim.inv_S_x
    c_x = settings.sim.c_x

    # Control
    u = cp.Variable((settings.scp.n, settings.sim.n_controls), name='u') # Current Control
    du = cp.Variable((settings.scp.n, settings.sim.n_controls), name='du') # Control Error
    u_bar = cp.Parameter((settings.scp.n, settings.sim.n_controls), name='u_bar') # Previous SCP Control

    # Affine Scaling for Control
    S_u = settings.sim.S_u
    inv_S_u = settings.sim.inv_S_u
    c_u = settings.sim.c_u

    # Discretized Augmented Dynamics Constraints
    A_d = cp.Parameter((settings.scp.n - 1, (settings.sim.n_states)*(settings.sim.n_states)), name='A_d')
    B_d = cp.Parameter((settings.scp.n - 1, settings.sim.n_states*settings.sim.n_controls), name='B_d')
    C_d = cp.Parameter((settings.scp.n - 1, settings.sim.n_states*settings.sim.n_controls), name='C_d')
    z_d = cp.Parameter((settings.scp.n - 1, settings.sim.n_states), name='z_d')
    nu  = cp.Variable((settings.scp.n - 1, settings.sim.n_states), name='nu') # Virtual Control

    # Linearized Nonconvex Nodal Constraints
    if settings.sim.constraints_nodal:
        g = []
        grad_g_x = []
        grad_g_u = []
        nu_vb = []
        for idx_ncvx, constraint in enumerate(settings.sim.constraints_nodal):
            if not constraint.convex:
                g.append(cp.Parameter(settings.scp.n, name = 'g_' + str(idx_ncvx)))
                grad_g_x.append(cp.Parameter((settings.scp.n, settings.sim.n_states), name='grad_g_x_' + str(idx_ncvx)))
                grad_g_u.append(cp.Parameter((settings.scp.n, settings.sim.n_controls), name='grad_g_u_' + str(idx_ncvx)))
                nu_vb.append(cp.Variable(settings.scp.n, name='nu_vb_' + str(idx_ncvx))) # Virtual Control for VB

    # Applying the affine scaling to state and control
    x_nonscaled = []
    u_nonscaled = []
    for k in range(settings.scp.n):
        x_nonscaled.append(S_x @ x[k] + c_x)
        u_nonscaled.append(S_u @ u[k] + c_u)

    constr = []
    cost = lam_cost * 0

    #############
    # CONSTRAINTS
    #############
    idx_ncvx = 0
    if settings.sim.constraints_nodal:
        for constraint in settings.sim.constraints_nodal:
            if constraint.nodes is None:
                nodes = range(settings.scp.n)
            else:
                nodes = constraint.nodes

            if constraint.convex and constraint.vectorized:
                constr += [constraint(x_nonscaled, u_nonscaled)]
                
            elif constraint.convex:
                constr += [constraint(x_nonscaled[node], u_nonscaled[node]) for node in nodes]

            elif not constraint.convex:
                constr += [((g[idx_ncvx][node] + grad_g_x[idx_ncvx][node] @ dx[node] + grad_g_u[idx_ncvx][node] @ du[node])) == nu_vb[idx_ncvx][node] for node in nodes]
                idx_ncvx += 1

    for i in range(settings.sim.idx_x_true.start, settings.sim.idx_x_true.stop):
        if settings.sim.x.initial_type[i] == 'Fix':
            constr += [x_nonscaled[0][i] == x_init[i]]  # Initial Boundary Conditions
        if settings.sim.x.final_type[i] == 'Fix':
            constr += [x_nonscaled[-1][i] == x_term[i]]   # Final Boundary Conditions
        if settings.sim.x.initial_type[i] == 'Minimize':
            cost += lam_cost * x_nonscaled[0][i]
        if settings.sim.x.final_type[i] == 'Minimize':
            cost += lam_cost * x_nonscaled[-1][i]
        if settings.sim.x.initial_type[i] == 'Maximize':
            cost -= lam_cost * x_nonscaled[0][i]
        if settings.sim.x.final_type[i] == 'Maximize':
            cost -= lam_cost * x_nonscaled[-1][i]

    if settings.scp.uniform_time_grid:
        constr += [u_nonscaled[i][settings.sim.idx_s] == u_nonscaled[i-1][settings.sim.idx_s] for i in range(1, settings.scp.n)]

    constr += [0 == la.inv(S_x) @ (x_nonscaled[i] - x_bar[i] - dx[i]) for i in range(settings.scp.n)] # State Error
    constr += [0 == la.inv(S_u) @ (u_nonscaled[i] - u_bar[i] - du[i]) for i in range(settings.scp.n)] # Control Error

    constr += [x_nonscaled[i] == \
                      cp.reshape(A_d[i-1], (settings.sim.n_states, settings.sim.n_states)) @ x_nonscaled[i-1] \
                    + cp.reshape(B_d[i-1], (settings.sim.n_states, settings.sim.n_controls)) @ u_nonscaled[i-1] \
                    + cp.reshape(C_d[i-1], (settings.sim.n_states, settings.sim.n_controls)) @ u_nonscaled[i] \
                    + z_d[i-1] \
                    + nu[i-1] for i in range(1, settings.scp.n)] # Dynamics Constraint
    
    constr += [u_nonscaled[i] <= settings.sim.u.max for i in range(settings.scp.n)]
    constr += [u_nonscaled[i] >= settings.sim.u.min for i in range(settings.scp.n)] # Control Constraints

    constr += [x_nonscaled[i][settings.sim.idx_x_true] <= settings.sim.x.true.max for i in range(settings.scp.n)]
    constr += [x_nonscaled[i][settings.sim.idx_x_true] >= settings.sim.x.true.min for i in range(settings.scp.n)] # State Constraints (Also implemented in CTCS but included for numerical stability)

    ########
    # COSTS
    ########
    
    inv = block([[inv_S_x, np.zeros((S_x.shape[0], S_u.shape[1]))], [np.zeros((S_u.shape[0], S_x.shape[1])), inv_S_u]])
    cost += sum(w_tr * cp.sum_squares(inv @ cp.hstack((dx[i], du[i]))) for i in range(settings.scp.n))  # Trust Region Cost
    cost += sum(settings.scp.lam_vc * cp.sum(cp.abs(nu[i-1])) for i in range(1, settings.scp.n)) # Virtual Control Slack
    
    idx_ncvx = 0
    if settings.sim.constraints_nodal:
        for constraint in settings.sim.constraints_nodal:
            if not constraint.convex:
                cost += settings.scp.lam_vb * cp.sum(cp.pos(nu_vb[idx_ncvx]))
                idx_ncvx += 1

    for idx, nodes in zip(np.arange(settings.sim.idx_y.start, settings.sim.idx_y.stop), settings.sim.ctcs_node_intervals):  
        if nodes[0] == 0:
            start_idx = 1
        else:
            start_idx = nodes[0]
        constr += [cp.abs(x_nonscaled[i][idx] - x_nonscaled[i-1][idx]) <= settings.sim.x.max[idx] for i in range(start_idx, nodes[1])]
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
        if not os.path.exists('solver'):
            cpg.generate_code(prob, solver = settings.cvx.solver, code_dir='solver', wrapper = True)
        else:
            # Prompt the use to indicate if they wish to overwrite the solver directory or use the existing compiled solver
            if settings.cvx.cvxpygen_override:
                cpg.generate_code(prob, solver = settings.cvx.solver, code_dir='solver', wrapper = True)
            else:
                overwrite = input("Solver directory already exists. Overwrite? (y/n): ")
                if overwrite.lower() == 'y':
                    cpg.generate_code(prob, solver = settings.cvx.solver, code_dir='solver', wrapper = True)
                else:
                    pass
    return prob