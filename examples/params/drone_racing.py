import numpy as np
import jax.numpy as jnp
import cvxpy as cp

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.utils import qdcm, SSMP, SSM, rot, gen_vertices
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.control import Control

from examples.plotting import plot_animation

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

x = State("x", shape=(14,))  # State variable with 14 dimensions

x.max = np.array([ 200.0,  100, 200,  100,  100,  100,  1,  1,  1,  1,  10,  10,  10, 100]) 
x.min = np.array([-200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10,   0])  # Lower Bound on the states

x.initial = np.array([10.0, 0, 20, 0, 0, 0, Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), 0])
x.final   = np.array([10.0, 0, 20, Free(0), Free(0), Free(0), Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(total_time)])

u = Control("u", shape=(6,))  # Control variable with 6 dimensions

u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])
u.min = np.array([0, 0, 0, -18.665, -18.665, -0.55562])  # Lower Bound on the controls
u.guess = np.repeat(np.expand_dims(np.array([0.0, 0, 10, 0, 0, 0]), axis=0), n, axis=0)


m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


### Gate Parameters ###
n_gates = 10

# Create cvxpy.Parameters for gate centers, A_gate, and A_gate_c (A @ center)
gate_center_params = []
A_gate_c_params = []
for i in range(n_gates):
    gate_center_params.append(cp.Parameter(3, name=f"gate_center_{i}"))
    A_gate_c_params.append(cp.Parameter(3, name=f"A_gate_c_{i}"))

# Initialize gate centers
initial_gate_centers = [
    np.array([59.436, 0.000, 20.0000]),
    np.array([92.964, -23.750, 25.5240]),
    np.array([92.964, -29.274, 20.0000]),
    np.array([92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([92.964, -75.080, 20.0000]),
    np.array([92.964, -68.556, 20.0000]),
    np.array([59.436, -81.358, 20.0000]),
    np.array([22.250, -42.672, 20.0000]),
]

# Set initial values for gate center parameters and A_gate_c_params
radii = np.array([2.5, 1e-4, 2.5])
A_gate_param = cp.Parameter((3, 3), name="A_gate")
A_gate_param.value = rot @ np.diag(1 / radii) @ rot.T

# Create modified centers (matching original behavior exactly)
modified_centers = []
for center in initial_gate_centers:
    modified_center = center.copy()
    modified_center[0] = modified_center[0] + 2.5
    modified_center[2] = modified_center[2] + 2.5
    modified_centers.append(modified_center)

for i, modified_center in enumerate(modified_centers):
    gate_center_params[i].value = modified_center  # Use modified centers for parameters
    A_gate_c_params[i].value = A_gate_param.value @ modified_center

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for modified_center in modified_centers:  # Use modified centers for vertices
    vertices.append(gen_vertices(modified_center, radii))
### End Gate Parameters ###


constraints = [
    ctcs(lambda x_, u_: (x_ - x.true.max)),
    ctcs(lambda x_, u_: (x.true.min - x_)),
]

for node, A_c in zip(gate_nodes, A_gate_c_params):
    constraints.append(
        nodal(
            lambda x_, u_, A=A_gate_param, Ac=A_c: cp.norm(A @ x_[:3] - Ac, "inf") <= 1,
            nodes=[node],
            convex=True,
        )
    )


@dynamics
def dynamics(x_, u_):
    # Unpack the state and control vectors
    v = x_[3:6]
    q = x_[6:10]
    w = x_[10:13]

    f = u_[:3]
    tau = u_[3:]

    q_norm = jnp.linalg.norm(q)
    q = q / q_norm

    # Compute the time derivatives of the state variables
    r_dot = v
    v_dot = (1 / m) * qdcm(q) @ f + jnp.array([0, 0, g_const])
    q_dot = 0.5 * SSMP(w) @ q
    w_dot = jnp.diag(1 / J_b) @ (tau - SSM(w) @ jnp.diag(J_b) @ w)
    t_dot = 1
    return jnp.hstack([r_dot, v_dot, q_dot, w_dot, t_dot])


x_bar = np.linspace(x.initial, x.final, n)

i = 0
origins = [x.initial[:3]]
ends = []
for center in modified_centers:  # Use modified centers for initial guess
    origins.append(center)
    ends.append(center)
ends.append(x.final[:3])
gate_idx = 0
for _ in range(n_gates + 1):
    for k in range(n // (n_gates + 1)):
        x_bar[i, :3] = origins[gate_idx] + (k / (n // (n_gates + 1))) * (
            ends[gate_idx] - origins[gate_idx]
        )
        i += 1
    gate_idx += 1

x.guess = x_bar

problem = TrajOptProblem(
    dynamics=dynamics,
    x=x,
    u=u,
    constraints=constraints,
    idx_time=len(x.max)-1,
    N=n,
    # licq_max=1E-8
)

problem.settings.prp.dt = 0.01

problem.settings.scp.w_tr = 2e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1  # 0e-1,  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e1  # 1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
# problem.settings.scp.cost_drop = 10  # SCP iteration to relax minimal final time objective
# problem.settings.scp.cost_relax = 0.8  # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.4  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight

plotting_dict = dict(vertices=vertices, gate_center_params=gate_center_params, A_gate_param=A_gate_param, A_gate_c_params=A_gate_c_params)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()