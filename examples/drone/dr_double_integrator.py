import numpy as np
import jax.numpy as jnp
import cvxpy as cp

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem  # noqa: E402
from openscvx.dynamics import dynamics  # noqa: E402
from openscvx.constraints import ctcs, nodal  # noqa: E402
from openscvx.utils import rot, gen_vertices  # noqa: E402
from openscvx.backend.state import State, Free, Minimize  # noqa: E402
from openscvx.backend.control import Control  # noqa: E402

from examples.plotting import plot_animation_double_integrator  # noqa: E402

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

x = State("x", shape=(7,))  # State variable with 14 dimensions

x.max = np.array([200.0, 100, 50, 100, 100, 100, 100])  # Upper Bound on the states
x.min = np.array([-200.0, -100, 15, -100, -100, -100, 0])  # Lower Bound on the states

x.initial = np.array([10.0, 0, 20, 0, 0, 0, 0])
x.final = np.array([10.0, 0, 20, Free(0), Free(0), Free(0), Minimize(total_time)])
x.guess = np.linspace(x.initial, x.final, n)

u = Control("u", shape=(3,))  # Control variable with 6 dimensions
f_max = 4.179446268 * 9.81
u.max = np.array([f_max, f_max, f_max])
u.min = np.array([-f_max, -f_max, -f_max])  # Lower Bound on the controls
initial_control = np.array([0.0, 0, 10,])
u.guess = np.repeat(initial_control[np.newaxis, :], n, axis=0)

m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


### Gate Parameters ###
n_gates = 10
gate_centers = [
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

radii = np.array([2.5, 1e-4, 2.5])
A_gate = rot @ np.diag(1 / radii) @ rot.T
A_gate_cen = []
for center in gate_centers:
    center[0] = center[0] + 2.5
    center[2] = center[2] + 2.5
    A_gate_cen.append(A_gate @ center)
nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
### End Gate Parameters ###


constraints = [
    ctcs(lambda x_, u_: (x_ - x.true.max)),
    ctcs(lambda x_, u_: (x.true.min - x_)),
]
for node, cen in zip(gate_nodes, A_gate_cen):
    constraints.append(
        nodal(
            lambda x_, u_, A=A_gate, c=cen: cp.norm(A @ x_[:3] - c, "inf") <= 1,
            nodes=[node],
            convex=True,
        )
    )  # use local variables inside the lambda function


@dynamics
def dynamics(x_, u_):
    # Unpack the state and control vectors
    v = x_[3:6]

    f = u_[:3]

    # Compute the time derivatives of the state variables
    r_dot = v
    v_dot = (1 / m) * f + jnp.array([0, 0, g_const])
    t_dot = 1
    return jnp.hstack([r_dot, v_dot, t_dot])



x_bar = np.linspace(x.initial, x.final, n)

i = 0
origins = [x.initial[:3]]
ends = []
for center in gate_centers:
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
)

problem.settings.prp.dt = 0.01
problem.settings.dis.custom_integrator = True

problem.settings.scp.w_tr = 2e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1  # 0e-1,  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e1  # 1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
problem.settings.scp.cost_drop = 10  # SCP iteration to relax minimal final time objective
problem.settings.scp.cost_relax = 0.8  # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.4  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight

plotting_dict = dict(vertices=vertices)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation_double_integrator(results, problem.settings).show()