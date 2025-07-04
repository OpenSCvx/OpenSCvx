import os
import sys

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem  # noqa: E402
from openscvx.dynamics import dynamics  # noqa: E402
from openscvx.utils import qdcm, SSMP, SSM, rot, gen_vertices  # noqa: E402
from openscvx.constraints import nodal  # noqa: E402
from openscvx.backend.state import State, Free, Minimize  # noqa: E402
from openscvx.backend.control import Control  # noqa: E402

from examples.plotting import plot_animation  # noqa: E402

n = 33  # Number of Nodes
total_time = 30.0  # Total time for the simulation

x = State("x", shape=(14,))  # State variable with 14 dimensions

x.max = np.array([200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100])  # Upper Bound on the states
x.min = np.array([-200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0])  # Lower Bound on the states
x.initial = np.array([10.0, 0, 20, 0, 0, 0, Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), 0])
x.final = np.array([10.0, 0, 20, Free(0), Free(0), Free(0), Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(total_time)])

u = Control("u", shape=(6,))  # Control variable with 6 dimensions
u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])  # Upper Bound on the controls
u.min = np.array([0, 0, 0, -18.665, -18.665, -0.55562])  # Lower Bound on the controls
u.guess = np.repeat(np.expand_dims(np.array([0., 0., 10., 0., 0., 0.]), axis=0), n, axis=0)


### Sensor Params ###
alpha_x = 4.0  # Angle for the x-axis of Sensor Cone
alpha_y = 4.0  # Angle for the y-axis of Sensor Cone
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)  # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame
norm_type = 2  # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
### End Sensor Params ###


### Gate Parameters ###
n_gates = 10
gate_centers = [
    np.array([59.436, 0.0000, 20.0000]),
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
nodes_per_gate = 3
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
### End Gate Parameters ###

n_subs = 10
init_poses = []
np.random.seed(5)
for i in range(n_subs):
    init_pose = np.array([100.0, -70.0, 20.0])
    init_pose[:2] = init_pose[:2] + np.random.random(2) * 20.0
    init_poses.append(init_pose)

init_poses = init_poses


def g_vp(x_, u_, p_s_I):
    p_s_s = R_sb @ qdcm(x_[6:10]).T @ (p_s_I - x_[0:3])
    return jnp.linalg.norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


def g_cvx_nodal(x_):  # Nodal Convex Inequality Constraints
    constr = []
    for node, cen in zip(gate_nodes, A_gate_cen):
        constr += [cp.norm(A_gate @ x_[node][:3] - cen, "inf") <= 1]
    return constr


constraints = []
for pose in init_poses:
    constraints.append(nodal(lambda x_, u_, p = pose: g_vp(x_, u_, p), convex=False))
for node, cen in zip(gate_nodes, A_gate_cen):
    constraints.append(
        nodal(
            lambda x_, u_, A=A_gate, c=cen: cp.norm(A @ x_[:3] - c, "inf") <= 1,
            nodes=[node], convex=True,
        )
    )  # use local variables inside the lambda function


@dynamics
def dynamics(x_, u_):
    m = 1.0  # Mass of the drone
    g_const = -9.81
    J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone
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

R_sb = R_sb  # Sensor to body frame
b = R_sb @ np.array([0, 1, 0])
for k in range(n):
    kp = []
    for pose in init_poses:
        kp.append(pose)
    kp = np.mean(kp, axis=0)
    a = kp - x_bar[k, :3]
    # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
    q_xyz = np.cross(b, a)
    q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
    q_no_norm = np.hstack((q_w, q_xyz))
    q = q_no_norm / la.norm(q_no_norm)
    x_bar[k, 6:10] = q

x.guess = x_bar

problem = TrajOptProblem(
    dynamics=dynamics,
    x=x,
    u=u,
    constraints=constraints,
    idx_time=len(x.max)-1,
    N=n,
)

problem.settings.prp.dt = 0.1

problem.settings.scp.w_tr = 8e1  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 2e1  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e2  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
problem.settings.scp.lam_vb = 4e0  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance
problem.settings.scp.cost_drop = 10  # SCP iteration to relax minimal final time objective
problem.settings.scp.cost_relax = 0.8  # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.05  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight

plotting_dict = dict(
    vertices=vertices,
    n_subs=n_subs,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    R_sb=R_sb,
    init_poses=init_poses,
    norm_type=norm_type,
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update_plotting_data(**plotting_dict)

    plot_animation(results, problem.settings).show()