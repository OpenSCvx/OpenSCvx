import numpy as np
import numpy.linalg as la
import jax.numpy as jnp

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.utils import qdcm, SSMP, SSM, get_kp_pose
from openscvx.constraints import boundary, ctcs
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control

from examples.plotting import plot_animation

n = 12  # Number of Nodes
total_time = 40.0  # Total time for the simulation

fuel_inds = 13  # Fuel Index in State
t_inds = 14
s_inds = 6  # Time dilation index in Control

x = State("x", shape=(15,))  # State variable with 15 dimensions
x.max = np.array([200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 2000, 40])  # Upper Bound on the states
x.min = np.array([-100.0, -100, -10, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0])  # Lower Bound on the states

x.initial = np.array([8.0, -0.2, 2.2, 0, 0, 0, Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), 0, 0])
x.final = np.array([Free(-10.0), Free(0), Free(2), Free(0), Free(0), Free(0), Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(0), 40])

u = Control("u", shape=(6,))  # Control variable with 6 dimensions

u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])
u.min = np.array([0, 0, 0, -18.665, -18.665, -0.55562])
initial_control = np.array([0, 0, 10, 0, 0, 0])
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)

init_pose = np.array([13.0, 0.0, 2.0])
min_range = 4.0
max_range = 16.0

### View Planning Params ###
n_subs = 1  # Number of Subjects
alpha_x = 6.0  # Angle for the x-axis of Sensor Cone
alpha_y = 8.0  # Angle for the y-axis of Sensor Cone
A_cone = np.diag(
    [
        1 / np.tan(np.pi / alpha_x),
        1 / np.tan(np.pi / alpha_y),
        0,
    ]
)  # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame
norm_type = np.inf  # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


@dynamics
def dynamics(x_, u_):
    m = 1.0  # Mass of the drone
    g_const = -9.18
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
    fuel_dot = jnp.linalg.norm(u_)[None]
    t_dot = 1
    return jnp.hstack([r_dot, v_dot, q_dot, w_dot, fuel_dot, t_dot])


def g_vp(x_):
    p_s_I = get_kp_pose(x_[t_inds], init_pose)
    p_s_s = R_sb @ qdcm(x_[6:10]).T @ (p_s_I - x_[:3])
    return jnp.linalg.norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)


def g_min(x_):
    p_s_I = get_kp_pose(x_[t_inds], init_pose)
    return min_range - jnp.linalg.norm(p_s_I - x_[:3])


def g_max(x_):
    p_s_I = get_kp_pose(x_[t_inds], init_pose)
    return jnp.linalg.norm(p_s_I - x_[:3]) - max_range


constraints = [
    ctcs(lambda x_, u_: np.sqrt(2e1) * g_vp(x_)),
    ctcs(lambda x_, u_: x_ - x.true_state.max),
    ctcs(lambda x_, u_: x.true_state.min - x_),
    ctcs(lambda x_, u_: g_min(x_)),
    ctcs(lambda x_, u_: g_max(x_)),
]


x_bar = np.linspace(x.initial, x.final, n)

x_bar[:, :3] = get_kp_pose(x_bar[:, t_inds], init_pose) + jnp.array([-5, 0.2, 0.2])[None, :]

b = R_sb @ np.array([0, 1, 0])
for k in range(n):
    kp = get_kp_pose(x_bar[k, t_inds], init_pose)
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
    idx_time=t_inds,
    N=n,
    licq_max=1e-8,
)

problem.settings.prp.dt = 0.1

problem.settings.scp.w_tr = 4e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-2  # Weight on the Minimal Fuel Objective
problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective

problem.settings.scp.ep_tr = 1e-6  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
problem.settings.scp.w_tr_adapt = 1.3  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e3  # Maximum Trust Region Weight

plotting_dict = dict(
    n_subs=n_subs,
    alpha_x=alpha_x,
    alpha_y=alpha_y,
    R_sb=R_sb,
    init_poses=init_pose,
    norm_type=norm_type,
    min_range=min_range,
    max_range=max_range,
    moving_subject=True,
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()