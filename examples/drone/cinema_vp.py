import os
import sys

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_animation
from openscvx import TrajOptProblem
from openscvx.utils import get_kp_pose

n = 12  # Number of Nodes
total_time = 40.0  # Total time for the simulation

fuel_inds = 13  # Fuel Index in State
t_inds = 14
s_inds = 6  # Time dilation index in Control

x = ox.State("x", shape=(15,))  # State variable with 15 dimensions
x.max = np.array(
    [200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 2000, 40]
)  # Upper Bound on the states
x.min = np.array(
    [-100.0, -100, -10, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0]
)  # Lower Bound on the states

x.initial = [
    8.0,
    -0.2,
    2.2,
    0,
    0,
    0,
    ("free", 1.0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    0,
    0,
]
x.final = [
    ("free", -10.0),
    ("free", 0),
    ("free", 2),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 1.0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("minimize", 0),
    40,
]

u = ox.Control("u", shape=(6,))  # Control variable with 6 dimensions

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
norm_type = "inf"
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


# Create symbolic dynamics
m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone

# Unpack the state and control vectors using symbolic expressions
v = x[3:6]
q = x[6:10]
q_norm = ox.linalg.Norm(q)
q_normalized = q / q_norm
w = x[10:13]

f = u[:3]
tau = u[3:]

# Define dynamics using symbolic expressions
r_dot = v
v_dot = (ox.Constant(1.0 / m)) * ox.spatial.QDCM(q_normalized) @ f + ox.Constant(
    np.array([0, 0, g_const], dtype=np.float64)
)
q_dot = ox.Constant(0.5) * ox.spatial.SSMP(w) @ q_normalized
J_b_inv = ox.Constant(1.0 / J_b)
J_b_diag = ox.linalg.Diag(ox.Constant(J_b))
w_dot = ox.linalg.Diag(J_b_inv) @ (tau - ox.spatial.SSM(w) @ J_b_diag @ w)
fuel_dot = ox.linalg.Norm(u)
t_dot = ox.Constant(np.array([1.0], dtype=np.float64))
dynamics = ox.Concat(r_dot, v_dot, q_dot, w_dot, fuel_dot, t_dot)


# Symbolic implementation of get_kp_pose function
def get_kp_pose_symbolic(t_expr, init_pose):
    loop_time = 40.0
    loop_radius = 20.0

    # Convert the trajectory parameters to symbolic constants
    loop_time_const = ox.Constant(loop_time)
    loop_radius_const = ox.Constant(loop_radius)
    two_pi_const = ox.Constant(2 * np.pi)
    init_pose_const = ox.Constant(init_pose)
    half_const = ox.Constant(0.5)

    # Compute symbolic trajectory: t_angle = t / loop_time * (2 * pi)
    t_angle = t_expr / loop_time_const * two_pi_const

    # x = loop_radius * sin(t_angle)
    x_pos = loop_radius_const * ox.Sin(t_angle)

    # y = x * cos(t_angle)
    y_pos = x_pos * ox.Cos(t_angle)

    # z = 0.5 * x * sin(t_angle)
    z_pos = half_const * x_pos * ox.Sin(t_angle)

    # Stack into position vector and add initial pose
    kp_trajectory = ox.Concat(x_pos, y_pos, z_pos) + init_pose_const
    return kp_trajectory


# Create symbolic constraints
constraints = [
    ox.ctcs(x <= ox.Constant(x.max)),
    ox.ctcs(ox.Constant(x.min) <= x),
]

# Get the symbolic keypoint pose based on time
kp_pose_symbolic = get_kp_pose_symbolic(x[t_inds], init_pose)

# View planning constraint using symbolic keypoint pose
R_sb_const = ox.Constant(R_sb)
A_cone_const = ox.Constant(A_cone)
c_const = ox.Constant(c)

p_s_s = R_sb_const @ ox.spatial.QDCM(x[6:10]).T @ (kp_pose_symbolic - x[:3])
vp_constraint = ox.Constant(np.sqrt(2e1)) * (
    ox.linalg.Norm(A_cone_const @ p_s_s, ord=norm_type) - (c_const.T @ p_s_s)
)

# Range constraints using symbolic keypoint pose
min_range_constraint = ox.Constant(min_range) - ox.linalg.Norm(kp_pose_symbolic - x[:3])
max_range_constraint = ox.linalg.Norm(kp_pose_symbolic - x[:3]) - ox.Constant(max_range)

constraints.extend(
    [
        ox.ctcs(vp_constraint <= ox.Constant(0.0)),
        ox.ctcs(min_range_constraint <= ox.Constant(0.0)),
        ox.ctcs(max_range_constraint <= ox.Constant(0.0)),
    ]
)


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


problem.settings.scp.w_tr = 4e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-2  # Weight on the Minimal Fuel Objective
problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective

problem.settings.scp.ep_tr = 1e-6  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
problem.settings.scp.w_tr_adapt = 1.3  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e3  # Maximum Trust Region Weight

plotting_dict = {
    "n_subs": n_subs,
    "alpha_x": alpha_x,
    "alpha_y": alpha_y,
    "R_sb": R_sb,
    "init_poses": init_pose,
    "norm_type": norm_type,
    "min_range": min_range,
    "max_range": max_range,
    "moving_subject": True,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()
