import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_animation
from openscvx.trajoptproblem import TrajOptProblem
from openscvx.utils import gen_vertices, rot

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

x = ox.State("x", shape=(14,))  # State variable with 14 dimensions

x.max = np.array([200.0, 100, 200, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100])
x.min = np.array(
    [-200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0]
)  # Lower Bound on the states

x.initial = [
    10.0,
    0,
    20,
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
]
x.final = [
    10.0,
    0,
    20,
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 1),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("minimize", total_time),
]

u = ox.Control("u", shape=(6,))  # Control variable with 6 dimensions

u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])
u.min = np.array([0, 0, 0, -18.665, -18.665, -0.55562])  # Lower Bound on the controls
u.guess = np.repeat(np.expand_dims(np.array([0.0, 0, 10, 0, 0, 0]), axis=0), n, axis=0)


m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone


### Gate Parameters ###
n_gates = 10

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
A_gate = rot @ np.diag(1 / radii) @ rot.T

# Create modified centers (matching original behavior exactly)
modified_centers = []
for center in initial_gate_centers:
    modified_center = center.copy()
    modified_center[0] = modified_center[0] + 2.5
    modified_center[2] = modified_center[2] + 2.5
    modified_centers.append(modified_center)

# Create symbolic parameters matching original structure
A_gate_const = ox.Constant(A_gate)
A_gate_c_params = []
for modified_center in modified_centers:
    A_gate_c_params.append(ox.Constant(A_gate @ modified_center))

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for modified_center in modified_centers:  # Use modified centers for vertices
    vertices.append(gen_vertices(modified_center, radii))
### End Gate Parameters ###


constraints = [
    ox.ctcs(x <= ox.Constant(x.max)),
    ox.ctcs(ox.Constant(x.min) <= x),
]

for node, A_c in zip(gate_nodes, A_gate_c_params):
    gate_constraint = (
        (ox.linalg.Norm(A_gate_const @ x[:3] - A_c, ord="inf") <= ox.Constant(1.0))
        .convex()
        .at([node])
    )
    constraints.append(gate_constraint)


# Define symbolic utility functions
def symbolic_qdcm(q):
    """Quaternion to Direction Cosine Matrix conversion using symbolic expressions"""
    # Normalize quaternion
    q_norm = ox.Sqrt(ox.Sum(q * q))
    q_normalized = q / q_norm

    w, x, y, z = q_normalized[0], q_normalized[1], q_normalized[2], q_normalized[3]

    # Create DCM elements
    r11 = ox.Constant(1.0) - ox.Constant(2.0) * (y * y + z * z)
    r12 = ox.Constant(2.0) * (x * y - z * w)
    r13 = ox.Constant(2.0) * (x * z + y * w)

    r21 = ox.Constant(2.0) * (x * y + z * w)
    r22 = ox.Constant(1.0) - ox.Constant(2.0) * (x * x + z * z)
    r23 = ox.Constant(2.0) * (y * z - x * w)

    r31 = ox.Constant(2.0) * (x * z - y * w)
    r32 = ox.Constant(2.0) * (y * z + x * w)
    r33 = ox.Constant(1.0) - ox.Constant(2.0) * (x * x + y * y)

    # Stack into 3x3 matrix
    row1 = ox.Concat(r11, r12, r13)
    row2 = ox.Concat(r21, r22, r23)
    row3 = ox.Concat(r31, r32, r33)

    return ox.Stack([row1, row2, row3])


def symbolic_ssmp(w):
    """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""
    x, y, z = w[0], w[1], w[2]
    zero = ox.Constant(0.0)

    # Create SSMP matrix
    row1 = ox.Concat(zero, -x, -y, -z)
    row2 = ox.Concat(x, zero, z, -y)
    row3 = ox.Concat(y, -z, zero, x)
    row4 = ox.Concat(z, y, -x, zero)

    return ox.Stack([row1, row2, row3, row4])


def symbolic_ssm(w):
    """Angular rate to 3x3 skew symmetric matrix"""
    x, y, z = w[0], w[1], w[2]
    zero = ox.Constant(0.0)

    # Create SSM matrix
    row1 = ox.Concat(zero, -z, y)
    row2 = ox.Concat(z, zero, -x)
    row3 = ox.Concat(-y, x, zero)

    return ox.Stack([row1, row2, row3])


def symbolic_diag(v):
    """Create diagonal matrix from vector"""
    if len(v) == 3:
        zero = ox.Constant(0.0)
        row1 = ox.Concat(v[0], zero, zero)
        row2 = ox.Concat(zero, v[1], zero)
        row3 = ox.Concat(zero, zero, v[2])
        return ox.Stack([row1, row2, row3])
    else:
        raise NotImplementedError("Only 3x3 diagonal matrices supported")


# Create symbolic dynamics
v = x[3:6]
q = x[6:10]
q_norm = ox.linalg.Norm(q)  # Cleaner than Sqrt(Sum(q * q))
q_normalized = q / q_norm
w = x[10:13]

f = u[:3]
tau = u[3:]

# Option 1: Full symbolic dynamics (more flexible but potentially slower)
# r_dot = v
# v_dot = (Constant(1.0 / m)) * symbolic_qdcm(q) @ f + Constant(
#     np.array([0, 0, g_const], dtype=np.float64)
# )
# q_dot = Constant(0.5) * symbolic_ssmp(w) @ q
# J_b_inv = Constant(1.0 / J_b)
# J_b_diag = symbolic_diag([Constant(J_b[0]), Constant(J_b[1]), Constant(J_b[2])])
# w_dot = symbolic_diag([J_b_inv[0], J_b_inv[1], J_b_inv[2]]) @ (tau - symbolic_ssm(w) @ J_b_diag @ w)
# t_dot = Constant(np.array([1.0], dtype=np.float64))
# dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)

# Option 2: Efficient dynamics using direct JAX lowering (better performance)
r_dot = v
v_dot = (ox.Constant(1.0 / m)) * ox.spatial.QDCM(q_normalized) @ f + ox.Constant(
    np.array([0, 0, g_const], dtype=np.float64)
)
q_dot = ox.Constant(0.5) * ox.spatial.SSMP(w) @ q_normalized
J_b_inv = ox.Constant(1.0 / J_b)
J_b_diag = ox.linalg.Diag(ox.Constant(J_b))
w_dot = ox.linalg.Diag(J_b_inv) @ (tau - ox.spatial.SSM(w) @ J_b_diag @ w)
t_dot = ox.Constant(np.array([1.0], dtype=np.float64))
dyn_expr = ox.Concat(r_dot, v_dot, q_dot, w_dot, t_dot)


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
    dynamics=dyn_expr,
    x=x,
    u=u,
    constraints=constraints,
    idx_time=len(x.max) - 1,
    N=n,
    # licq_max=1E-8
)

problem.settings.prp.dt = 0.01

problem.settings.scp.w_tr = 2e0  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1  # 0e-1,  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = (
    1e1  # 1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
)
problem.settings.scp.ep_tr = 1e-3  # Trust Region Tolerance
problem.settings.scp.ep_vb = 1e-4  # Virtual Control Tolerance
problem.settings.scp.ep_vc = 1e-8  # Virtual Control Tolerance for CTCS
# problem.settings.scp.cost_drop = 10  # SCP iteration to relax minimal final time objective
# problem.settings.scp.cost_relax = 0.8  # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.4  # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight

plotting_dict = {
    "vertices": vertices,
    "gate_centers": modified_centers,
    "A_gate": A_gate_const,
    "A_gate_c_params": A_gate_c_params,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()
