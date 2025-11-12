import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_animation
from openscvx import TrajOptProblem
from openscvx.utils import gen_vertices, rot

n = 22  # Number of Nodes
total_time = 24.0  # Total time for the simulation

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 100, 200])
position.min = np.array([-200.0, -100, 15])
position.initial = np.array([10.0, 0, 20])
position.final = [10.0, 0, 20]
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]
velocity.guess = np.linspace(velocity.initial, [0, 0, 0], n)

attitude = ox.State("attitude", shape=(4,))  # Quaternion [qw, qx, qy, qz]
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1), ("free", 0), ("free", 0), ("free", 0)]
attitude.guess = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))

angular_velocity = ox.State("angular_velocity", shape=(3,))  # Angular velocity [wx, wy, wz]
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.guess = np.zeros((n, 3))

# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
thrust_force.guess = np.repeat(np.array([[0.0, 0, 10]]), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))


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

# Create symbolic parameters for each gate center with initial values
A_gate_const = A_gate
gate_center_params = []
for i, modified_center in enumerate(modified_centers):
    # Create a Parameter with initial value
    param = ox.Parameter(f"gate_{i}_center", shape=(3,), value=modified_center)
    gate_center_params.append(param)

nodes_per_gate = 2
gate_nodes = np.arange(nodes_per_gate, n, nodes_per_gate)
vertices = []
for modified_center in modified_centers:  # Use modified centers for vertices
    vertices.append(gen_vertices(modified_center, radii))
### End Gate Parameters ###


# Define list of all states (needed for TrajOptProblem and constraints)
states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add gate constraints
for node, gate_center_param in zip(gate_nodes, gate_center_params):
    # Symbolically compute A_gate @ position - A_gate @ gate_center
    gate_constraint = (
        (
            ox.linalg.Norm(A_gate_const @ position - A_gate_const @ gate_center_param, ord="inf")
            <= 1.0
        )
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
    r11 = 1.0 - 2.0 * (y * y + z * z)
    r12 = 2.0 * (x * y - z * w)
    r13 = 2.0 * (x * z + y * w)

    r21 = 2.0 * (x * y + z * w)
    r22 = 1.0 - 2.0 * (x * x + z * z)
    r23 = 2.0 * (y * z - x * w)

    r31 = 2.0 * (x * z - y * w)
    r32 = 2.0 * (y * z + x * w)
    r33 = 1.0 - 2.0 * (x * x + y * y)

    # Stack into 3x3 matrix
    row1 = ox.Concat(r11, r12, r13)
    row2 = ox.Concat(r21, r22, r23)
    row3 = ox.Concat(r31, r32, r33)

    return ox.Stack([row1, row2, row3])


def symbolic_ssmp(w):
    """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""
    x, y, z = w[0], w[1], w[2]
    zero = 0.0

    # Create SSMP matrix
    row1 = ox.Concat(zero, -x, -y, -z)
    row2 = ox.Concat(x, zero, z, -y)
    row3 = ox.Concat(y, -z, zero, x)
    row4 = ox.Concat(z, y, -x, zero)

    return ox.Stack([row1, row2, row3, row4])


def symbolic_ssm(w):
    """Angular rate to 3x3 skew symmetric matrix"""
    x, y, z = w[0], w[1], w[2]
    zero = 0.0

    # Create SSM matrix
    row1 = ox.Concat(zero, -z, y)
    row2 = ox.Concat(z, zero, -x)
    row3 = ox.Concat(-y, x, zero)

    return ox.Stack([row1, row2, row3])


def symbolic_diag(v):
    """Create diagonal matrix from vector"""
    if len(v) == 3:
        zero = 0.0
        row1 = ox.Concat(v[0], zero, zero)
        row2 = ox.Concat(zero, v[1], zero)
        row3 = ox.Concat(zero, zero, v[2])
        return ox.Stack([row1, row2, row3])
    else:
        raise NotImplementedError("Only 3x3 diagonal matrices supported")


# Create symbolic dynamics
# Normalize quaternion for dynamics
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Option 1: Full symbolic dynamics (more flexible but potentially slower)
# r_dot = velocity
# v_dot = (Constant(1.0 / m)) * symbolic_qdcm(attitude) @ thrust_force + Constant(
#     np.array([0, 0, g_const], dtype=np.float64)
# )
# q_dot = Constant(0.5) * symbolic_ssmp(angular_velocity) @ attitude
# J_b_inv = Constant(1.0 / J_b)
# J_b_diag = symbolic_diag([Constant(J_b[0]), Constant(J_b[1]), Constant(J_b[2])])
# w_dot = symbolic_diag([J_b_inv[0], J_b_inv[1], J_b_inv[2]]) @ (
#     torque - symbolic_ssm(angular_velocity) @ J_b_diag @ angular_velocity
# )

# Option 2: Efficient dynamics using direct JAX lowering (better performance)
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + ox.Constant(np.array([0, 0, g_const], dtype=np.float64)),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
}


# Generate initial guess for position trajectory through gates
position_bar = np.linspace(position.initial, position.final, n)

i = 0
origins = [position.initial]
ends = []
for center in modified_centers:  # Use modified centers for initial guess
    origins.append(center)
    ends.append(center)
ends.append(position.final)
gate_idx = 0
for _ in range(n_gates + 1):
    for k in range(n // (n_gates + 1)):
        position_bar[i] = origins[gate_idx] + (k / (n // (n_gates + 1))) * (
            ends[gate_idx] - origins[gate_idx]
        )
        i += 1
    gate_idx += 1

position.guess = position_bar

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = TrajOptProblem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
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
    "A_gate_c_params": [A_gate @ center for center in modified_centers],
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()
