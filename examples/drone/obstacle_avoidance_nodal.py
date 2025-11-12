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
from openscvx.utils import generate_orthogonal_unit_vectors

n = 6
total_time = 4.0  # Total time for the simulation

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
position.max = np.array([200.0, 10, 20])
position.min = np.array([-200.0, -100, 0])
position.initial = np.array([10.0, 0, 2])
position.final = [-10.0, 0, 2]

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([100, 100, 100])
velocity.min = np.array([-100, -100, -100])
velocity.initial = np.array([0, 0, 0])
velocity.final = [("free", 0), ("free", 0), ("free", 0)]

attitude = ox.State("attitude", shape=(4,))  # Quaternion [qw, qx, qy, qz]
attitude.max = np.array([1, 1, 1, 1])
attitude.min = np.array([-1, -1, -1, -1])
attitude.initial = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]
attitude.final = [("free", 1.0), ("free", 0), ("free", 0), ("free", 0)]

angular_velocity = ox.State("angular_velocity", shape=(3,))  # Angular velocity [wx, wy, wz]
angular_velocity.max = np.array([10, 10, 10])
angular_velocity.min = np.array([-10, -10, -10])
angular_velocity.initial = [("free", 0), ("free", 0), ("free", 0)]
angular_velocity.final = [("free", 0), ("free", 0), ("free", 0)]

# Define control components
thrust_force = ox.Control("thrust_force", shape=(3,))  # Thrust forces [fx, fy, fz]
thrust_force.max = np.array([0, 0, 4.179446268 * 9.81])
thrust_force.min = np.array([0, 0, 0])
initial_control = np.array([0.0, 0.0, 50.0])
thrust_force.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)

torque = ox.Control("torque", shape=(3,))  # Control torques [tau_x, tau_y, tau_z]
torque.max = np.array([18.665, 18.665, 0.55562])
torque.min = np.array([-18.665, -18.665, -0.55562])
torque.guess = np.zeros((n, 3))


# Define list of all states (needed for TrajOptProblem and constraints)
states = [position, velocity, attitude, angular_velocity]
controls = [thrust_force, torque]


m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone

# Normalize quaternion for dynamics
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Define dynamics as dictionary mapping state names to their derivatives
J_b_inv = 1.0 / J_b
J_b_diag = ox.linalg.Diag(J_b)

dynamics = {
    "position": velocity,
    "velocity": (1.0 / m) * ox.spatial.QDCM(attitude_normalized) @ thrust_force
    + np.array([0, 0, g_const], dtype=np.float64),
    "attitude": 0.5 * ox.spatial.SSMP(angular_velocity) @ attitude_normalized,
    "angular_velocity": ox.linalg.Diag(J_b_inv)
    @ (torque - ox.spatial.SSM(angular_velocity) @ J_b_diag @ angular_velocity),
}


A_obs = []
radius = []
axes = []

# Default values for the obstacle centers
obstacle_center_positions = [
    np.array([-5.1, 0.1, 2]),
    np.array([0.1, 0.1, 2]),
    np.array([5.1, 0.1, 2]),
]

# Define obstacle centers as parameters for runtime updates
obstacle_centers = [
    ox.Parameter("obstacle_center_1", shape=(3,), value=obstacle_center_positions[0]),
    ox.Parameter("obstacle_center_2", shape=(3,), value=obstacle_center_positions[1]),
    ox.Parameter("obstacle_center_3", shape=(3,), value=obstacle_center_positions[2]),
]

np.random.seed(0)
for _ in obstacle_center_positions:
    ax = generate_orthogonal_unit_vectors()
    axes.append(generate_orthogonal_unit_vectors())
    rad = np.random.rand(3) + 0.1 * np.ones(3)
    radius.append(rad)
    A_obs.append(ax @ np.diag(rad**2) @ ax.T)

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Add obstacle constraints using symbolic expressions (as nodal constraints)
for center, A in zip(obstacle_centers, A_obs):
    A_const = A

    # Obstacle constraint: (pos - center)^T @ A @ (pos - center) >= 1
    diff = position - center
    obstacle_constraint = 1.0 <= diff.T @ A_const @ diff
    constraints.append(obstacle_constraint)

# Set initial guesses
position.guess = np.linspace(position.initial, position.final, n)
velocity.guess = np.linspace(velocity.initial, [0, 0, 0], n)
attitude.guess = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
angular_velocity.guess = np.zeros((n, 3))

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
)

problem.settings.prp.dt = 0.01
problem.settings.scp.lam_vb = 1e0
problem.settings.scp.cost_drop = 4  # SCP iteration to relax minimal final time objective
problem.settings.scp.cost_relax = 0.5  # Minimal Time Relaxation Factor

plotting_dict = {
    "obstacles_centers": obstacle_center_positions,
    "obstacles_axes": axes,
    "obstacles_radii": radius,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()
