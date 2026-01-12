"""6-DOF Powered Descent Guidance (PDG) for planetary landing.

This example demonstrates optimal trajectory generation for a 6-DOF rocket performing
powered descent guidance with full attitude dynamics. The problem includes:

- 6-DOF dynamics (position, velocity, quaternion attitude, angular velocity)
- Fuel-optimal mass minimization
- Thrust magnitude and pointing constraints
- Glideslope constraint for safe landing approach
- Attitude constraints (tilt angle, angular velocity limits)
- Aerodynamic forces
"""

import os
import sys

import numpy as np
import jax.numpy as jnp

# Add grandparent directory to path to import examples.plotting
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from openscvx import Problem

n = 10  # Number of discretization nodes

# ====================#
# *———* Parameters *———*#
# ====================#
gI = ox.Parameter("gI", value=1.0)
l = ox.Parameter("l", value=0.25)
J1 = ox.Parameter("J1", value=0.168 * 2e-2)
J2 = ox.Parameter("J2", value=0.168)
J3 = ox.Parameter("J3", value=0.168)
g0 = ox.Parameter("g0", value=1.0)
Isp = ox.Parameter("Isp", value=30.0)
m_dry = ox.Parameter("m_dry", value=1.0)
v_max = ox.Parameter("v_max", value=3.0)
w_max = ox.Parameter("w_max", value=0.3752)
del_max = ox.Parameter("del_max", value=20.0)
theta_max = ox.Parameter("theta_max", value=75.0)
T_min = ox.Parameter("T_min", value=1.5)
T_max = ox.Parameter("T_max", value=6.5)
gamma = ox.Parameter("gamma", value=75.0)
beta = ox.Parameter("beta", value=0.01)
c_ax = ox.Parameter("c_ax", value=0.5)
c_ayz = ox.Parameter("c_ayz", value=1.0)
S_a = ox.Parameter("S_a", value=0.5)
rho = ox.Parameter("rho", value=1.0)
l_p = ox.Parameter("l_p", value=0.05)

# ==================#
# *———* States *———*#
# ==================#
mass = ox.State("mass", shape=(1,))
mass.max = np.array([2.0])
mass.min = np.array([1.0])
mass.initial = np.array([2.0])
mass.final = [("maximize", 1.0)]
mass.guess = np.linspace(mass.initial, 1.0, n).reshape(-1, 1)

position = ox.State("position", shape=(3,))
position.max = np.array([10.0, 10.0, 10.0])
position.min = np.array([-10.0, -10.0, -10.0])
position.initial = np.array([7.5, 4.5, 2.0])
position.final = np.array([0.0, 0.0, 0.0])
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(3,))
velocity.max = np.array([1.0, 1.0, 1.0])
velocity.min = np.array([-1.0, -3.0, -1.0])  # Allow -2.8 for initial y-velocity
velocity.initial = np.array([-0.5, -2.8, 0.0])
velocity.final = np.array([-0.1, 0.0, 0.0])
velocity.guess = np.linspace(velocity.initial, velocity.final, n)

# Quaternion attitude [q1, q2, q3, q4] = [x, y, z, w] where q4 is scalar part (scalar last convention)
# This matches the original scvxgen code convention
attitude = ox.State("attitude", shape=(4,))
attitude.max = np.array([1.0, 1.0, 1.0, 1.0])
attitude.min = np.array([-1.0, -1.0, -1.0, -1.0])
attitude.initial = [("free", 0.0), ("free", 0.0), ("free", 0.0), ("free", 1.0)]
attitude.final = np.array([0.0, 0.0, 0.0, 1.0])
attitude.guess = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))

angular_velocity = ox.State("angular_velocity", shape=(3,))
# Use parameter value directly for bounds (will be evaluated at runtime)
w_max_val = w_max.value
angular_velocity.max = np.array([w_max_val, w_max_val, w_max_val])
angular_velocity.min = np.array([-w_max_val, -w_max_val, -w_max_val])
angular_velocity.initial = np.array([0.0, 0.0, 0.0])
angular_velocity.final = np.array([0.0, 0.0, 0.0])
angular_velocity.guess = np.zeros((n, 3))

# ====================#
# *———* Controls *———*#
# ====================#
thrust = ox.Control("thrust", shape=(3,))
# Use parameter values directly for bounds
T_max_val = T_max.value
gI_val = gI.value
m_dry_val = m_dry.value
thrust.max = np.array([T_max_val, T_max_val, T_max_val])
thrust.min = np.array([-T_max_val, -T_max_val, -T_max_val])
thrust.initial = np.array([gI_val * mass.initial[0], 0.0, 0.0])
thrust.final = np.array([gI_val * m_dry_val, 0.0, 0.0])
# Set initial guess for control
thrust_initial_guess = np.linspace(thrust.initial, thrust.final, n)
thrust.guess = thrust_initial_guess
# Note: Control rate limits can be added as cross-node constraints if needed,
# e.g., using (thrust.at(k) - thrust.at(k-1)) <= max_rate

# ====================#
# *———* Dynamics *———*#
# ====================#
# Normalize quaternion for dynamics
# Original format: [q1, q2, q3, q4] = [x, y, z, w] (scalar last)
q_norm = ox.linalg.Norm(attitude)
attitude_normalized = attitude / q_norm

# Convert from [x, y, z, w] to [w, x, y, z] for OpenSCvx spatial operations
# OpenSCvx QDCM and SSMP expect [w, x, y, z] format (scalar first)
q_openscvx = ox.Concat(attitude_normalized[3], attitude_normalized[0], attitude_normalized[1], attitude_normalized[2])

# Direction cosine matrix (DCM) from quaternion
# QDCM expects [w, x, y, z] format
CBI = ox.spatial.QDCM(q_openscvx)  # Body to Inertial DCM

# Inertia tensor (diagonal)
J_diag = ox.linalg.Diag(ox.Concat(J1, J2, J3))
J_inv = ox.linalg.Diag(ox.Concat(1.0 / J1, 1.0 / J2, 1.0 / J3))

# Aerodynamic coefficient matrix
CA = ox.linalg.Diag(ox.Concat(c_ax, c_ayz, c_ayz))

# Velocity in body frame (transform from inertial to body)
# CBI is body-to-inertial, so CBI.T is inertial-to-body
v_body = CBI.T @ velocity

# Aerodynamic force in body frame
v_norm = ox.linalg.Norm(velocity)
A_body = -0.5 * rho * v_norm * S_a * CA @ v_body

# Lever arm and center of pressure (in body frame, as parameters for cross products)
l_val = l.value
l_p_val = l_p.value
r_arm_vec = ox.Parameter("r_arm", shape=(3,), value=np.array([-l_val, 0.0, 0.0]))
r_cp_vec = ox.Parameter("r_cp", shape=(3,), value=np.array([l_p_val, 0.0, 0.0]))

# Quaternion kinematics using SSMP
# SSMP expects quaternion in [w, x, y, z] format (scalar first)
# Compute derivative in OpenSCvx format
q_dot_openscvx = 0.5 * ox.spatial.SSMP(angular_velocity) @ q_openscvx
# Convert back to original format [x, y, z, w] = [q1, q2, q3, q4]
q_dot = ox.Concat(q_dot_openscvx[1], q_dot_openscvx[2], q_dot_openscvx[3], q_dot_openscvx[0])

# Mass depletion
# Thrust is in body frame (control input)
m_dot = -(1.0 / (Isp * g0)) * ox.linalg.Norm(thrust) - beta

# Translation kinematics
r_dot = velocity

# Translation dynamics
# Thrust and aerodynamic force are in body frame, transform to inertial
# QDCM produces body-to-inertial DCM, so CBI @ v_body = v_inertial
v_dot = (1.0 / mass[0]) * (CBI @ (thrust + A_body)) + np.array([-gI.value, 0.0, 0.0])

# Attitude dynamics
# Torque from thrust: r_arm x T (in body frame)
# Torque from aerodynamics: r_cp x A_body (in body frame)
# Coriolis term: w x (J @ w)
torque_thrust = ox.spatial.SSM(r_arm_vec) @ thrust
torque_aero = ox.spatial.SSM(r_cp_vec) @ A_body
Jw = J_diag @ angular_velocity
coriolis = ox.spatial.SSM(angular_velocity) @ Jw
w_dot = J_inv @ (torque_thrust + torque_aero - coriolis)

dynamics = {
    "mass": m_dot,
    "position": r_dot,
    "velocity": v_dot,
    "attitude": q_dot,
    "angular_velocity": w_dot,
}

# ==========================================#
# *———* Inequality constraints (`≤ 0`) *———*#
# ==========================================#
constraints = []

# Box constraints for all states
states_list = [mass, position, velocity, attitude, angular_velocity]
for state in states_list:
    constraints.extend(
        [
            ox.ctcs(state <= state.max),
            ox.ctcs(state.min <= state),
        ]
    )

# Minimum mass constraint: m_dry - m <= 0
constraints.append(ox.ctcs(m_dry - mass[0] <= 0))

# Maximum glideslope angle: ||[r2, r3]|| - tan(gamma) * r1 <= 0
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[1:]) - np.tan(gamma.value * np.pi / 180.0) * position[0] <= 0)
)

# Maximum speed: ||v||^2 - v_max^2 <= 0
constraints.append(ox.ctcs(ox.linalg.Norm(velocity) ** 2 - v_max ** 2 <= 0))

# Maximum tilt angle: cos(theta_max) - 1 + 2*(q2^2 + q3^2) <= 0
# In original format [q1, q2, q3, q4] = [x, y, z, w], q2 and q3 are indices 1 and 2
constraints.append(
    ox.ctcs(
        ox.Cos(theta_max * np.pi / 180.0) - 1.0 + 2.0 * (attitude[1] ** 2 + attitude[2] ** 2) <= 0
    )
)

# Maximum angular speed: ||w||^2 - w_max^2 <= 0
constraints.append(ox.ctcs(ox.linalg.Norm(angular_velocity) ** 2 - w_max ** 2 <= 0))

# Maximum thrust pointing angle: ||T|| - T1 / cos(del_max) <= 0
# This constraint relates thrust magnitude to its x-component
constraints.append(
    ox.ctcs(ox.linalg.Norm(thrust) - thrust[0] / ox.Cos(del_max * np.pi / 180.0) <= 0)
)

# Maximum thrust magnitude: ||T||^2 - T_max^2 <= 0
constraints.append(ox.ctcs(ox.linalg.Norm(thrust) ** 2 - T_max ** 2 <= 0))

# Minimum thrust magnitude: T_min^2 - ||T||^2 <= 0
constraints.append(ox.ctcs(T_min ** 2 - ox.linalg.Norm(thrust) ** 2 <= 0))

# ============================#
# *———* Numerical values *———*#
# ============================#
# Time configuration
time = ox.Time(
    initial=0.0,
    final=("free", 10.0),
    min=0.0,
    max=20.0,
)

# ====================#
# *———* Settings *———*#
# ====================#
# Build the problem
problem = Problem(
    dynamics=dynamics,
    states=states_list,
    controls=[thrust],
    time=time,
    constraints=constraints,
    N=n,
)

# Set solver parameters
problem.settings.scp.k_max = 500
problem.settings.scp.w_tr_adapt = 1.04
problem.settings.scp.w_tr = 6e-1
problem.settings.scp.lam_cost = 4e-1
problem.settings.scp.lam_vc = 1.5e0

problem.settings.dis.dis_type = "ZOH"
problem.settings.dis.solver = "Dopri8"

problem.settings.cvx.solver = "CLARABEL"
problem.settings.cvx.solver_args = {"enforce_dpp": True}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process()

    print("Optimization completed!")
    print(f"Final mass: {results.trajectory['mass'][-1, 0]:.4f}")
    print(f"Final position: {results.trajectory['position'][-1, :]}")
    print(f"Final velocity: {results.trajectory['velocity'][-1, :]}")

