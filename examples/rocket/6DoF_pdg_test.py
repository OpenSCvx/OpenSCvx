"""6DoF PDG Rocket Trajectory Optimization"""

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
from openscvx.utils import rot, gen_vertices


n = 10

mass = ox.State("mass", shape=(1,))
mass.max = [2.0]
mass.min = [1.0]
mass.initial = [2.0]
mass.final = [ox.Maximize(1.0)]

position = ox.State("position", shape=(3,))
position.max = [10.0, 10.0, 10.0]
position.min = [-10.0, -10.0, -10.0]
position.initial = [7.5, 4.5, 2.0]
position.final = [0.0, 0.0, 0.0]

velocity = ox.State("velocity", shape=(3,))
velocity.max = [1.0, 1.0, 1.0]
velocity.min = [-1.0, -3.0, -1.0]
velocity.initial = [-0.5, -2.8, 0.0]
velocity.final = [-0.1, 0.0, 0.0]

attitude = ox.State("attitude", shape=(4,))
attitude.max = [1.0, 1.0, 1.0, 1.0]
attitude.min = [-1.0, -1.0, -1.0, -1.0]
attitude.initial = [ox.Free(0.0), ox.Free(0.0), ox.Free(0.0), ox.Free(1.0)]
attitude.final = [0.0, 0.0, 0.0, 1.0]

angular_velocity = ox.State("angular_velocity", shape=(3,))
angular_velocity.max = [0.3752, 0.3752, 0.3752]
angular_velocity.min = [-0.3752, -0.3752, -0.3752]
angular_velocity.initial = [0.0, 0.0, 0.0]
angular_velocity.final = [0.0, 0.0, 0.0]

g_I = ox.Parameter("g_I", value=1.0)
l = ox.Parameter("l", value=0.25)
J = ox.Parameter("J", value=np.array([0.168, 0.168, 0.168]))
g0 = ox.Parameter("g0", value=9.81)
Isp = ox.Parameter("Isp", value=20.0)
m_dry = ox.Parameter("m_dry", value=1.0)
v_max = ox.Parameter("v_max", value=75.0)
w_max = ox.Parameter("w_max", value=0.3752)
del_max = ox.Parameter("del_max", value=1.5)
theta_max = ox.Parameter("theta_max", value=6.5)
T_min = ox.Parameter("T_min", value=1.0)
T_max = ox.Parameter("T_max", value=3.0)
gamma = ox.Parameter("gamma", value=0.3752)
beta = ox.Parameter("beta", value=20.0)
c_ax = ox.Parameter("c_ax", value=0.5)
c_ayz = ox.Parameter("c_ayz", value=1.0)
S_a = ox.Parameter("S_a", value=0.5)
rho = ox.Parameter("rho", value=0.01)
l_p = ox.Parameter("l_p", value=0.05)  

CBI = np.array([[attitude[3]**2 + attitude[0]**2 - attitude[1]**2 - attitude[2]**2,  2*(attitude[0]*attitude[1]-attitude[3]*attitude[2]), 2*(attitude[3]*attitude[1] + attitude[0]*attitude[2])],
                [2*(attitude[3]*attitude[2] + attitude[0]*attitude[1]), attitude[3]**2 - attitude[0]**2 + attitude[1]**2 - attitude[2]**2, 2*(attitude[1]*attitude[2] - attitude[3]*attitude[0])],
                [2*(attitude[0]*attitude[2] - attitude[3]*attitude[1]), 2*(attitude[3]*attitude[0] + attitude[1]*attitude[2]), attitude[3]**2 - attitude[0]**2 - attitude[1]**2 + attitude[2]**2]]).T  # direction cosine matrix (DCM)

r_arm = ox.Parameter("r_arm", value=np.array([-l.value, 0, 0]))
J = ox.Parameter("J", value=np.diag([0.168, 0.168, 0.168]))
CA = ox.linalg.Diag(ox.Concat(c_ax, c_ayz, c_ayz))
v_body = CBI.T @ velocity
A = -0.5 * rho * v_max**2 * S_a * CA @ v_body
r_cp = ox.Parameter("r_cp", value=np.array([0.05, 0, 0]))

thrust = ox.Control("thrust", shape=(3,))
thrust.max = [T_max.value, T_max.value, T_max.value]
thrust.min = [T_min.value, T_min.value, T_min.value]
thrust.guess = np.repeat(np.array([[T_min.value, T_min.value, T_min.value]]), n, axis=0)


q1_dot = 0.5 * angular_velocity[0] * attitude[3] - 0.5 * angular_velocity[1] * attitude[2] + 0.5 * angular_velocity[2] * attitude[1]
q2_dot = 0.5 * angular_velocity[0] * attitude[2] - 0.5 * angular_velocity[2] * attitude[0] + 0.5 * angular_velocity[1] * attitude[3]
q3_dot = 0.5 * angular_velocity[1] * attitude[0] - 0.5 * angular_velocity[0] * attitude[1] + 0.5 * angular_velocity[2] * attitude[3]
q4_dot = -0.5 * angular_velocity[0] * attitude[0] - 0.5 * angular_velocity[1] * attitude[1] - 0.5 * angular_velocity[2] * attitude[2]

dynamics = {
    "mass": -(1/(Isp * g0)) * ox.linalg.Norm(thrust) - beta,
    "position": velocity,
    "velocity": (1/mass) * CBI.T @ (thrust + A) + ox.Concat(ox.Constant(-g_I.value), ox.Constant(0), ox.Constant(0)),
    "attitude": ox.Concat(q1_dot, q2_dot, q3_dot, q4_dot),
    "angular_velocity": ox.linalg.Inv(J) @ (ox.spatial.SSM(r_arm) @ thrust + ox.spatial.SSM(r_cp) @ A - ox.spatial.SSM(angular_velocity) @ J @ angular_velocity)
}

states = [mass, position, velocity, attitude, angular_velocity]
controls = [thrust]

constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

constraint_exprs.append(ox.ctcs(mass - m_dry <= 0))
constraint_exprs.append(ox.ctcs(ox.linalg.Norm(position[1:]) - ox.Tan(gamma * np.pi / 180.0) * position[0] <= 0))
constraint_exprs.append(ox.ctcs(ox.linalg.Norm(velocity)**2 - v_max**2 <= 0))
constraint_exprs.append(ox.ctcs(ox.Cos(theta_max * np.pi / 180.0) - 1.0 + 2.0 * (attitude[1]**2 + attitude[2]**2) <= 0))
constraint_exprs.append(ox.ctcs(ox.linalg.Norm(angular_velocity)**2 - w_max**2 <= 0))
constraint_exprs.append(ox.ctcs(ox.linalg.Norm(thrust) - thrust[0] / ox.Cos(del_max * np.pi / 180.0) <= 0))
constraint_exprs.append(ox.ctcs(ox.linalg.Norm(thrust)**2 - T_max**2 <= 0))
constraint_exprs.append(ox.ctcs(T_min**2 - ox.linalg.Norm(thrust)**2 <= 0))



time = ox.Time(
    initial=0.0,
    final=("minimize", 10.0),
    min=0.0,
    max=10.0,
)

problem = Problem(
    N = n,
    states = states, 
    controls = controls, 
    dynamics = dynamics, 
    constraints = constraint_exprs, 
    time = time, 
    time_dilation_factor_min = 0.2, 
    time_dilation_factor_max = 2.0
)

if __name__ == "__main__":
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()