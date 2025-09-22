import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_control_norm, plot_xy_xz_yz
from openscvx.trajoptproblem import TrajOptProblem

n = 30
total_time = 95.0  # Total simulation time

# Define State and Control symbolic variables
x = ox.State("x", shape=(8,))

# Set bounds on state
v_max = 500 * 1e3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
#                      x       y     z      vx      vy      vz     m    t
x.min = np.array([-3000, -3000, 0, -v_max, -v_max, -v_max, 1505, 0])
x.max = np.array([3000, 3000, 3000, v_max, v_max, v_max, 1905, 1e2])

# Set initial, final, and guess
x.initial = [2000, 0, 1500, 80, 30, -75, 1905, 0]
x.final = [0, 0, 0, 0, 0, 0, ("maximize", 1590), ("free", total_time)]
x.guess = np.linspace(x.initial, x.final, n)

u = ox.Control("u", shape=(3,))

T_bar = 3.1 * 1e3
T1 = 0.3 * T_bar
T2 = 0.8 * T_bar
n_eng = 6

# Set bounds on control
u.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
u.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2) / 2]), axis=0), n, axis=0)


# Define Parameters for physical constants
g_e = 9.807  # Gravitational acceleration on Earth in m/s^2

# Create parameters for the problem
I_sp = ox.Parameter("I_sp")
g = ox.Parameter("g")
theta = ox.Parameter("theta")

# These will be computed symbolically in constraints
theta_val = 27 * np.pi / 180  # Cant angle value for parameter setup
rho_min = n_eng * T1 * np.cos(theta_val)  # Minimum thrust-to-weight ratio
rho_max = n_eng * T2 * np.cos(theta_val)  # Maximum thrust-to-weight ratio

# Define constraints using symbolic expressions
constraints = [
    # State bounds
    ox.ctcs(x <= ox.Constant(x.max), idx=0),
    ox.ctcs(ox.Constant(x.min) <= x, idx=0),
    # Thrust magnitude constraints
    ox.ctcs(ox.Constant(rho_min) <= ox.linalg.Norm(u[:3]), idx=1),
    ox.ctcs(ox.linalg.Norm(u[:3]) <= ox.Constant(rho_max), idx=1),
    # Thrust pointing constraint (thrust cant angle)
    ox.ctcs(ox.Constant(np.cos((180 - 40) * np.pi / 180)) <= u[2] / ox.linalg.Norm(u[:3]), idx=2),
    # Glideslope constraint
    ox.ctcs(ox.linalg.Norm(x[:2]) <= ox.Constant(np.tan(86 * np.pi / 180)) * x[2], idx=3),
]


# Define dynamics using symbolic expressions
m = x[6]
T = u
r_dot = x[3:6]
g_vec = ox.Constant(np.array([0, 0, 1], dtype=np.float64)) * g  # Gravitational acceleration vector
v_dot = T / m - g_vec
m_dot = -ox.linalg.Norm(T) / (I_sp * ox.Constant(g_e) * ox.Constant(np.cos(theta_val)))
t_dot = ox.Constant(np.array([1], dtype=np.float64))
dynamics_expr = ox.Concat(r_dot, v_dot, m_dot, t_dot)


# Set parameter values
params = {
    "I_sp": 225,  # Specific impulse in seconds
    "g": 3.7114,  # Gravitational acceleration on Mars in m/s^2
    "theta": theta_val,  # Cant angle of the thrusters in radians
}

# Build the problem
problem = TrajOptProblem(
    dynamics=dynamics_expr,
    x=x,
    u=u,
    params=params,
    idx_time=7,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
)

# Apply custom scaling to the mass term (assume mass is at index 6 in the state vector)
problem.settings.sim.scaling_x_overrides = [
    # (1800, 1505, 6),
    # (200, 0, 7)
]

# problem.settings.sim.scaling_u_overrides = [
#     # Example: (upper_bound, lower_bound, idx)
#     (n_eng * T_bar, -n_eng * T_bar, 0),  # Custom scaling for control 0
#     (n_eng * T_bar, -n_eng * T_bar, 1),  # Custom scaling for control 1
#     (n_eng * T_bar, -n_eng * T_bar, 2),  # Custom scaling for control 2
# ]


# Set solver parameters
problem.settings.prp.dt = 0.01

problem.settings.sim.save_compiled = False

problem.settings.scp.w_tr_adapt = 1.04
problem.settings.scp.w_tr = 2e0
problem.settings.scp.lam_cost = 2.5e-1
problem.settings.scp.lam_vc = 1.2e0
# problem.settings.scp.ep_tr = 5e-3
# problem.settings.scp.ep_vc = 1e-10

# problem.settings.dis.solver = "Dopri8"

problem.settings.cvx.solver = "CLARABEL"
problem.settings.cvx.solver_args = {"enforce_dpp": True}


plotting_dict = {
    "rho_min": rho_min,
    "rho_max": rho_max,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)

    # plot_animation_3DoF_rocket(results, problem.settings).show()
    # plot_scp_animation(results, problem.settings).show()
    # plot_state(results, problem.settings).show()
    # plot_control(results, problem.settings).show()
    plot_control_norm(results, problem.settings).show()
    plot_xy_xz_yz(results, problem.settings).show()

    # If installed with extras, you can use the following to plot with pyqtgraph
    # plot_animation_pyqtgraph(results, problem.settings)
