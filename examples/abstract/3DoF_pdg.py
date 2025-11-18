import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_control_norm, plot_xy_xz_yz
from openscvx import TrajOptProblem

n = 30
total_time = 95.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(3,))  # 3D position [x, y, z]
v_max = 500 * 1e3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
position.max = np.array([3000, 3000, 3000])
position.min = np.array([-3000, -3000, 0])
position.initial = np.array([2000, 0, 1500])
position.final = np.array([0, 0, 0])
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(3,))  # 3D velocity [vx, vy, vz]
velocity.max = np.array([v_max, v_max, v_max])
velocity.min = np.array([-v_max, -v_max, -v_max])
velocity.initial = np.array([80, 30, -75])
velocity.final = np.array([0, 0, 0])
velocity.guess = np.linspace(velocity.initial, velocity.final, n)

mass = ox.State("mass", shape=(1,))  # Vehicle mass
mass.max = np.array([1905])
mass.min = np.array([1505])
mass.initial = np.array([1905])
mass.final = [("maximize", 1590)]
mass.guess = np.linspace(mass.initial, 1590, n).reshape(-1, 1)

# Define control
thrust = ox.Control("thrust", shape=(3,))  # Thrust force vector [Tx, Ty, Tz]

T_bar = 3.1 * 1e3
T1 = 0.3 * T_bar
T2 = 0.8 * T_bar
n_eng = 6

# Set bounds on control
thrust.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
thrust.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
thrust.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2) / 2]), axis=0), n, axis=0)

# Define list of all states and controls
states = [position, velocity, mass]
controls = [thrust]


# Define Parameters for physical constants
g_e = 9.807  # Gravitational acceleration on Earth in m/s^2

# Create parameters for the problem
I_sp = ox.Parameter("I_sp", value=225.0)
g = ox.Parameter("g", value=3.7114)
theta = ox.Parameter("theta", value=27 * np.pi / 180)

# These will be computed symbolically in constraints
rho_min = n_eng * T1 * np.cos(theta.value)  # Minimum thrust-to-weight ratio
rho_max = n_eng * T2 * np.cos(theta.value)  # Maximum thrust-to-weight ratio

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend(
        [
            ox.ctcs(state <= state.max, idx=0),
            ox.ctcs(state.min <= state, idx=0),
        ]
    )

# Thrust magnitude constraints
constraints.extend(
    [
        ox.ctcs(rho_min <= ox.linalg.Norm(thrust), idx=1),
        ox.ctcs(ox.linalg.Norm(thrust) <= rho_max, idx=1),
    ]
)

# Thrust pointing constraint (thrust cant angle)
constraints.append(
    ox.ctcs(np.cos((180 - 40 * np.pi / 180)) <= thrust[2] / ox.linalg.Norm(thrust), idx=2)
)

# Glideslope constraint
constraints.append(
    ox.ctcs(ox.linalg.Norm(position[:2]) <= np.tan(86 * np.pi / 180) * position[2], idx=3)
)


# Define dynamics as dictionary mapping state names to their derivatives
g_vec = np.array([0, 0, 1], dtype=np.float64) * g  # Gravitational acceleration vector

dynamics = {
    "position": velocity,
    "velocity": thrust / mass[0] - g_vec,
    "mass": -ox.linalg.Norm(thrust) / (I_sp * g_e * ox.Cos(theta)),
}

# Build the problem
time = ox.Time(
    initial=0.0,
    final=("free", total_time),
    min=0.0,
    max=1e2,
)

problem = TrajOptProblem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)

# Apply custom scaling overrides (if needed)
# Note: In the new API, state indices refer to components within each state in the states list
# states = [position (3D), velocity (3D), mass (1D)] -> indices 0-2, 3-5, 6
problem.settings.sim.scaling_x_overrides = [
    # Example: (upper_bound, lower_bound, idx)
    # (1800, 1505, 6),  # Custom scaling for mass (index 6 in flattened state)
]

# problem.settings.sim.scaling_u_overrides = [
#     # Example: (upper_bound, lower_bound, idx)
#     (n_eng * T_bar, -n_eng * T_bar, 0),  # Custom scaling for thrust[0]
#     (n_eng * T_bar, -n_eng * T_bar, 1),  # Custom scaling for thrust[1]
#     (n_eng * T_bar, -n_eng * T_bar, 2),  # Custom scaling for thrust[2]
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
