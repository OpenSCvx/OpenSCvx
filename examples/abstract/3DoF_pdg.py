import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting import plot_control_norm, plot_xy_xz_yz
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter
from openscvx.backend.state import Free, Maximize, State
from openscvx.constraints import ctcs
from openscvx.dynamics import dynamics
from openscvx.trajoptproblem import TrajOptProblem

n = 30
total_time = 95.0  # Total simulation time

# Define State and Control symbolic variables
x = State("x", shape=(8,))

# Set bounds on state
v_max = 500 * 1e3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
#                      x       y     z      vx      vy      vz     m    t
x.min = np.array([-3000, -3000, 0, -v_max, -v_max, -v_max, 1505, 0])
x.max = np.array([3000, 3000, 3000, v_max, v_max, v_max, 1905, 1e2])

# Set initial, final, and guess
x.initial = np.array([2000, 0, 1500, 80, 30, -75, 1905, 0])
x.final = np.array([0, 0, 0, 0, 0, 0, Maximize(1590), Free(total_time)])
x.guess = np.linspace(x.initial, x.final, n)

u = Control("u", shape=(3,))

T_bar = 3.1 * 1e3
T1 = 0.3 * T_bar
T2 = 0.8 * T_bar
n_eng = 6

# Set bounds on control
u.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
u.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2) / 2]), axis=0), n, axis=0)


# Define Parameters for obstacle radius and center
I_sp = Parameter("I_sp")
I_sp.value = 225  # Specific impulse in seconds

g = Parameter("g")
g.value = 3.7114  # Gravitational acceleration on Mars in m/s^2

g_e = 9.807  # Gravitational acceleration on Earth in m/s^2

theta = Parameter("theta")
theta.value = 27 * jnp.pi / 180  # Cant angle of the thrusters in radians

rho_min = n_eng * T1 * np.cos(theta.value)  # Minimum thrust-to-weight ratio
rho_max = n_eng * T2 * np.cos(theta.value)  # Maximum thrust-to-weight ratio

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(lambda x_, u_: x_ - x.true.max, idx=0),
    ctcs(lambda x_, u_: x.true.min - x_, idx=0),
    ctcs(lambda x_, u_: rho_min - jnp.linalg.norm(u_[:3]), idx=1, scaling=1e-4),
    ctcs(lambda x_, u_: jnp.linalg.norm(u_[:3]) - rho_max, idx=1, scaling=1e-4),
    ctcs(
        lambda x_, u_: jnp.cos((180 - 40) * jnp.pi / 180) - u_[2] / jnp.linalg.norm(u_[:3]), idx=2
    ),
    ctcs(
        lambda x_, u_: jnp.linalg.norm(jnp.array([x_[0], x_[1]]))
        - jnp.tan((86) * jnp.pi / 180) * x_[2],
        idx=3,
    ),
    # nodal(lambda x_, u_: u_[:2] == 0, nodes = [-1], convex = True)22
]


# Define dynamics
@dynamics
def dynamics_fn(x_, u_, I_sp_, g_, theta_):
    m = x_[6]

    T = u_

    r_dot = x_[3:6]

    g_vec = jnp.array([0, 0, g_])  # Gravitational acceleration vector

    v_dot = T / m - g_vec

    m_dot = -jnp.linalg.norm(T) / (I_sp_ * g_e * jnp.cos(theta_))

    t_dot = 1
    return jnp.hstack([r_dot, v_dot, m_dot, t_dot])


# Build the problem
problem = TrajOptProblem(
    dynamics=dynamics_fn,
    x=x,
    u=u,
    params=Parameter.get_all(),
    idx_time=7,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
    licq_max=1e-6,
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
