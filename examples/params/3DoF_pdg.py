import numpy as np
import jax.numpy as jnp

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.backend.state import State, Free, Maximize, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control

from examples.plotting import plot_scp_animation, plot_control_norm, plot_xy_xz_yz, plot_animation_3DoF_rocket
from openscvx.plotting import plot_state, plot_control

n = 30
total_time = 75.0  # Total simulation time

# Define State and Control symbolic variables
x = State("x", shape=(8,))

# Set bounds on state
v_max = 800 * 1E3 / 3600  # Maximum velocity in m/s (800 km/h converted to m/s)
#                      x       y     z      vx      vy      vz     m    t
x.min = np.array([ -3000,   -3000,    0,  -v_max, -v_max, -v_max, 1534,   0])
x.max = np.array([  3000,    3000, 3000,   v_max,  v_max,  v_max, 1905, 2E2])

# Set initial, final, and guess
x.initial = np.array([2000, 0, 1500,  80,  30,  -75,           1905,                  0])
x.final   = np.array([   0, 0,    0,   0,   0,    0,  Maximize(1590),   Free(total_time)])
x.guess   = np.linspace(x.initial, x.final, n)

u = Control("u", shape=(3,))

T_bar = 3.1 * 1E3
T1 = 0.3 * T_bar
T2 = 0.8 * T_bar
n_eng = 6

# Set bounds on control
u.min = n_eng * np.array([-T_bar, -T_bar, -T_bar])
u.max = n_eng * np.array([T_bar, T_bar, T_bar])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0, n_eng * (T2)/2]), axis=0), n, axis=0)


# Define Parameters for obstacle radius and center
I_sp = Parameter("I_sp")
I_sp.value = 225  # Specific impulse in seconds

g = Parameter("g")
g.value = 3.7114 # Gravitational acceleration on Mars in m/s^2

g_e = 9.807      # Gravitational acceleration on Earth in m/s^2

theta = Parameter("theta")
theta.value = 27 * jnp.pi / 180  # Cant angle of the thrusters in radians

rho_min = n_eng * T1 * np.cos(theta.value) # Minimum thrust-to-weight ratio
rho_max = n_eng * T2 * np.cos(theta.value) # Maximum thrust-to-weight ratio

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(lambda x_, u_: x_ - x.true.max, idx=0),
    ctcs(lambda x_, u_: x.true.min - x_, idx=0),
    ctcs(lambda x_, u_: rho_min - jnp.linalg.norm(u_[:3]), idx=1, scaling=1E-5),
    ctcs(lambda x_, u_: jnp.linalg.norm(u_[:3]) - rho_max, idx=1, scaling=1E-5),
    ctcs(lambda x_, u_: jnp.cos((180-40) * jnp.pi/180) - u_[2] / jnp.linalg.norm(u_[:3]), idx=2),
    ctcs(lambda x_, u_: jnp.linalg.norm(jnp.array([x_[0], x_[1]])) - jnp.tan((86) * jnp.pi / 180) * x_[2], idx=3),
    nodal(lambda x_, u_: u_[:2] == 0, nodes = [-1], convex = True)
]

# Define dynamics
@dynamics
def dynamics_fn(x_, u_, I_sp_, g_, theta_):
    m = x_[6]
    
    T = u_

    r_dot = x_[3:6]
    
    g_vec = jnp.array([0, 0, g_])  # Gravitational acceleration vector
    
    v_dot = T/m - g_vec

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
    licq_max=1e-4,
)

# Set solver parameters
problem.settings.prp.dt = 0.01

problem.settings.scp.w_tr_adapt = 1.04
problem.settings.scp.w_tr = 3e0
problem.settings.scp.lam_cost = 1e2
problem.settings.scp.lam_vc = 1e0
problem.settings.scp.ep_tr = 1e-6
problem.settings.scp.ep_vc = 1e-10
# problem.settings.scp.uniform_time_grid = True
# problem.settings.scp.k_max = 30

problem.settings.cvx.solver = "CLARABEL"
problem.settings.cvx.solver_args = {"enforce_dpp": True}


plotting_dict = dict(
    rho_min = rho_min,
    rho_max = rho_max,
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)

    plot_animation_3DoF_rocket(results, problem.settings).show()
    # plot_scp_animation(results, problem.settings).show()
    # plot_state(results, problem.settings).show()
    # plot_control(results, problem.settings).show()
    plot_control_norm(results, problem.settings).show()
    plot_xy_xz_yz(results, problem.settings).show()