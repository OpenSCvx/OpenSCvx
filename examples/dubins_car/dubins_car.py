import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_dubins_car
from openscvx.trajoptproblem import TrajOptProblem

n = 8
total_time = 1.2  # Total simulation time

# Define State and Control symbolic variables
x = ox.State("x", shape=(4,))

u = ox.Control("u", shape=(2,))

# Set bounds on state
x.min = np.array([-5.0, -5.0, -2 * jnp.pi, 0])
x.max = np.array([5.0, 5.0, 2 * jnp.pi, 20])

# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = [0, -2, 0, 0]
x.final = [0, 2, ("free", 0), ("minimize", total_time)]
x.guess = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)

# Set bounds on control
u.min = np.array([0, -5])
u.max = np.array([10, 5])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)

# Define Parameters for obstacle radius and center
obs_center = ox.Parameter("obs_center", shape=(2,))
obs_radius = ox.Parameter("obs_radius", shape=())


# Parameter values will be set through params dictionary

# Define constraints using symbolic expressions
constraints = [
    ox.ctcs(obs_radius <= ox.linalg.Norm(x[:2] - obs_center)),
    ox.ctcs(x <= ox.Constant(x.max)),
    ox.ctcs(ox.Constant(x.min) <= x),
]


# Define dynamics using symbolic expressions
rx_dot = u[0] * ox.Sin(x[2])
ry_dot = u[0] * ox.Cos(x[2])
theta_dot = u[1]
t_dot = ox.Constant(np.array([1.0], dtype=np.float64))
dynamics = ox.Concat(rx_dot, ry_dot, theta_dot, t_dot)


# Set parameter values
params = {
    "obs_radius": 1.0,
    "obs_center": np.array([-2.01, 0.0]),
}

# Build the problem
problem = TrajOptProblem(
    dynamics=dynamics,
    x=x,
    u=u,
    params=params,
    idx_time=3,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
    licq_max=1e-8,
    time_dilation_factor_min=0.02,
)

# Set solver parameters
problem.settings.prp.dt = 0.01
# problem.settings.scp.w_tr_adapt = 1.3
problem.settings.scp.w_tr = 1e0
problem.settings.scp.lam_cost = 4e1
problem.settings.scp.lam_vc = 1e3
problem.settings.scp.uniform_time_grid = True

# Enable CLI printing for optimization iterations
problem.settings.dev.printing = True

problem.settings.cvx.cvxpygen = True
problem.settings.cvx.solver = "qocogen"
problem.settings.cvx.solver_args = {}
# problem.settings.cvx.cvxpygen_override = True


plotting_dict = {
    "obs_radius": params["obs_radius"],
    "obs_center": params["obs_center"],
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)

    plot_dubins_car(results, problem.settings).show()

    # Second run with different parameters
    params["obs_center"] = np.array([0.5, 0.0])
    total_time = 0.7  # Adjust total time for second run
    problem.settings.scp.lam_cost = 1e-1  # Disable minimal time objective for second run
    problem.settings.scp.w_tr = 1e0
    problem.settings.scp.lam_vc = 1e2  # Adjust virtual control weight
    x.guess[:, 0:4] = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)
    u.guess[:, 0:2] = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)

    plotting_dict["obs_center"] = np.array([0.5, 0.0])

    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)
    plot_dubins_car(results, problem.settings).show()
