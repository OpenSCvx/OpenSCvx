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
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control
from examples.plotting import plot_dubins_car

n = 8
total_time = 2.0  # Total simulation time

# Define State and Control symbolic variables
x = State("x", shape=(4,))

u = Control("u", shape=(2,))

# Set bounds on state
x.min = np.array([-5., -5., -2 * jnp.pi,  0])
x.max = np.array([ 5.,  5.,  2 * jnp.pi, 50])

# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = np.array([0, -2, 0, 0])
x.final   = np.array([0, 2, Free(0), Minimize(total_time)])
x.guess   = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)

# Set bounds on control
u.min = np.array([0, -5])
u.max = np.array([10, 5])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)

# Define Parameters for obstacle radius and center
obs_center = Parameter("obs_center", shape=(2,))
obs_radius = Parameter("obs_radius", shape=())


obs_radius.value = 1.0
obs_center.value = np.array([-0.01, 0.0])  # Center of the obstacle

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(lambda x_, u_, obs_radius_, obs_center_: obs_radius_ - jnp.linalg.norm(x_[:2] - obs_center_)),
    ctcs(lambda x_, u_: x_ - x.true_state.max),
    ctcs(lambda x_, u_: x.true_state.min - x_)
]

# Define dynamics
@dynamics
def dynamics_fn(x_, u_):
    rx_dot = u_[0] * jnp.sin(x_[2])
    ry_dot = u_[0] * jnp.cos(x_[2])
    theta_dot = u_[1]
    x_dot = jnp.asarray([rx_dot, ry_dot, theta_dot])
    t_dot = 1
    return jnp.hstack([x_dot, t_dot])

# Build the problem
problem = TrajOptProblem(
    dynamics=dynamics_fn,
    x=x,
    u=u,
    params = Parameter.get_all(),
    idx_time=3,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
    licq_max=1e-8,
)

# Set solver parameters
problem.settings.prp.dt = 0.01
problem.settings.scp.w_tr_adapt = 1.3
problem.settings.scp.w_tr = 1e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 6e2
problem.settings.scp.uniform_time_grid = True


plotting_dict = dict(
    obs_radius=obs_radius,
    obs_center=obs_center,
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)

    plot_dubins_car(results, problem.settings).show()


    # Second run with different parameters
    obs_center.value = np.array([0.5, 0.0])
    total_time = 0.7  # Adjust total time for second run
    problem.settings.scp.lam_cost = 1E-1  # Disable minimal time objective for second run
    problem.settings.scp.w_tr = 1e0
    problem.settings.scp.lam_vc = 1e2  # Adjust virtual control weight
    x.guess[:,0:4]   = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)
    u.guess[:,0:2] = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)
    u.guess[:,2] = np.repeat(total_time, n)  # Adjust initial control guess


    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)
    plot_dubins_car(results, problem.settings).show()