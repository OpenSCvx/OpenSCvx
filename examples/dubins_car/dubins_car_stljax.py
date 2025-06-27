import numpy as np
import jax.numpy as jnp
import cvxpy as cp

import os
import sys

# NOTE: This example requires the 'stljax' package.
# You can install it via pip:
#     pip install stljax


from stljax.formula import Predicate, Or

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control
from examples.plotting import plot_dubins_car, plot_dubins_car_disjoint

n = 8
total_time = 4.0  # Total simulation time

# Define State and Control symbolic variables
x = State("x", shape=(4,))

u = Control("u", shape=(2,))

# Set bounds on state
x.min = np.array([-5., -5., -2 * jnp.pi,  0])
x.max = np.array([ 5.,  5.,  2 * jnp.pi, 10])

# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = np.array([0, -2, 0, 0])
x.final   = np.array([0, 2, Free(0), Minimize(total_time)])
x.guess   = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)

# Set bounds on control
u.min = np.array([0, -5])
u.max = np.array([10, 5])

# Set initial control guess
u.guess = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)

# Define Parameters for wp radius and center
wp1_center = Parameter("wp1_center", shape=(2,))
wp1_radius = Parameter("wp1_radius", shape=())

wp2_center = Parameter("wp2_center", shape=(2,))
wp2_radius = Parameter("wp2_radius", shape=())

wp1_radius.value = 0.5
wp1_center.value = np.array([-2.1, 0.0])  # Center of the wp 1

wp2_radius.value = 0.5
wp2_center.value = np.array([1.9, 0.0])  # Center of the wp 2

# Define STL predicates for each waypoint (positive inside, negative outside)
def pred_wp1(x_):
    return (wp1_radius.value - jnp.linalg.norm(x_[:2] - wp1_center.value))

def pred_wp2(x_):
    return (wp2_radius.value - jnp.linalg.norm(x_[:2] - wp2_center.value))

# STL predicates
wp1_pred = Predicate('wp1', pred_wp1)
wp2_pred = Predicate('wp2', pred_wp2)
# Logical OR: in wp1 or wp2
phi = Or(wp1_pred, wp2_pred)

# Remove visit_wp_OR and replace the first constraint with stljax-based version
constraints = [
    ctcs(lambda x_, u_: -phi(x_), nodes=(3,5)),
    ctcs(lambda x_, u_: x_ - x.true.max),
    ctcs(lambda x_, u_: x.true.min - x_),
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
    params=Parameter.get_all(),
    idx_time=3,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
    licq_max=1e-8,
)

# Set solver parameters
problem.settings.prp.dt = 0.01
problem.settings.scp.w_tr_adapt = 1.1
problem.settings.scp.w_tr = 1e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 6e2
problem.settings.scp.uniform_time_grid = True


plotting_dict = dict(
    wp1_radius = wp1_radius.value,
    wp1_center = wp1_center.value,
    wp2_radius = wp2_radius.value,
    wp2_center = wp2_center.value
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)

    plot_dubins_car_disjoint(results, problem.settings).show()