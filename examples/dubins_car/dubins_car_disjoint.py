import os
import sys

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting import plot_dubins_car_disjoint
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter
from openscvx.backend.state import Free, Minimize, State
from openscvx.constraints import ctcs, nodal
from openscvx.dynamics import dynamics
from openscvx.trajoptproblem import TrajOptProblem

n = 8
total_time = 6.0  # Total simulation time
# Define State and Control symbolic variables
x = State("x", shape=(4,))
u = Control("u", shape=(2,))
# Set bounds on state
x.min = np.array([-5.0, -5.0, -2 * jnp.pi, 0])
x.max = np.array([5.0, 5.0, 2 * jnp.pi, 20])
# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = np.array([0, -2, 0, 0])
x.final = np.array([Free(1), Free(-1.5), Free(0), Minimize(total_time)])
x.guess = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)
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


def visit_wp_OR(x_, u_, wp1_center_, wp1_radius_, wp2_center_, wp2_radius_):
    # Visit wp1 or wp2
    # Returns a value <= 0 if x_ is within either wp1 or wp2
    d1 = jnp.linalg.norm(x_[:2] - wp1_center_)
    d2 = jnp.linalg.norm(x_[:2] - wp2_center_)
    v1 = wp1_radius_ - d1
    v2 = wp2_radius_ - d2
    alpha = 10.0  # smoothing parameter; higher = closer to max
    smooth_max = (1 / alpha) * jnp.log(jnp.exp(alpha * v1) + jnp.exp(alpha * v2))
    return -smooth_max


# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(
        lambda x_, u_, wp1_radius_, wp1_center_, wp2_radius_, wp2_center_: visit_wp_OR(
            x_, u_, wp1_center_, wp1_radius_, wp2_center_, wp2_radius_
        ),
        nodes=(3, 5),
    ),
    ctcs(lambda x_, u_: x_ - x.true.max),
    ctcs(lambda x_, u_: x.true.min - x_),
    nodal(lambda x_, u_: cp.norm(x_[0][:2] - x_[-1][:2]) <= 1, convex=True, vectorized=True),
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
plotting_dict = {
    "wp1_radius": wp1_radius.value,
    "wp1_center": wp1_center.value,
    "wp2_radius": wp2_radius.value,
    "wp2_center": wp2_center.value,
}
if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)
    plot_dubins_car_disjoint(results, problem.settings).show()
