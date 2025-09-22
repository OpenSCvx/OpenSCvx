import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting import plot_dubins_car_disjoint
from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Concat,
    Constant,
    Cos,
    Exp,
    Log,
    Norm,
    Parameter,
    Sin,
    ctcs,
)
from openscvx.backend.state import State
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
x.initial = [0, -2, 0, 0]
x.final = [("free", 1), ("free", -1.5), ("free", 0), ("minimize", total_time)]
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

# Create symbolic expressions for the dynamics
pos = x[:2]
theta = x[2]
time = x[3]
velocity = u[0]
angular_velocity = u[1]

# Define dynamics using symbolic expressions
rx_dot = velocity * Sin(theta)
ry_dot = velocity * Cos(theta)
theta_dot = angular_velocity
t_dot = Constant(1.0)
dyn_expr = Concat(rx_dot, ry_dot, theta_dot, t_dot)


# Create symbolic visit waypoint OR constraint
def create_visit_wp_OR_expr():
    # Visit wp1 or wp2 using smooth max
    d1 = Norm(pos - wp1_center)
    d2 = Norm(pos - wp2_center)
    v1 = wp1_radius - d1
    v2 = wp2_radius - d2
    alpha = Constant(10.0)  # smoothing parameter; higher = closer to max
    smooth_max = (Constant(1.0) / alpha) * Log(Exp(alpha * v1) + Exp(alpha * v2))
    return -smooth_max


visit_wp_expr = create_visit_wp_OR_expr()


# Define constraints using symbolic expressions
constraints = [
    # Visit waypoint constraints using smooth max
    ctcs(visit_wp_expr <= Constant(0.0)).over((3, 5)),
    # State bounds constraints
    ctcs(x <= Constant(x.max)),
    ctcs(Constant(x.min) <= x),
    # TODO: (norrisg) Make the `cp.norm(x_[0][:2] - x_[-1][:2]) <= 1` work, allow cross-nodal 
    # constraints
]


# Set parameter values
params = {
    "wp1_center": np.array([-2.1, 0.0]),
    "wp1_radius": 0.5,
    "wp2_center": np.array([1.9, 0.0]),
    "wp2_radius": 0.5,
}

# Build the problem
problem = TrajOptProblem(
    dynamics=dyn_expr,
    x=x,
    u=u,
    params=params,
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
    "wp1_radius": params["wp1_radius"],
    "wp1_center": params["wp1_center"],
    "wp2_radius": params["wp2_radius"],
    "wp2_center": params["wp2_center"],
}
if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)
    plot_dubins_car_disjoint(results, problem.settings).show()
