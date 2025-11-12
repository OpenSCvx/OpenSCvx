import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import plot_dubins_car_disjoint
from openscvx import TrajOptProblem

# NOTE: This example requires the 'stljax' package.
# You can install it via pip:
#     pip install stljax
n = 8
total_time = 4.0  # Total simulation time

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.min = np.array([-5.0, -5.0])
position.max = np.array([5.0, 5.0])
position.initial = np.array([0, -2])
position.final = np.array([0, 2])
position.guess = np.linspace(position.initial, position.final, n)

theta = ox.State("theta", shape=(1,))  # Heading angle
theta.min = np.array([-2 * jnp.pi])
theta.max = np.array([2 * jnp.pi])
theta.initial = np.array([0])
theta.final = [("free", 0)]
theta.guess = np.zeros((n, 1))

# Define control components
speed = ox.Control("speed", shape=(1,))  # Forward speed
speed.min = np.array([0])
speed.max = np.array([10])
speed.guess = np.zeros((n, 1))

angular_rate = ox.Control("angular_rate", shape=(1,))  # Angular velocity
angular_rate.min = np.array([-5])
angular_rate.max = np.array([5])
angular_rate.guess = np.zeros((n, 1))

# Define list of all states and controls
states = [position, theta]
controls = [speed, angular_rate]
# Define Parameters for wp radius and center
wp1_center = ox.Parameter("wp1_center", shape=(2,), value=np.array([-2.1, 0.0]))
wp1_radius = ox.Parameter("wp1_radius", shape=(), value=0.5)
wp2_center = ox.Parameter("wp2_center", shape=(2,), value=np.array([2.09999, 0.0]))
wp2_radius = ox.Parameter("wp2_radius", shape=(), value=0.5)


# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Sin(theta[0]),  # x_dot
        speed[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "theta": angular_rate[0],
}

# Create symbolic expressions for waypoint predicates
wp1_pred = wp1_radius - ox.linalg.Norm(position - wp1_center)
wp2_pred = wp2_radius - ox.linalg.Norm(position - wp2_center)

# Create symbolic OR expression using the new Or node
visit_wp_or_expr = ox.stl.Or(wp1_pred, wp2_pred)

# Generate box constraints for all states
constraints = []
for state in states:
    constraints.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

# Visit waypoint constraints using symbolic Or
constraints.append(ox.ctcs(-visit_wp_or_expr <= 0.0).over((3, 5)))

# Build the problem
time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=10,
)

problem = TrajOptProblem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraints,
    N=n,
)
# Set solver parameters
problem.settings.prp.dt = 0.01
problem.settings.scp.w_tr_adapt = 1.1
problem.settings.scp.w_tr = 1e0
problem.settings.scp.lam_cost = 1e-1
problem.settings.scp.lam_vc = 6e2
problem.settings.scp.uniform_time_grid = True
# Extract parameter values from problem.parameters (not Parameter objects)
plotting_dict = {
    "wp1_center": problem.parameters.get("wp1_center", None),
    "wp1_radius": problem.parameters.get("wp1_radius", None),
    "wp2_center": problem.parameters.get("wp2_center", None),
    "wp2_radius": problem.parameters.get("wp2_radius", None),
}

# Only add waypoints that are actually defined
if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)
    results.update(plotting_dict)
    plot_dubins_car_disjoint(results, problem.settings).show()
