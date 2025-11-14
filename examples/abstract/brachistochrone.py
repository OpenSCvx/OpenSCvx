import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import openscvx as ox
from examples.plotting import (
    plot_brachistochrone_position,
    plot_brachistochrone_velocity,
)
from openscvx import TrajOptProblem

n = 2
total_time = 2.0
g = 9.81

# Define state components
position = ox.State("position", shape=(2,))  # 2D position [x, y]
position.max = np.array([10.0, 10.0])
position.min = np.array([0.0, 0.0])
position.initial = np.array([0.0, 10.0])
position.final = [10.0, 5.0]
position.guess = np.linspace(position.initial, position.final, n)

velocity = ox.State("velocity", shape=(1,))  # Scalar speed
velocity.max = np.array([10.0])
velocity.min = np.array([0.0])
velocity.initial = np.array([0.0])
velocity.final = [("free", 10.0)]
velocity.guess = np.linspace(0.0, 10.0, n).reshape(-1, 1)

# Define control
theta = ox.Control("theta", shape=(1,))  # Angle from vertical
theta.max = np.array([100.5 * jnp.pi / 180])
theta.min = np.array([0.0])
theta.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1)

# Define list of all states (needed for TrajOptProblem and constraints)
states = [position, velocity]
controls = [theta]

# Define dynamics as dictionary mapping state names to their derivatives
dynamics = {
    "position": ox.Concat(
        velocity[0] * ox.Sin(theta[0]),  # x_dot
        -velocity[0] * ox.Cos(theta[0]),  # y_dot
    ),
    "velocity": g * ox.Cos(theta[0]),
}

# Generate box constraints for all states
constraint_exprs = []
for state in states:
    constraint_exprs.extend([ox.ctcs(state <= state.max), ox.ctcs(state.min <= state)])

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = TrajOptProblem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=time,
    constraints=constraint_exprs,
    N=n,
    licq_max=1e-8,
)

problem.settings.prp.dt = 0.01

# problem.settings.cvx.solver = "qocogen"
# problem.settings.cvx.cvxpygen = True
problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}

problem.settings.scp.w_tr = 1e1  # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
problem.settings.scp.uniform_time_grid = True

problem.settings.sim.save_compiled = False

plotting_dict = {}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_brachistochrone_position(results).show()
    plot_brachistochrone_velocity(results).show()
