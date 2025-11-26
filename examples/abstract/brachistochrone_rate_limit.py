"""Brachistochrone problem with cross-node rate limiting constraints.

This example demonstrates the use of cross-node constraints with NodeReference
to limit the distance traveled between consecutive trajectory nodes. This shows
how inter-node constraints can be used to enforce smoothness and continuity.

The rate limit constraint ensures that the bead doesn't teleport large distances
between nodes, which can happen with coarse discretizations. By tuning the
max_step parameter, you can see how tight constraints affect convergence:
- max_step too small: Problem becomes infeasible (can't reach goal)
- max_step too large: No effect on solution
- max_step just right: Smooths the trajectory while maintaining optimality
"""

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

# Problem parameters
n = 2
total_time = 2.0
g = 9.81

# Rate limiting parameter - try different values to see the effect:
# Limit is sqrt(125)
max_step = np.sqrt(124.9)  # Maximum distance allowed between consecutive nodes

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

# ==================== CROSS-NODE RATE LIMITING CONSTRAINT ====================
# This is the key addition: limit the distance between consecutive nodes
# Using NodeReference with relative indexing: ||position[k] - position[k-1]|| <= max_step

# Create the cross-node constraint expression using relative indexing
# 'k' represents the current node, 'k-1' represents the previous node
pos_k = position.node("k")  # Position at current node
pos_k_prev = position.node("k-1")  # Position at previous node

# Compute the distance between consecutive nodes
step_distance = ox.linalg.Norm(pos_k - pos_k_prev, ord=2)

# Create inequality constraint: step_distance <= max_step
# For n=2, we only have one interval (from node 0 to node 1)
# The constraint evaluates at node 1, where 'k'=1 and 'k-1'=0
rate_limit_constraint = (step_distance <= max_step).at([1])

# Add to constraint list
constraint_exprs.append(rate_limit_constraint)

# ==================== PROBLEM SETUP ====================

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

problem.settings.cvx.solver_args = {"abstol": 1e-6, "reltol": 1e-9}

# problem.settings.cvx.solver = "qocogen"
# problem.settings.cvx.cvxpygen = True
problem.settings.scp.w_tr = 1e1  # Weight on the Trust Region
problem.settings.scp.lam_cost = 1e0  # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e1  # Weight on the Virtual Control Objective
problem.settings.scp.uniform_time_grid = True

problem.settings.sim.save_compiled = False

plotting_dict = {}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    # Compute and display actual step sizes
    position_traj = results.trajectory["position"]
    step_sizes = []
    for i in range(1, len(position_traj)):
        step_size = np.linalg.norm(position_traj[i] - position_traj[i - 1])
        step_sizes.append(step_size)

    results.update(plotting_dict)

    plot_brachistochrone_position(results).show()
    plot_brachistochrone_velocity(results).show()
