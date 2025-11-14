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

x = ox.State(
    "x", shape=(3,)
)  # State variable with 3 dimensions (x, y, v) - time handled separately

x.max = np.array([10.0, 10.0, 10.0])  # Upper Bound on the states
x.min = np.array([0.0, 0.0, 0.0])  # Lower Bound on the states
x.initial = np.array([0, 10, 0])
x.final = [10, 5, ("free", 10)]
x.guess = np.linspace(x.initial, x.final, n)

u = ox.Control("u", shape=(1,))  # Control variable with 1 dimension
u.max = np.array([100.5 * jnp.pi / 180])  # Upper Bound on the controls
u.min = np.array([0])  # Lower Bound on the controls
u.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(
    -1, 1
)  # Reshaped as a guess needs to be set with a 2D array, in this case (n,1)

g = 9.81

x_dot = x[2] * ox.Sin(u[0])
y_dot = -x[2] * ox.Cos(u[0])
v_dot = g * ox.Cos(u[0])
dyn_expr = ox.Concat(x_dot, y_dot, v_dot)  # Time derivative handled separately
constraint_exprs = [
    ox.ctcs(x <= x.max),
    ox.ctcs(x.min <= x),
]

time = ox.Time(
    initial=0.0,
    final=("minimize", total_time),
    min=0.0,
    max=total_time,
)

problem = TrajOptProblem(
    dynamics={"x": dyn_expr},  # Dictionary mapping state name to dynamics
    states=[x],  # Wrapped in list for new API
    controls=[u],  # Wrapped in list for new API
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
