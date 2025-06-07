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
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control

from examples.plotting import plot_brachistochrone_position, plot_brachistochrone_velocity

n = 2
total_time = 2.0

x = State("x", shape=(4,))  # State variable with 4 dimensions

x.max = np.array([10.0, 10.0, 10.0, total_time])  # Upper Bound on the states
x.min = np.array([0.0, 0.0, 0.0, 0.0])  # Lower Bound on the states
x.initial = np.array([0, 10, 0, 0])
x.final = np.array([10, 5, Free(10), Minimize(total_time)])
x.guess = np.linspace(x.initial, x.final, n)

u = Control("u", shape=(1,))  # Control variable with 1 dimension
u.max = np.array([100.5 * jnp.pi / 180])  # Upper Bound on the controls
u.min = np.array([0])  # Lower Bound on the controls
u.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(-1, 1) # Reshaped as a guess needs to be set with a 2D array, in this case (n,1)

g = 9.81

@dynamics
def dynamics(x_, u_):
    # Ensure the control is within bounds
    u_ = jnp.clip(u_, u.min, u.max)

    x_dot =  x_[2] * jnp.sin(u_[0])
    y_dot = -x_[2] * jnp.cos(u_[0])
    v_dot = g * jnp.cos(u_[0])

    t_dot = 1
    return jnp.hstack([x_dot, y_dot, v_dot, t_dot])

constraints = [
    ctcs(lambda x_, u_: x_ - x.true.max),
    ctcs(lambda x_, u_: x.true.min - x_)
]


problem = TrajOptProblem(
    dynamics=dynamics,
    x=x,
    u=u,
    idx_time=3,  # Index of time variable in state vector
    constraints=constraints,
    N=n,
    licq_max=1e-8,
)

problem.settings.prp.dt = 0.01

problem.settings.scp.w_tr_adapt = 1.00

problem.settings.cvx.solver = "qocogen"
problem.settings.cvx.cvxpygen = True

problem.settings.scp.w_tr = 1e1        # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e0    # Weight on the Minimal Time Objective
problem.settings.scp.lam_vc = 1e1      # Weight on the Virtual Control Objective
problem.settings.scp.uniform_time_grid = True

plotting_dict = dict()

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_brachistochrone_position(results).show()
    plot_brachistochrone_velocity(results).show()