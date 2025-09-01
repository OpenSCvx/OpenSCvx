import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting import (
    plot_brachistochrone_position,
    plot_brachistochrone_velocity,
)
from openscvx.backend.canonicalizer import canonicalize
from openscvx.backend.control import Control
from openscvx.backend.expr import Concat, Constant, Cos, Sin
from openscvx.backend.lower import lower_to_jax
from openscvx.backend.preprocessing import (
    collect_and_assign_slices,
    validate_constraints_at_root,
    validate_dynamics_dimension,
    validate_shapes,
    validate_variable_names,
)
from openscvx.backend.state import Free, Minimize, State
from openscvx.constraints import ctcs
from openscvx.dynamics import dynamics
from openscvx.trajoptproblem import TrajOptProblem

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
u.guess = np.linspace(5 * jnp.pi / 180, 100.5 * jnp.pi / 180, n).reshape(
    -1, 1
)  # Reshaped as a guess needs to be set with a 2D array, in this case (n,1)

g = 9.81


x_dot = x[2] * Sin(u[0])
y_dot = -x[2] * Cos(u[0])
v_dot = g * Cos(u[0])
t_dot = 1
dyn_expr = Concat(x_dot, y_dot, v_dot, t_dot)
constraint_exprs = [
    x <= Constant(np.array([x.max])),
    Constant(np.array([x.min])) <= x,
]

# Validate expressions
all_exprs = [dyn_expr] + constraint_exprs
validate_variable_names(all_exprs)
collect_and_assign_slices(all_exprs)
validate_shapes(all_exprs)
validate_constraints_at_root(constraint_exprs)
validate_dynamics_dimension(dyn_expr, x)

# Canonicalize all expressions after validation
dyn_expr = canonicalize(dyn_expr)
constraint_exprs = [canonicalize(expr) for expr in constraint_exprs]

dyn_fn = lower_to_jax(dyn_expr)
fns = lower_to_jax(constraint_exprs)

dyn = dynamics(dyn_fn)
constraints = [ctcs(fn) for fn in fns]
# constraints = [ctcs(lambda x_, u_: x_ - x.true.max), ctcs(lambda x_, u_: x.true.min - x_)]


problem = TrajOptProblem(
    dynamics_fn=dyn,
    x=x,
    u=u,
    idx_time=3,  # Index of time variable in state vector
    constraints_fn=constraints,
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

plotting_dict = {}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_brachistochrone_position(results).show()
    plot_brachistochrone_velocity(results).show()
