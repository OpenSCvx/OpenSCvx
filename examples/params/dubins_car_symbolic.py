import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.parameter import Parameter
from openscvx.backend.control import Control



n = 8
total_time = 2.0  # Total simulation time

# Define State and Control symbolic variables
r =     State("r", shape=(2,))
theta = State("theta", shape=(1,))
t =     State("t", shape=(1,))

v = Control("v", shape=(1,))
w = Control("w", shape=(1,))

# Set bounds on state
x.min = np.array([-5., -5., -2 * jnp.pi, 0])
x.max = np.array([ 5.,  5.,  2 * jnp.pi, 5])

# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = np.array([0, -2, Free(0), 0])
x.final   = np.array([0, 2, Free(0), Minimize(total_time)])
x.guess   = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)

# Set bounds and guess for control
v.min = np.array([0])
v.max = np.array([10])
v.guess = np.repeat(np.expand_dims(np.array([0]), axis=0), n, axis=0)

w.min = np.array([-5])
w.max = np.array([5])
w.guess = np.repeat(np.expand_dims(np.array([0]), axis=0), n, axis=0)


# Define Parameters for obstacle radius and center
obs_radius = Parameter("obs_radius")
obs_radius.value = 1.0

obs_radius = 1.0

obs_center = Parameter("obs_center", shape=(2,))
obs_center.value = np.array([-0.01, 0.0])

obs_center = np.array([-0.01, 0.0])  # Center of the obstacle

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(lambda x_var, u_var: obs_radius - jnp.linalg.norm(x_var[:2] - obs_center)),
    ctcs(lambda x_var, u_var: x_var - x.max),
    ctcs(lambda x_var, u_var: x.min - x_var)
]

# Define dynamics
@dynamics
def dynamics_fn(x_var, u_var):
    rx_dot = u_var[0] * jnp.sin(x_var[2])
    ry_dot = u_var[0] * jnp.cos(x_var[2])
    theta_dot = u_var[1]
    x_dot = jnp.asarray([rx_dot, ry_dot, theta_dot])
    t_dot = 1
    return jnp.hstack([x_dot, t_dot])

# Build the problem
problem = TrajOptProblem(
    dynamics=f,
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

# Optional: For plotting
plotting_dict = dict(
    obs_radius=obs_radius,
    obs_center=obs_center,
)