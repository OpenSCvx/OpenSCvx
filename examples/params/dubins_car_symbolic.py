import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.backend.expr import State, Control, Parameter, Free, Minimize


n = 8
total_time = 2.0  # Total simulation time

# Define State and Control symbolic variables
x = State("x", shape=(4,))
u = Control("u", shape=(2,))

# Set bounds on state
x.min = np.array([-5., -5., -2 * jnp.pi, 0])
x.max = np.array([5., 5., 2 * jnp.pi, 50])

# Set initial, final, and guess for state trajectory using symbolic boundary expressions
x.initial = np.array([0, -2, Free(0), 0])
x.final   = np.array([0, 2, Free(0), Minimize(total_time)])
x.guess   = np.linspace([0, -2, 0, 0], [0, 2, 0, total_time], n)

# Set bounds and guess for control
u.min = np.array([0, -5])
u.max = np.array([10, 5])
u.guess = np.repeat(np.expand_dims(np.array([0, 0]), axis=0), n, axis=0)

# Define Parameters for obstacle radius and center
obs_radius = Parameter("obs_radius")
obs_radius.value = 1.0

obs_center = Parameter("obs_center", shape=(2,))
obs_center.value = np.array([-0.01, 0.0])

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(lambda x, u, obs_center, obs_radius: obs_radius - jnp.linalg.norm(x[:2] - obs_center)),
    ctcs(lambda x, u: x - x.max),
    ctcs(lambda x, u: x.min - x)
]

# Define dynamics
@dynamics
def dynamics_fn(x, u):
    rx_dot = u[0] * sin(x[2])
    ry_dot = u[0] * cos(x[2])
    theta_dot = u[1]
    x_dot = asarray([rx_dot, ry_dot, theta_dot])
    t_dot = 1
    return hstack([x_dot, t_dot])

# Build the problem
problem = TrajOptProblem(
    dynamics=dynamics_fn,
    constraints=constraints,
    idx_time=3,
    N=n,
    time_init=total_time,
    x=x,
    u=u,
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
    obs_radius=obs_radius.value,
    obs_center=obs_center.value,
)