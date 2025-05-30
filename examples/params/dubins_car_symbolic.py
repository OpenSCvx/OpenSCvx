import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs, nodal
from openscvx.backend.expr import State, Control, Parameter, Free, Minimize


n = 8
total_time = 2.0  # Total simulation time

# Define State and Control symbolic variables
r =     State("r", shape=(2,))
theta = State("theta", shape=(1,))
t =     State("t", shape=(1,))

v = Control("v", shape=(1,))
w = Control("w", shape=(1,))

# Set bounds on state
r.min = np.array([0., -5.])
r.max = np.array([5., 5.])
r.initial = np.array([0, -2])
r.final   = np.array([0, 2])
r.guess = np.linspace([0, -2], [0, 2], n)

theta.min = -2 * jnp.pi
theta.max = 2 * jnp.pi
theta.initial = Free(0)
theta.final   = Free(0)
theta.guess = np.linspace(0, 0, n)

t.min = 0
t.max = 5
t.initial = Free(0)
t.final   = Minimize(total_time)
t.guess = np.linspace(0, total_time, n)

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

obs_center = Parameter("obs_center", shape=(2,))
obs_center.value = np.array([-0.01, 0.0])

# Define constraints using symbolic x, u, and parameters
constraints = [
    ctcs(obs_radius - norm(x[:2] - obs_center)),
    ctcs(r - r.max),
    ctcs(r.min - r),
    ctcs(theta - theta.max),
    ctcs(theta.min - theta),
    ctcs(t - t.max),
    ctcs(t.min - t)
]

# Define dynamics
f = [u[0] * sin(theta),
     u[0] * cos(theta),
     u[1],
     1]

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
    obs_radius=obs_radius.value,
    obs_center=obs_center.value,
)