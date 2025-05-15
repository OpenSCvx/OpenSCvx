import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import boundary, ctcs, nodal
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors

n = 8
total_time = 2  # Total time for the simulation

#                      rx,  ry,     theta,   t
max_state = np.array([ 5.,  5.,  2*jnp.pi,  50])
min_state = np.array([-5., -5., -2*jnp.pi,   0])

initial_state = boundary(jnp.array([0, -2, 0, 0]))

final_state = boundary(jnp.array([0, 2, 0, total_time]))
final_state.type[2] = "Free"
final_state.type[3] = "Minimize"

#                           v, omega
initial_control = np.array([0,    0.])
max_control = np.array([10,  5])  # Upper Bound on the controls
min_control = np.array([ 0, -5])  # Lower Bound on the controls


@dynamics
def dynamics(x, u):
    rx_dot = u[0] * jnp.sin(x[2])
    ry_dot = u[0] * jnp.cos(x[2])
    theta_dot = u[1]
    x_dot = jnp.array([rx_dot, ry_dot, theta_dot])

    t_dot = 1
    return jnp.hstack([x_dot, t_dot])

obs_radius = 1
obs_center = np.array([-0.01, 0])
constraints = [
    ctcs(lambda x, u: obs_radius - jnp.linalg.norm(x[:2] - obs_center)),
    ctcs(lambda x, u: x - max_state),
    ctcs(lambda x, u: min_state - x)
]


u_bar = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
x_bar = np.linspace(initial_state.value, final_state.value, n)

problem = TrajOptProblem(
    dynamics=dynamics,
    constraints=constraints,
    idx_time=len(max_state)-1,
    N=n,
    time_init=total_time,
    x_guess=x_bar,
    u_guess=u_bar,
    initial_state=initial_state,  # Initial State
    final_state=final_state,
    x_max=max_state,
    x_min=min_state,
    u_max=max_control,
    u_min=min_control,
    licq_max=1e-8,
)

problem.params.prp.dt = 0.01

problem.params.scp.w_tr_adapt = 1.1

problem.params.scp.w_tr = 1e0        # Weight on the Trust Reigon
problem.params.scp.lam_cost = 1e-1   # Weight on the Minimal Time Objective
problem.params.scp.lam_vc = 1e2      # Weight on the Virtual Control Objective
problem.params.scp.uniform_time_grid = True

problem.params.dis.custom_integrator = False

plotting_dict = dict(
    obs_radius = obs_radius,
    obs_center = obs_center,
)