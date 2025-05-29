import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import boundary, ctcs, nodal
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors


n = 2
total_time = 2.0

max_state = np.array([10.0, 10.0, 10.0, 10.0])  # Upper Bound on the states
min_state = np.array([0.0, 0.0, 0.0, 0.0])  # Lower Bound on the states
initial_state = boundary(jnp.array([0, 10, 0, 0]))

final_state = boundary(jnp.array([10, 5, 10, total_time]))
final_state.type[2] = "Free"
final_state.type[3] = "Minimize"


max_control = np.array([100.5 * jnp.pi / 180])  # Upper Bound on the controls
min_control = np.array([0])  # Lower Bound on the controls
initial_control = np.array([5 * jnp.pi / 180])
final_control = np.array([100.5 * jnp.pi / 180])

g = 9.81

@dynamics
def dynamics(x, u):
    # Ensure the control is within bounds
    u = jnp.clip(u, min_control, max_control)

    x_dot =  x[2] * jnp.sin(u[0])
    y_dot = -x[2] * jnp.cos(u[0])
    v_dot = g * jnp.cos(u[0])

    t_dot = 1
    return jnp.hstack([x_dot, y_dot, v_dot, t_dot])

constraints = [
    ctcs(lambda x, u: x - max_state),
    ctcs(lambda x, u: min_state - x)
]

x_bar = np.linspace(initial_state.value, final_state.value, n)
u_bar = np.linspace(initial_control, final_control, n)


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
    licq_max=1e-4,
)

problem.params.prp.dt = 0.01

problem.params.scp.w_tr_adapt = 1.05

# problem.params.cvx.solver = "qocogen"
# problem.params.cvx.cvxpygen = True

problem.params.scp.w_tr = 1e1        # Weight on the Trust Reigon
problem.params.scp.lam_cost = 1e1    # Weight on the Minimal Time Objective
problem.params.scp.lam_vc = 1e2      # Weight on the Virtual Control Objective
# problem.params.scp.ep_tr = 1e-4      # Trust Region Tolerance
# problem.params.scp.ep_vb = 1e-4      # Virtual Control Tolerance
problem.params.scp.uniform_time_grid = True

plotting_dict = dict()
