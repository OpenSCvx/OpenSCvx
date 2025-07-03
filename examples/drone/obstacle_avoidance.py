import numpy as np
import jax.numpy as jnp

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from openscvx.trajoptproblem import TrajOptProblem  # noqa: E402
from openscvx.dynamics import dynamics  # noqa: E402
from openscvx.constraints import ctcs  # noqa: E402
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors  # noqa: E402
from openscvx.backend.state import State, Free, Minimize  # noqa: E402
from openscvx.backend.parameter import Parameter  # noqa: E402
from openscvx.backend.control import Control  # noqa: E402

from examples.plotting import plot_animation  # noqa: E402

n = 6
total_time = 4.0  # Total time for the simulation

x = State("x", shape=(14,))  # State variable with 14 dimensions

x.max = np.array([200., 10, 20, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 10])
x.min = np.array(
    [-200., -100, 0, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0]
)

x.initial = np.array([10.0, 0, 2, 0, 0, 0, Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), 0])
x.final = np.array([-10.0, 0, 2, Free(0), Free(0), Free(0), Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(total_time)])

u = Control("u", shape=(6,))  # Control variable with 6 dimensions
u.max=np.array(
    [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562]
)  # Upper Bound on the controls
u.min=np.array(
    [0, 0, 0, -18.665, -18.665, -0.55562]
)  # Lower Bound on the controls
initial_control = np.array([0., 0., u.max[2], 0., 0., 0.])
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)


@dynamics
def dynamics(x_, u_):
    m = 1.0  # Mass of the drone
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone
    # Unpack the state and control vectors
    v = x_[3:6]
    q = x_[6:10]
    w = x_[10:13]

    f = u_[:3]
    tau = u_[3:]

    q_norm = jnp.linalg.norm(q)
    q = q / q_norm

    # Compute the time derivatives of the state variables
    r_dot = v
    v_dot = (1 / m) * qdcm(q) @ f + jnp.array([0, 0, g_const])
    q_dot = 0.5 * SSMP(w) @ q
    w_dot = jnp.diag(1 / J_b) @ (tau - SSM(w) @ jnp.diag(J_b) @ w)
    t_dot = 1
    return jnp.hstack([r_dot, v_dot, q_dot, w_dot, t_dot])


def g_obs(center, A, x_):
    value = 1 - (x_[:3] - center).T @ A @ (x_[:3] - center)
    return value


A_obs = []
radius = []
axes = []

# Convert obstacle centers to Parameters for real-time updates
obs_center_1 = Parameter("obs_center_1", shape=(3,))
obs_center_2 = Parameter("obs_center_2", shape=(3,))
obs_center_3 = Parameter("obs_center_3", shape=(3,))

obs_center_1.value = np.array([-5.1, 0.1, 2])
obs_center_2.value = np.array([0.1, 0.1, 2])
obs_center_3.value = np.array([5.1, 0.1, 2])

obstacle_centers = [obs_center_1, obs_center_2, obs_center_3]

np.random.seed(0)
for _ in obstacle_centers:
    ax = generate_orthogonal_unit_vectors()
    axes.append(generate_orthogonal_unit_vectors())
    rad = np.random.rand(3) + 0.1 * np.ones(3)
    radius.append(rad)
    A_obs.append(ax @ np.diag(rad**2) @ ax.T)

constraints = []
constraints.append(ctcs(lambda x_, u_, obs_center_1_: g_obs(obs_center_1_, A_obs[0], x_)))
constraints.append(ctcs(lambda x_, u_, obs_center_2_: g_obs(obs_center_2_, A_obs[1], x_)))
constraints.append(ctcs(lambda x_, u_, obs_center_3_: g_obs(obs_center_3_, A_obs[2], x_)))
constraints.append(ctcs(lambda x_, u_: x_ - x.true.max))
constraints.append(ctcs(lambda x_, u_: x.true.min - x_))

x.guess = np.linspace(x.initial, x.final, n)

problem = TrajOptProblem(
    dynamics=dynamics,
    x=x,
    u=u,
    params=Parameter.get_all(),
    constraints=constraints,
    idx_time=len(x.max)-1,
    N=n,
    licq_max=1E-8
)

problem.settings.scp.w_tr_adapt = 1.8
problem.settings.scp.w_tr = 1e1
problem.settings.scp.lam_cost = 1e1  # Weight on the Nonlinear Cost
problem.settings.scp.lam_vc = 1e2  # Weight on the Virtual Control Objective
problem.settings.scp.cost_drop = 4  # SCP iteration to relax minimal final time objective
problem.settings.scp.cost_relax = 0.5  # Minimal Time Relaxation Factor

problem.settings.prp.dt = 0.01
plotting_dict = dict(
    obstacles_centers=obstacle_centers,
    obstacles_axes=axes,
    obstacles_radii=radius,
)

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    # plot_scp_animation_pyqtgraph(results, problem.settings)
    plot_animation(results, problem.settings)