import os
import sys

import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from examples.plotting import plot_animation
from openscvx.backend.control import Control
from openscvx.backend.expr import (
    QDCM,
    SSM,
    SSMP,
    Concat,
    Constant,
    Diag,
    Norm,
    ctcs,
)
from openscvx.backend.state import State
from openscvx.trajoptproblem import TrajOptProblem
from openscvx.utils import generate_orthogonal_unit_vectors

n = 6
total_time = 4.0  # Total time for the simulation

x = State("x", shape=(14,))  # State variable with 14 dimensions

x.max = np.array([200.0, 10, 20, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100])
x.min = np.array([-200.0, -100, 0, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0])

x.initial = [
    10.0,
    0,
    2,
    0,
    0,
    0,
    ("free", 1.0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    0,
]
x.final = [
    -10.0,
    0,
    2,
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 1.0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("free", 0),
    ("minimize", total_time),
]

u = Control("u", shape=(6,))  # Control variable with 6 dimensions
u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])  # Upper Bound on the controls
u.min = np.array([0, 0, 0, -18.665, -18.665, -0.55562])  # Lower Bound on the controls
initial_control = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)


m = 1.0  # Mass of the drone
g_const = -9.18
J_b = jnp.array([1.0, 1.0, 1.0])  # Moment of Inertia of the drone

# Create symbolic dynamics
v = x[3:6]
q = x[6:10]
q_norm = Norm(q)
q_normalized = q / q_norm
w = x[10:13]

f = u[:3]
tau = u[3:]

# Define dynamics using symbolic expressions
r_dot = v
v_dot = (Constant(1.0 / m)) * QDCM(q_normalized) @ f + Constant(
    np.array([0, 0, g_const], dtype=np.float64)
)
q_dot = Constant(0.5) * SSMP(w) @ q_normalized
J_b_inv = Constant(1.0 / J_b)
J_b_diag = Diag(Constant(J_b))
w_dot = Diag(J_b_inv) @ (tau - SSM(w) @ J_b_diag @ w)
t_dot = Constant(np.array([1.0], dtype=np.float64))
dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)


A_obs = []
radius = []
axes = []

# Define obstacle centers as constants
# TODO: (norrisg) Convert to use `Parameter`!
obstacle_centers = [
    np.array([-5.1, 0.1, 2]),
    np.array([0.1, 0.1, 2]),
    np.array([5.1, 0.1, 2]),
]

np.random.seed(0)
for _ in obstacle_centers:
    ax = generate_orthogonal_unit_vectors()
    axes.append(generate_orthogonal_unit_vectors())
    rad = np.random.rand(3) + 0.1 * np.ones(3)
    radius.append(rad)
    A_obs.append(ax @ np.diag(rad**2) @ ax.T)

constraints = [
    ctcs(x <= Constant(x.max)),
    ctcs(Constant(x.min) <= x),
]

# Add obstacle constraints using symbolic expressions
for center, A in zip(obstacle_centers, A_obs):
    center_const = Constant(center)
    A_const = Constant(A)
    pos = x[:3]

    # Obstacle constraint: (pos - center)^T @ A @ (pos - center) >= 1
    diff = pos - center_const
    obstacle_constraint = 1.0 <= diff.T @ A_const @ diff
    constraints.append(obstacle_constraint)

x.guess = np.linspace(x.initial, x.final, n)

problem = TrajOptProblem(
    dynamics=dyn_expr,
    x=x,
    u=u,
    constraints=constraints,
    idx_time=len(x.max) - 1,
    N=n,
)

problem.settings.prp.dt = 0.01
problem.settings.scp.lam_vb = 1e0
problem.settings.scp.cost_drop = 4  # SCP iteration to relax minimal final time objective
problem.settings.scp.cost_relax = 0.5  # Minimal Time Relaxation Factor

plotting_dict = {
    "obstacles_centers": obstacle_centers,
    "obstacles_axes": axes,
    "obstacles_radii": radius,
}

if __name__ == "__main__":
    problem.initialize()
    results = problem.solve()
    results = problem.post_process(results)

    results.update(plotting_dict)

    plot_animation(results, problem.settings).show()
