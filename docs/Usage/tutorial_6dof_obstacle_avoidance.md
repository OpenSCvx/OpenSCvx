# 6DoF Obstacle Avoidance

<a href="https://colab.research.google.com/drive/1xLPC_UJWC35oPRIAY3vkxi8WEYnHCysQ?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

This example demonstrates how to use OpenSCvx to solve a trajectory optimization problem in which a drone will navigate around obstacles to fly from point A to point B in minimum time. We will solve this problem in 6DoF, meaning there is 6 degrees of freedom in the problem, mainly 3 translational and 3 rotational degrees. Mathematically we can express this problem as the following,

$$
\begin{align}
\min_{x,u, t}\ &t_f, \\
\mathrm{s.t.}\ &\dot{x}(t) = f(t, x(t),u(t)) & \forall t\in[t_i, t_f], \\
& 1- (p(t) - p^i_{\mathrm{obs}})^\top A^i_\mathrm{obs} (r(t) - r^i_{\mathrm{obs}}) \leq 0  & \forall t\in[t_i, t_f], \forall i\in[0, N_\mathrm{obs}],\\
& x(t) \leq x_{\mathrm{max}}, x(t) \geq x_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& u(t) \leq u_{\mathrm{max}}, u(t) \geq u_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& x(0) = x_\mathrm{init}, \\
& p(t_f) = p_\mathrm{terminal}, \\
\end{align}
$$

where the state vector $x$ is expressed as  $x = \begin{bmatrix} r^\top & v^\top & q^\top & w^\top \end{bmatrix}^\top$. $p$ denotes the position of the drone, $v$ is the velocity, $q$ is the quaternion, $w$ is the angular velocity. The control vector $u$ is expressed as $u = \begin{bmatrix}f^\top & \tau^\top \end{bmatrix}^\top$. Here $f$ is the force in the body frame and $\tau$ is the torque of the body frame relative to the inertial frame. The function $f(t, x(t),u(t))$ describes the dynamics of the drone. The term $1- (r(t) - r^i_{\mathrm{obs}})^\top A^i_\mathrm{obs} (r(t) - r^i_{\mathrm{obs}})$ describes the obstacle avoidance constraints for $N_\mathrm{obs}$ number of obstacles, where $A_\mathrm{obs}$ is a positive definite matrix that describes the shape of the obstacle.

## Imports
You'll need to import a few libraries to get started. The following code will import the necessary libraries for the example:

```python
import numpy as np
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs
from openscvx.utils import qdcm, SSMP, SSM, generate_orthogonal_unit_vectors
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.control import Control
```

## Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 6             # Number of discretization nodes
total_time = 4.0  # Initial ToF Guess for the simulation
```

## State Definition
Create a State object and configure its properties:

```python
# Create state variable
x = State("x", shape=(14,))

# Set bounds
x.max = np.array([
    200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100
])
x.min = np.array([
    -200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0
])

# Set initial conditions (some states are free, others are fixed)
x.initial = np.array([
    10.0, 0, 20, 0, 0, 0, Free(1), Free(0), Free(0), 
    Free(0), Free(0), Free(0), Free(0), 0
])

# Set final conditions (most states are free, time is minimized)
x.final = np.array([
    10.0, 0, 20, Free(0), Free(0), Free(0), Free(1), 
    Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(total_time)
])

# Set initial guess for SCP
x.guess = np.linspace(x.initial, x.final, n)
```

## Control Definition
Create a Control object and configure its properties:

```python
# Create control variable
u = Control("u", shape=(6,))

# Set bounds
u.max = np.array([
    0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562
])
u.min = np.array([
    0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562
])

# Set initial guess for SCP
initial_control = np.array([0.0, 0, 10, 0, 0, 0])
u.guess = np.repeat(
    np.expand_dims(initial_control, axis=0), n, axis=0
)
```

## Dynamics
To describe the dynamics of the drone, lets first introduce some notation to describe in what frame quantities are being represented in. A quantity expressed in the frame $\mathcal{A}$ is denoted by the subscript $\Box_{\mathcal{A}}$. To parameterize the attitude of frame $\mathcal{B}$ with respect to frame $\mathcal{A}$, the unit quaternion, $q_{\mathcal{A} \to \mathcal{B}} \in \mathcal{S}^3$ where $\mathcal{S}^3\subset\mathbb{R}^4$ is the unit 3-sphere, is used. Here the inertial and body frames are denoted by $\mathcal{I}$ and $\mathcal{B}$ respectively. The dynamics of the drone can be expressed as follows:

$$
\begin{align*}
    % \label{eq:6dof_def}
    & \dot{r}_\mathcal{I}(t) = v_\mathcal{I}(t),\\
    & \dot{v}_\mathcal{I}(t) = \frac{1}{m}\left(C(q_{\mathcal{B \to I}}(t)) f_{ \mathcal{B}}(t)\right) + g_{\mathcal{I}},\\
    & \dot{q}_{\mathcal{I}\to \mathcal{B}} = \frac{1}{2} \Omega(\omega_\mathcal{B}(t))  q_{\mathcal{I \to B}}(t),\\
    & \dot{\omega}_\mathcal{B}(t) =  J_{\mathcal{B}}^{-1} \left(M_{\mathcal{B}}(t) - \left[\omega_\mathcal{B}(t)\times\right]J_{\mathcal{B}} \omega_\mathcal{B}(t) \right),
\end{align*} 
$$

where the operator $C:\mathcal{S}^3\mapsto SO(3)$ represents the direction cosine matrix (DCM), where $SO(3)$ denotes the special orthogonal group. Mathematically this is expressed as,

$$
C(q_{\mathcal{B \to I}}) = \begin{bmatrix}
    1 - 2(q_2^2 + q_3^2) & 2(q_1q_2 - q_3q_0) & 2(q_1q_3 + q_2q_0) \\
    2(q_1q_2 + q_3q_0) & 1 - 2(q_1^2 + q_3^2) & 2(q_2q_3 - q_1q_0) \\
    2(q_1q_3 - q_2q_0) & 2(q_2q_3 + q_1q_0) & 1 - 2(q_1^2 + q_2^2)
\end{bmatrix}
$$

Programatically we can express this as follows:
```python
def qdcm(q: jnp.ndarray) -> jnp.ndarray:
    # Convert a quaternion to a direction cosine matrix
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )
```

For a vector $\xi \in \mathbb{R}^3$, the skew-symmetric operators $\Omega(\xi)$ and $[\xi \times]$ are defined as follows 

$$
\begin{align}
[\xi \times] = \begin{bmatrix} 0 & -\xi_3 & \xi_2 \\ \xi_2 & 0 & -\xi_1 \\ -\xi_2 & \xi_1 & 0 \end{bmatrix}, \ 
\Omega(\xi) = \begin{bmatrix} 0 & -\xi_1 & \xi_2 & \xi_3 \\ \xi_1 & 0 & \xi_3 & -\xi_2 \\ \xi_2 & -\xi_3 & 0 & \xi_1 \\ \xi_3 & \xi_2 & -\xi_1 & 0 \end{bmatrix}
\end{align} 
$$

Again programmatically we can express this as follows:
```python
def SSMP(w: jnp.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])


def SSM(w: jnp.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
```

Finally, now that we have all the pieces we need, we can express the dynamics of the drone in a single function. The function takes in the state and control vectors and returns the time derivative of the state vector. 

!!! Note
    The time state derivative is handled under the hood as opposed to being explicitly defined in the dynamics function. This is done to allow for more flexibility in the problem formulation. The time state derivative is handled by the ```TrajOptProblem``` class and is not exposed to the user.

```python
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
```

## Constraints
To define the obstacle avoidance constraints, we can define a utility 
function which very clsoely follows the mathematical definition in the above problem formulation. 
```python

def g_obs(center, A, x):
    value = 1 - (x[:3] - center).T @ A @ (x[:3] - center)
    return value
```

Lets go ahead and define what our obstacles will look like. To get a more fundamental understanding of how I am parameterizing ellipsoids, I would direct the interested reader [here](https://tcg.mae.cornell.edu/pubs/Pope_FDA_08.pdf). 

```python
A_obs = []
radius = []
axes = []

def generate_orthogonal_unit_vectors(vectors=None):
    """
    Generates 3 orthogonal unit vectors to model the axis of the ellipsoid via QR decomposition

    Parameters:
    vectors (np.ndarray): Optional, axes of the ellipsoid to be orthonormalized.
                            If none specified generates randomly.

    Returns:
    np.ndarray: A 3x3 matrix where each column is a unit vector.
    """
    if vectors is None:
        key = jax.random.PRNGKey(0)

        vectors = jax.random.uniform(key, (3, 3))
    Q, _ = jnp.linalg.qr(vectors)
    return Q

obstacle_centers = [
    np.array([-5.1, 0.1, 2]),
    np.array([0.1, 0.1, 2]),
    np.array([5.1, 0.1, 2]),
]

np.random.seed(0)   # Lets randomly generate some obstacles
for _ in obstacle_centers:
    ax = generate_orthogonal_unit_vectors()
    axes.append(generate_orthogonal_unit_vectors())
    rad = np.random.rand(3) + 0.1 * np.ones(3)
    radius.append(rad)
    A_obs.append(ax @ np.diag(rad**2) @ ax.T)

```

Now lets go ahead and instantiate all of the constraints. Since we wish to enforce the the drone collision free for *all time*, not only at node points, we will use the ```ctcs``` decorator to create a constraint function. We will also enforce our minimum and maximum state constraints using the same decorator. Control minimum and maximum constraints are handled under the hood by ```OCP.py```.

!!! note
    In the min-max constraints we exclude the last state which is the augmented state $y$, which models the integral of constraint violation as this state will be handled differently under the hood on ```OCP.py``` with the Linear Constraint Qualification (LICQ) constraint.

```python
constraints = []
for center, A in zip(obstacle_centers, A_obs):
    constraints.append(ctcs(lambda x_, u_: g_obs(center, A, x_)))     # Obstacle Avoidance Constraint
constraints.append(ctcs(lambda x_, u_: x_[:-1] - x.max))          # Max State Constraint
constraints.append(ctcs(lambda x_, u_: x.min - x_[:-1]))          # Min State Constraint
```

## Initial Guess
The initial guesses for both state and control trajectories are already set in the State and Control objects above. The state guess uses linear interpolation between initial and final conditions, while the control guess uses a constant value.

!!! tip
    The Penalized Trust Region method is very nice in that the initial guess is not required to be dynamically feasible nor satisfy constraints. However, it is a good idea to have a guess that is close to the solution to reduce the number of iterations as well as keep things numerically stable. A good place to start is a linear interpolation between the initial and final state and a constant guess for control.

```python
u_bar = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
x_bar = np.linspace(x.initial, x.final, n)
```

## Problem Instantiation
Finally now that we have all the pieces we need, we can go ahead and instantiate the ```TrajOptProblem``` class. The class takes in the dynamics, constraints, State and Control objects, and number of discretization nodes.

```python
problem = TrajOptProblem(
    dynamics=dynamics,
    constraints=constraints,
    x=x,
    u=u,
    idx_time=13,  # Index of time variable in state vector
    N=n,
    licq_max=1e-8,
)
```

## Additional Parameters
Since we want this thing to run fast lets go ahead and select some fine tuning parameters. 

```python
problem.settings.scp.w_tr_adapt = 1.8          # Weight for the trust region adaptation
problem.settings.dis.custom_integrator = True  # Use the custom RK45 integrator
```

Lets also set some simulation parameters
```python
problem.settings.prp.dt = 0.01 # Time step of the nonlinear propagation
```

## Plotting
Finally, we can go ahead and plot the obstacles. We generally leave the plotting up to the users as they are usually very application specific. We do however include a few basic plots. Here we are just appending relevant information to a dictionary which can be used for plotting. 

```python
plotting_dict = dict(
    obstacles_centers=obstacle_centers,
    obstacles_axes=axes,
    obstacles_radii=radius,
)
```

## Running the Simulation
To run the simulation, follow these steps:

1. Initialize the problem:
   ```python
   problem.initialize()
   ```
   
2. Solve the Problem:
   ```python
   results = problem.solve()
   ```

3. Postprocess the solution for verification and plotting:
   ```python
   results = problem.post_process(results)
   results.update(plotting_dict)
   ``` 