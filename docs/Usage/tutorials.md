# Examples
For each example we provide a link to a Google Colab notebook which can be used to interactively run the example without the need to setup an environment. The examples are also available in the `examples/` folder in the repo.

## 6DoF Obstacle Avoidance 

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

### Imports
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

### Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 6             # Number of discretization nodes
total_time = 4.0  # Initial ToF Guess for the simulation
```

### State Definition
Create a State object and configure its properties:

```python
# Create state variable
x = State("x", shape=(14,))

# Set bounds
x.max = np.array([200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100])
x.min = np.array([-200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0])

# Set initial conditions (some states are free, others are fixed)
x.initial = np.array([10.0, 0, 20, 0, 0, 0, Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), 0])

# Set final conditions (most states are free, time is minimized)
x.final = np.array([10.0, 0, 20, Free(0), Free(0), Free(0), Free(1), Free(0), Free(0), Free(0), Free(0), Free(0), Free(0), Minimize(total_time)])

# Set initial guess for SCP
x.guess = np.linspace(x.initial, x.final, n)
```

### Control Definition
Create a Control object and configure its properties:

```python
# Create control variable
u = Control("u", shape=(6,))

# Set bounds
u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])
u.min = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])

# Set initial guess for SCP
initial_control = np.array([0.0, 0, 10, 0, 0, 0])
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
```

### Dynamics
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


### Constraints
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

### Initial Guess
The initial guesses for both state and control trajectories are already set in the State and Control objects above. The state guess uses linear interpolation between initial and final conditions, while the control guess uses a constant value.

!!! tip
    The Penalized Trust Region method is very nice in that the initial guess is not required to be dynamically feasible nor satisfy constraints. However, it is a good idea to have a guess that is close to the solution to reduce the number of iterations as well as keep things numerically stable. A good place to start is a linear interpolation between the initial and final state and a constant guess for control.


```python


u_bar = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
x_bar = np.linspace(x.initial, x.final, n)

```
### Problem Instantiation
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

### Additional Parameters
Since we want this thing to run fast lets go ahead and select some fine tuning parameters. 

```python
problem.settings.scp.w_tr_adapt = 1.8          # Weight for the trust region adaptation
problem.settings.dis.custom_integrator = True  # Use the custom RK45 integrator
```

Lets also set some simulation parameters
```python
problem.settings.prp.dt = 0.01 # Time step of the nonlinear propagation
```


### Plotting
Finally, we can go ahead and plot the obstacles. We generally leave the plotting up to the users as they are usually very application specific. We do however include a few basic plots. Here we are just appending relevant information to a dictionary which can be used for plotting. 

```python
plotting_dict = dict(
    obstacles_centers=obstacle_centers,
    obstacles_axes=axes,
    obstacles_radii=radius,
)
```


### Running the Simulation
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

## 6DoF Line-of-Sight Guidance 



<a href="https://colab.research.google.com/drive/1b3NEx288h4r4HuvCOj-fexmt90PPhKUw?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>

Awesome! Now that we have a basic understanding of how to use OpenSCvx, lets go ahead and solve a more complex problem. In this example we will be solving a 6DoF line-of-sight guidance problem. This examples comes from a RA-L paper of mine which you can find [here](https://haynec.github.io/papers/los/). The problem is a bit more complex than the previous example, but the same principles apply.

In this problem, it is still a minimum time problem, but now there are 10 gates in which the drone must pass through in a predefined sequence while maintaining a line-of-sight to several key points throughout the entire trajectory. The problem can be expressed as follows:

$$
\begin{align}
\min_{x,u, t}\ &t_f, \\
\mathrm{s.t.}\ &\dot{x}(t) = f(t, x(t),u(t)) & \forall t\in[t_i, t_f], \\
& \lVert A_{\mathrm{cone}} C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp},i}_{\mathcal{I}} - r_{\mathcal{I}}(t))\rVert_\rho - c^\top C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp}, i}_{\mathcal{I}} - r_{\mathcal{I}}(t)) \leq 0 & \forall i \in [0, N_\mathrm{kp}], \forall t\in[t_i, t_f],\\
& \lVert A_{\mathrm{gate}} (r(t_i) - r^{i}_{\mathrm{gate}})\rVert_\infty \leq 1 & \forall i\in[0, N_\mathrm{gates}],\\
& x(t) \leq x_{\mathrm{max}}, x(t) \geq x_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& u(t) \leq u_{\mathrm{max}}, u(t) \geq u_{\mathrm{min}} & \forall t\in[t_i, t_f],\\
& x(0) = x_\mathrm{init}, \\
& p(t_f) = p_\mathrm{terminal}, \\
\end{align}
$$

where the state vector is the same as before, $x = \begin{bmatrix} p^\top & v^\top & q^\top & w^\top \end{bmatrix}^\top$. The control vector is also quite famaliar, $u = \begin{bmatrix}f^\top & \tau^\top \end{bmatrix}^\top$. The function $f(t, x(t),u(t))$ describes the dynamics of the drone. 

#### LoS Contraint Formulation
The constraints are where things get a little more interesting. First we have the line of sight (LoS) constraint. I find it easiest to internally break it down into the following two components,

1. 
    A transformation component which take the location of a keypoint in the inertial frame, $r^{\mathrm{kp},i}_{\mathcal{I}}$, and expresses it in the sensor frame, $r^{\mathrm{kp},i}_{\mathcal{S}}$, as follows,

    $$ r^{\mathrm{kp},i}_{\mathcal{S}} = C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp},i}_{\mathcal{I}} - r_{\mathcal{I}}(t))$$

2.  A norm cone component expressed as follows,

    $$\lVert A_{\mathrm{C}} r^{\mathrm{kp},i}_{\mathcal{S}}\rVert_\rho \leq c^\top r^{\mathrm{kp},i}_{\mathcal{S}}$$

The long expression for the LoS constraint is obtained by simply plugging the first expression into the second. 

#### Gate Constraint Formulation
The gate constraints are a little more straightforward and are notably convex.

$$\lVert A_{\mathrm{gate}} (r(t_i) - r^{i}_{\mathrm{gate}})\rVert_\infty \leq 1$$

The gate itself is assumed to be square, hence the $\infty$-norm but the user could certinaly choose a different norm. The only complication is that they are not path constraints, meaning I only want to enforce them at one single time instant as opposed to the entire trajecory and to make matters worse, the time instant is not known a priori. One could fix this but that would very likely lead to non-optimal solutions with respect to minimum time. 

### Imports
You'll need to import a few libraries to get started. The following code will import the necessary libraries for the example:

```python
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import jax.numpy as jnp

from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.utils import qdcm, SSMP, SSM, rot, gen_vertices
from openscvx.constraints import ctcs, nodal
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.control import Control
```

### Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 33            # Number of discretization nodes
total_time = 40.0 # Initial ToF Guess for the simulation
```

### State Definition
It is neccesary to define minimum and maximum elementwise bounds for the state, which will both be used for scaling purposes as well as strict constraints. Out of convience, we will also include time, $t$, in the state vector. 

```python
#                       px,   py, pz,   vx,   vy,   vz, qw, qx, qy, qz,  wx,  wy,  wz,   t
max_state = np.array([ 200,  100, 50,  100,  100,  100,  1,  1,  1,  1,  10,  10,  10, 100])
min_state = np.array([-200, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10,   0])

```

It is neccesary for the user to characterize the boundary conditions. By default all boundary conditions are assumed to be fixed, meaning ```initial_state.type = "Fixed"``` and ```final_state.type = "Fixed"```. Here since we only care that the drone finishes in minimum time, we will set the other states to be free, ```final_state.type[3:13] = "Free"```. Lastly, since we said this is a minimum time problem, we will set the time at the final state to be a minimization variable, ```final_state.type[13] = "Minimize"```.

```python
max_state = np.array([200.0, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 100])
min_state = np.array([-200.0, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0])

initial_state = boundary(jnp.array([10.0, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
initial_state.type[6:13] = "Free"

final_state = boundary(jnp.array([10.0, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, total_time]))
final_state.type[3:13] = "Free"
final_state.type[13] = "Minimize"
```
### Control Definition
Create a Control object and configure its properties:

```python
# Create control variable
u = Control("u", shape=(6,))

# Set bounds
u.max = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])
u.min = np.array([0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562])

# Set initial guess for SCP
initial_control = np.array([0.0, 0, 10, 0, 0, 0])
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
```
### Problem Parameters
We will need to define a few parameters to describe the gates, sensor and keypoints for the problem.

#### Sensor Paramters
Here we define the parameters we'll use to model the sensor with as follows,

```python
alpha_x = 6.0                                        # Angle for the x-axis of Sensor Cone
alpha_y = 6.0                                        # Angle for the y-axis of Sensor Cone
A_cone = np.diag([1 / np.tan(np.pi / alpha_x),
                  1 / np.tan(np.pi / alpha_y),
                  0,])                               # Conic Matrix in Sensor Frame
c = jnp.array([0, 0, 1])                             # Boresight Vector in Sensor Frame
norm_type = 2                                        # Norm Type
R_sb = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Rotation Matrix from Sensor to Body Frame
```

#### Gate Parameters
Here we define the parameters we'll use to model the gates with as follows,

```python
def gen_vertices(center, radii):
    """
    Obtains the vertices of the gate.
    """
    vertices = []
    vertices.append(center + rot @ [radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, radii[2]])
    vertices.append(center + rot @ [-radii[0], 0, -radii[2]])
    vertices.append(center + rot @ [radii[0], 0, -radii[2]])
    return vertices


n_gates = 10                              # Number of gates
gate_centers = [                          # Center of the gates
    np.array([ 59.436,  0.0000, 20.0000]),
    np.array([ 92.964, -23.750, 25.5240]),
    np.array([ 92.964, -29.274, 20.0000]),
    np.array([ 92.964, -23.750, 20.0000]),
    np.array([130.150, -23.750, 20.0000]),
    np.array([152.400, -73.152, 20.0000]),
    np.array([ 92.964, -75.080, 20.0000]),
    np.array([ 92.964, -68.556, 20.0000]),
    np.array([ 59.436, -81.358, 20.0000]),
    np.array([ 22.250, -42.672, 20.0000]),
]

radii = np.array([2.5, 1e-4, 2.5])                 # Radii of the gates
A_gate = rot @ np.diag(1 / radii) @ rot.T
A_gate_cen = []
for center in gate_centers:
    center[0] = center[0] + 2.5
    center[2] = center[2] + 2.5
    A_gate_cen.append(A_gate @ center)             # Premultiplying A_gate @ center to ensure OCP is DPP compliant
n_per_gate = 3                                     # Number of nodes between each gate
gate_nodes = np.arange(n_per_gate, n, n_per_gate)  # Which node to enforce the gate constraint
vertices = []
for center in gate_centers:
    vertices.append(gen_vertices(center, radii))
```

#### Keypoint Parameters
We can randomly generate some keypoints for the drone to observe. The keypoints are assumed to be in the inertial frame and can be generated as follows,

```python
n_subs = 10                          # Number of keypoints
init_poses = []
np.random.seed(0)
for i in range(n_subs):
    init_pose = np.array([100.0, -60.0, 20.0])
    init_pose[:2] = init_pose[:2] + np.random.random(2) * 20.0
    init_poses.append(init_pose)

init_poses = init_poses

```

### Dynamics
For brevity, I will not go into the details of the vehicle dynamics as they are unchanged from the previous example. 

### Constraints
In this problem, we have both continous constraints as well as discrete constraints. 
#### Continuous Constraints
To define the LoS constraints, we can define a utility 
function which very clsoely follows the mathematical definition in the above problem formulation. 
```python

def g_vp(p_s_I, x):
    p_s_s = R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[0:3])
    return jnp.linalg.norm(A_cone @ p_s_s, ord=norm_type) - (c.T @ p_s_s)
```
We can then instantiate the LoS and min-max constraints using the ```ctcs```decorator as follows. 

!!! note
    In the min-max constraints we exclude the last state which is the augmented state $y$, which models the integral of constraint violation as this state will be handled differently under the hood on ```OCP.py``` with the Linear Constraint Qualification (LICQ) constraint. 

```python
constraints = []
constraints.append(ctcs(lambda x_, u_: x_[:-1] - x.max))
constraints.append(ctcs(lambda x_, u_: x.min - x_[:-1]))
for pose in init_poses:
    constraints.append(ctcs(lambda x_, u_, p=pose: g_vp(p, x_)))
```

#### Discrete Constraints
Here we will instatiate the gate constraints by looping over the gate parameters. As these constraints are only enforced at specific nodes, we will use the ```nodal``` decorator and pass in the appropriate node for it to be enforced at. 

```python
for node, cen in zip(gate_nodes, A_gate_cen):
    constraints.append(
        nodal(lambda x, u, A=A_gate, c=cen: cp.norm(A @ x[:3] - c, "inf") <= 1,
              nodes=[node],
              convex=True,
        ) 
    )  # Use local variables inside the lambda function
```

### Initial Guess
Unlike before, just a linear interpolation isn't going to quite cut it for this problem. We will need to use a more sophisticated method to generate the initial guess. We can start with a linear interpolation in position between the start to each gate in sequence and then to the final state. 

```python
# Set up the initial guess for control
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)

# Set up the initial guess for state
x_bar = np.linspace(x.initial, x.final, n)

i = 0
origins = [x.initial[:3]]
ends = []
for center in gate_centers:
    origins.append(center)
    ends.append(center)
ends.append(x.final[:3])
gate_idx = 0
for _ in range(n_gates + 1):
    for k in range(n // (n_gates + 1)):
        x_bar[i, :3] = origins[gate_idx] + (k / (n // (n_gates + 1))) * (
            ends[gate_idx] - origins[gate_idx]
        )
        i += 1
    gate_idx += 1

# Set the state guess
x.guess = x_bar
```

### Problem Instantiation
Finally now that we have all the pieces we need, we can go ahead and instantiate the ```TrajOptProblem``` class. The class takes in the dynamics, constraints, number of discretization nodes, initial guess for ToF, initial guess for state and control, initial and final state, minimum and maximum state and control bounds. 
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

### Additional Parameters
We can define the PTR weights and other parameters as follows.

!!! tip
    Tuning is probably one of the hardest things to do when working with these type of algorithms. There are some approaches to automate this process (which will soon be included in OpenSCvx once they are published). A good place to start is to set ```lam_cost = 0```, ```lam_vc = 1E1``` and ```w_tr = 1E0```. Then you can slowly increase the cost weight and decrease the trust region weight until you find a good balance.

```python
problem.settings.scp.w_tr = 2e0                     # Weight on the Trust Reigon
problem.settings.scp.lam_cost = 1e-1                # Weight on the Cost
problem.settings.scp.lam_vc = 1e1                   # Weight on the Virtual Control
problem.settings.scp.ep_tr = 1e-3                   # Trust Region Tolerance
problem.settings.scp.ep_vc = 1e-8                   # Virtual Control Tolerance
problem.settings.scp.cost_drop = 10                 # SCP iteration to relax cost weight
problem.settings.scp.cost_relax = 0.8               # Minimal Time Relaxation Factor
problem.settings.scp.w_tr_adapt = 1.4               # Trust Region Adaptation Factor
problem.settings.scp.w_tr_max_scaling_factor = 1e2  # Maximum Trust Region Weight
```

We can also use the custom RK45 integrator to speed things up a bit. 

```python
problem.settings.dis.custom_integrator = True       # Use the custom RK45 integrator
```

Lets also set some propagation parameters
```python
problem.settings.prp.dt = 0.01 # Time step of the nonlinear propagation
```


### Plotting
Finally, we can go ahead and plot the obstacles. We generally leave the plotting up to the users as they are usually very application specific. We do however include a few basic plots. Here we are just appending relevant information to a dictionary which can be used for plotting. 

```python
plotting_dict = dict(
    obstacles_centers=obstacle_centers,
    obstacles_axes=axes,
    obstacles_radii=radius,
)
```

### Running the Simulation
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