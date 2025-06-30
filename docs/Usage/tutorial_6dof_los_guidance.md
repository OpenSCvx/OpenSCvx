# 6DoF Line-of-Sight Guidance

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

### LoS Contraint Formulation
The constraints are where things get a little more interesting. First we have the line of sight (LoS) constraint. I find it easiest to internally break it down into the following two components,

1. 
    A transformation component which take the location of a keypoint in the inertial frame, $r^{\mathrm{kp},i}_{\mathcal{I}}$, and expresses it in the sensor frame, $r^{\mathrm{kp},i}_{\mathcal{S}}$, as follows,

    $$ r^{\mathrm{kp},i}_{\mathcal{S}} = C(q_{\mathcal{S}\to\mathcal{B}})C(q_{\mathcal{B}\to\mathcal{I}}(t))(r^{\mathrm{kp},i}_{\mathcal{I}} - r_{\mathcal{I}}(t))$$

2.  A norm cone component expressed as follows,

    $$\lVert A_{\mathrm{C}} r^{\mathrm{kp},i}_{\mathcal{S}}\rVert_\rho \leq c^\top r^{\mathrm{kp},i}_{\mathcal{S}}$$

The long expression for the LoS constraint is obtained by simply plugging the first expression into the second. 

### Gate Constraint Formulation
The gate constraints are a little more straightforward and are notably convex.

$$\lVert A_{\mathrm{gate}} (r(t_i) - r^{i}_{\mathrm{gate}})\rVert_\infty \leq 1$$

The gate itself is assumed to be square, hence the $\infty$-norm but the user could certinaly choose a different norm. The only complication is that they are not path constraints, meaning I only want to enforce them at one single time instant as opposed to the entire trajecory and to make matters worse, the time instant is not known a priori. One could fix this but that would very likely lead to non-optimal solutions with respect to minimum time. 

## Imports
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

## Problem Definition
Lets first define the number of discretization nodes and an initial guess for ToF.

```python
n = 33            # Number of discretization nodes
total_time = 40.0 # Initial ToF Guess for the simulation
```

## State Definition
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

## Control Definition
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

## Problem Parameters
We will need to define a few parameters to describe the gates, sensor and keypoints for the problem.

### Sensor Paramters
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

### Gate Parameters
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

### Keypoint Parameters
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

## Dynamics
For brevity, I will not go into the details of the vehicle dynamics as they are unchanged from the previous example. 

## Constraints
In this problem, we have both continous constraints as well as discrete constraints. 
### Continuous Constraints
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

### Discrete Constraints
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

## Initial Guess
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

## Problem Instantiation
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

## Additional Parameters
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