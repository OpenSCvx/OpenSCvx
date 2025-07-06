# Basic Problem Setup
Here we will cover all the necessary elements to setup your problem along with some tips and best practices to get the most out of the package.

## State Specification
To specify the state, you create a `State` object and configure its properties:

```python
from openscvx.backend.state import State, Free, Fix, Minimize, Maximize

# Create state variable
x = State("x", shape=(n_states,))

# Set bounds
x.min = np.array([min_values_for_each_state])
x.max = np.array([max_values_for_each_state])

# Set initial conditions (can be fixed values or boundary condition objects)
x.initial = np.array([value1, Free(guess2), Fix(value3), Minimize(guess4)])

# Set final conditions (can be fixed values or boundary condition objects)
x.final = np.array([value1, Free(guess2), Maximize(guess3), Minimize(guess4)])

# Set initial guess for SCP (shape: (n_nodes, n_states))
x.guess = np.linspace(
    initial_values, final_values, n_nodes
)
```

The boundary condition options are:

- `Fix(value)` - Fixed value that cannot be optimized
- `Free(guess)` - Free variable that can be optimized within bounds
- `Minimize(guess)` - Variable to be minimized
- `Maximize(guess)` - Variable to be maximized

## Control Specification
To specify the control, you create a `Control` object and configure its properties:

```python
from openscvx.backend.control import Control

# Create control variable
u = Control("u", shape=(n_controls,))

# Set bounds
u.min = np.array([min_values_for_each_control])
u.max = np.array([max_values_for_each_control])

# Set initial guess for SCP (shape: (n_nodes, n_controls))
u.guess = np.repeat(
    np.expand_dims(initial_control, axis=0), 
    n_nodes, axis=0
)
```

## Dynamics
The dynamics function must take the following form:

```python
from openscvx.dynamics import dynamics

@dynamics
def dynamics(x_, u_):
    "Insert your dynamics functions here"
    return jnp.hstack(["Stack up your dynamics here"]) 
```

Here `x` is the state and `u` is the control. The function must return a vector of the same size as the state.

!!! Note
    Under the hood the dynamics and constraint functions will be compiled using JAX so it is necessary to only use `jax` functions and `jax.ndarrays` within the dynamics and nonconvex constraint functions.

## Costs
You can choose states to either maximize or minimize by using the `Minimize` and `Maximize` boundary condition objects in the state's `initial` or `final` properties. Additional augmented states can be created automatically by the system to capture costs that are not already defined within the states.

## Constraints
We support both continuous and discrete constraints using the decorators `ctcs` and `nodal`. 

### Continuous Constraints
To specify continuous constraints, you can use the `ctcs` decorator. This decorator takes a function that returns a scalar. This function must take the following form: `g(x, u, **args) <= 0`. 

```python
from openscvx.constraints import ctcs
a = 1
b = 2.0
constraints = [
    ctcs(lambda x_, u_: g1(x_, u_)),
    ctcs(lambda x_, u_: g2(x_, u_, a, b))
]
```

The input state `x` will be an array of shape `(n_x)` and the control `u` will be an array of shape `(n_u)`. The function must return a scalar value.

### Nodal Constraints
To specify constraints that are evaluated at specific nodes, use the `nodal` decorator:

```python
from openscvx.constraints import nodal

constraints = [
    nodal(lambda x_, u_: g(x_, u_), nodes=[0, 5, 10])  # Evaluate at nodes 0, 5, 10
]
```

## Parameters
You can define parameters that can be used in constraints and dynamics:

```python
from openscvx.backend.parameter import Parameter

# Create parameters
obs_center = Parameter("obs_center", shape=(2,))
obs_radius = Parameter("obs_radius", shape=())

# Set parameter values
obs_center.value = np.array([1.0, 2.0])
obs_radius.value = 0.5

# Use in constraints
constraints.append(
    ctcs(lambda x_, u_, obs_center_, obs_radius_: 
         obs_radius_ - jnp.linalg.norm(x_[:2] - obs_center_))
)
```

## Initial Guess
This is a very important part of the problem setup. While it is not strictly necessary for the initial guess to be dynamically feasible or satisfy constraints, it is best practice to use an initial guess that is as close to the solution as possible. This will help the solver converge faster and avoid local minima.

For state trajectories, a good starting point is to linearly interpolate between the initial and final states:

```python
x.guess = np.linspace(x.initial, x.final, n_nodes)
```

For control trajectories, you can use a constant guess:

```python
u.guess = np.repeat(np.expand_dims(initial_control, axis=0), n_nodes, axis=0)
```

## Instantiate the Problem
The problem can be instantiated as follows:

```python
from openscvx.trajoptproblem import TrajOptProblem

problem = TrajOptProblem(
    dynamics=dynamics,
    constraints=constraints,
    x=x,
    u=u,
    idx_time=time_index,  # Index of time variable in state vector
    N=n_nodes,
)
```

## Configure SCP Weights
The weights are used to scale the cost, trust region, and dynamic feasibility. A good place to start is to set `lam_cost = 0`, `lam_vc = 1E1` and `w_tr = 1E0`. Then you can slowly increase the cost weight and decrease the trust region weight until you find a good balance.

```python
problem.settings.scp.w_tr = 1E0      # Weight on the Trust Region
problem.settings.scp.lam_cost = 0E0  # Weight on the Cost
problem.settings.scp.lam_vc = 1E1    # Weight on the Virtual Control Objective
```

If you have nonconvex nodal constraints then you will also need to include `problem.settings.scp.lam_vb = 1E0`.

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