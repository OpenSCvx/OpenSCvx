# Basic Problem Setup
Here we will cover all the neccesary elements to setup your problem along with some tips and best practices to get the most out of the package.

## State Specification
To specify the state, four elements are required:

1. The maximum value of each element of the state
2. The minimum value of each element of the state
3. The initial value of each element of the state
4. The terminal value of each element of the state

For maximum and minimum values, this is done as follows,

```python
max_state = np.array(["Insert your max state here"])
min_state = np.array(["Insert your min state here"])
```
For the initial and terminal values, this is done as follows,

```python
from openscvx.constraints.boundary import BoundaryConstraint as bc
initial_state = bc(jnp.array(["Insert your initial state here"]))
terminal_state = bc(jnp.array(["Insert your terminal state here"]))
```
By default, all elements in the initial and terminal state are assumed to be fixed. To change this, you can modify the ```initial_state.type[] = "Type"``` and ```final_state.type``` attributes. The options are: ```"Fix", "Free", "Minimize"```.

## Control Specification
To specify the control, it is sufficient to specify the maximum and minimum values of each element of the control. This is done as follows,

```python
max_control = np.array(["Maximum value of each element of the control"])
min_control = np.array(["Minimum value of each element of the control"])
```

## Dynamics
The dynamics function must take the following form:

```python
def dynamics(x, u):
    "Insert your dynamics functions here"
    return jnp.hstack(["Stack up your dynamics here"]) 
```

Here ```x``` is the state and ```u``` is the control. The function must return a vector of the same size as the state. Augmented states can be created here to capture costs that are not already defined within the states. For example, if you have a minimum fuel problem, you can do so by adding a new state, $\mathrm{fuel} = \int^{t_f}_{t_i}\|u\|_2\, dt$, in which case the dynamics will simply be $\dot{\mathrm{fuel}} = \|u\|_2$. 

```python
def dynamics(x, u):
    "Insert your dynamics functions here"

    fuel_dot = jnp.linalg.norm(u)
    return jnp.hstack(["Stack up your dynamics here", fuel_dot]) 
```

## Constraints
We support a both continuous and discrete constraints using the decorators ```ctcs``` and ```nodal```. 

### Continuous Constraints
To specify continuous constraints, you can use the ```ctcs``` decorator. This decorator takes a function that returns a scalar. This function must take the following form:```g(x, u, **args) <= 0```. 

```python
constraints.append(ctcs(lambda x, u, args = args: g(x, u, **args)))
```

### Nodal Constraints
To specify discrete or nodal constraints you can use the ```nodal``` decorator.

```python
constraints.append(nodal(lambda x, u, args: g(x, u, **args)))
```

If this constraint is convex, please set ```convex = True``` and use [```cvxpy``` atoms](https://www.cvxpy.org/tutorial/functions/index.html) to define the constraint as a boolean expression. For example,  
```python
constraints.append(nodal(lambda x, u, A=A_gate, c=cen: cp.norm(A @ x[:3] - c, "inf") <= 1,convex=True))
```
By default this constraint will be applied at all nodes. To change this, you can add ```nodes = ["Insert a list of the desired nodes here"]``` to the decorator.

If the constraint is nonconvex, either set ```convex = False``` or leave it out as it defualts to ```False```. This will linearize the constraint and apply it at the nodes specified in the decorator. For example, 

```python
constraints.append(nodal(lambda x, u, c=center, A=A_obs_s: g_obs(x, u, c, A), convex=False))
```

## Initial Guess
This is a very important part of the problem setup. While it is not strictly neccesary for the inital guess to be dynamically feasiable or satisfy constraints, it is best practice to use an initial guess that is as close to the solution as possible. This will help the solver converge faster and avoid local minima. Generally a decent first place to start is to linearly interpolate between the initial and terminal states. This can be done as follows,

```python
x_bar = np.linspace(initial_state.value, final_state.value, n)
```

and to keep the control constant, you can do so as follows,

```python
u_bar = np.repeat(np.expand_dims(initial_control, axis=0), n, axis=0)
```

## Instantiate the Problem
The problem can be instantiated as follows,
```python
problem = TrajOptProblem(
    dynamics=dynamics,
    constraints=constraints,
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
)
```

## Choose SCvx Weights
The weights are used to scale the cost, trust region, and dynamic feasibility. A good place to start is to set ```lam_cost = 0, lam_vc = 1E1``` and ```w_tr = 1E0```. Then you can slowly increase the cost weight and decrease the trust region weight until you find a good balance.

```python
problem.params.scp.w_tr = 1E0  # Weight on the Trust Reigon
problem.params.scp.lam_cost = 0E0  # Weight on the Cost
problem.params.scp.lam_vc = 1E1  # Weight on the Virtual Control Objective
```
and if you have nonconvex nodal constraints


If you have nonconvex nodal constraints then you will also need to include ``` params.scp.lam_vb = 1E0```. 

## Running the Simulation
To run the simulation, follow these steps:

1. Initialize the problem:
   ```python
   problem.initialize()
   ```
   
2. Solve the Problem:
   ```python
   problem.solve()
   ```
3. Postprocess the solution for verification and plotting:
   ```python
   results = problem.post_process()
   results.update(plotting_dict)
   ```