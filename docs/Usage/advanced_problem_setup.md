# Advanced Problem Setup

## Using Parameters in Dynamics and Constraints

OpenSCvx allows you to define symbolic parameters that can be used in both dynamics and constraints. This enables flexible, reusable problem definitions. 

!!! Note
    When using parameters, the argument names in your functions must match the parameter name with an underscore suffix (e.g., `g_` for a parameter named `g`).


### Example: 3DoF Rocket Landing Dynamics with Parameters

```python
from openscvx.backend.parameter import Parameter
from openscvx.dynamics import dynamics
import jax.numpy as jnp

I_sp = Parameter("I_sp")
g = Parameter("g")
theta = Parameter("theta")

I_sp.value = 225
g.value = 3.7114
theta.value = 27 * jnp.pi / 180

@dynamics
def rocket_dynamics(x_, u_, I_sp_, g_, theta_):
    m = x_[6]
    T = u_
    r_dot = x_[3:6]
    g_vec = jnp.array([0, 0, g_])
    v_dot = T/m - g_vec
    m_dot = -jnp.linalg.norm(T) / (I_sp_ * 9.807 * jnp.cos(theta_))
    t_dot = 1
    return jnp.hstack([r_dot, v_dot, m_dot, t_dot])
```

When building your problem, collect all parameters with `Parameter.get_all()` and pass them to your problem setup:

```python
problem = TrajOptProblem(
    dynamics=rocket_dynamics,
    x=x,
    u=u,
    params=Parameter.get_all(),
    ...
)
```

---

## Using Parameters in Constraints

You can also use parameters in both CTCS and nodal constraints. Again, the argument names must match the parameter name with an underscore.

### CTCS Constraint Example with Parameters

```python
from openscvx.backend.parameter import Parameter
from openscvx.constraints import ctcs
import jax.numpy as jnp

obs_center = Parameter("obs_center", shape=(2,))
obs_radius = Parameter("obs_radius", shape=())
obs_center.value = jnp.array([-2.01, 0.0])
obs_radius.value = 1.0

@ctcs
def obstacle_avoidance(x_, u_, obs_center_, obs_radius_):
    return obs_radius_ - jnp.linalg.norm(x_[:2] - obs_center_)
```

### Nodal Constraint Example with Parameters

```python
from openscvx.backend.parameter import Parameter
from openscvx.constraints import nodal
import jax.numpy as jnp

g = Parameter("g")
g.value = 3.7114

@nodal
def terminal_velocity_constraint(x_, u_, g_):
    # Enforce a terminal velocity constraint using the gravity parameter
    return x_[5] + g_ * x_[7]  # e.g., vz + g * t <= 0 at final node
```

---

## CTCS Constraints: Advanced Options

### Penalty Function

You can specify a penalty function for CTCS constraints using the `penalty` flag. Built-in options include:

- `squared_relu` - $\max(0, g)^2$
- `huber` - $\begin{cases} \frac{1}{2} g^2 & \text{if } |g| \leq \delta \\ \delta (|g| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$
- `smooth relu` - $\|\max(0, g)+c\|-c$

Example:

```python
@ctcs(penalty="huber")
def g(x_, u_):
    return jnp.linalg.norm(x_[:2]) - 1.0
```

Or provide a custom penalty function:

```python
# True Huber function with delta=0.25
huber = lambda x: jnp.where(jnp.abs(x) <= 0.25, 0.5 * x**2, 0.25 * (jnp.abs(x) - 0.5 * 0.25))
@ctcs(penalty=huber)
def g(x_, u_):
    return jnp.linalg.norm(x_[:2]) - 1.0
```

### Node-Specific Regions

To enforce a constraint only over a portion of the trajectory:

```python
@ctcs(nodes=(3, 8))
def g(x_, u_):
    return jnp.linalg.norm(x_[:2]) - 1.0
```

### Multiple Augmented States

To associate different constraints with different augmented states:

```python
@ctcs(idx=0)
def g1(x_, u_):
    return jnp.linalg.norm(x_[:2]) - 1.0

@ctcs(idx=1)
def g2(x_, u_):
    return x_ - max_state
```

---

## Nodal Constraints: Advanced Options

### Convex and Nonconvex Nodal Constraints

- For convex constraints, set `convex=True` and use cvxpy atoms:

```python
@nodal(convex=True)
def g(x_, u_):
    return cp.norm(x_) <= 1.0
```

- For nonconvex constraints, use the default (`convex=False`) and express as $g(x,u) \leq 0$:

```python
@nodal
def g(x_, u_):
    return 1 - x_[0]
```

### Node Ranges

To enforce a constraint at specific nodes:

```python
@nodal(nodes=[4])
def g(x_, u_):
    return x_[0] - x_wp
```

### Vectorized Nodal Constraints

To enforce a constraint between nodes or to apply a custom vectorized constraint across the trajectory, use `vectorized=True`. This allows you to write constraints that operate on the entire trajectory arrays for `x` and `u`.

Example:

```python
@nodal(vectorized=True)
def g(x, u):
    # x and u are arrays of shape (N, n_x) and (N, n_u)
    return (x[1:, t_idx] - x[:-1, t_idx]) - t_max
```

This replaces the deprecated `internodal` argument.

## Constraints Parameters

To allow for a large number of problems to be effectively solved with the same code, we have a number of parameters that can be set to modify the behavior of the constraints.

### Constraint Time Constraints

For an inequality constraint using the definition ($g(\cdot) \leq 0$) that you wish to be enforced in continuous time, the `@ctcs()` decorator can be applied. For example if I have a ball constraint, $\|x-r_c\|_2\leq r_{\min}$, meaning I want my agent to stay at least $r_{\min}$ distance to a point $r_c$ for the entire trajectory, I can do the following:

```python

@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min)

```
!!! Note
    When specifying `ctcs` constraints, all constraints regardless of convexity are treated to the sam reformulation technique and linearized. This is not the case for `nodal` constraints.

#### Penalty Function

In order to enforce this constraint a penalty function is used. THis can be set using the `penalty` flag, which will defualt to `"squared_relu"`, $\max(0, g(x))^2$. However this may not be the best choice depending on the nature of your constraints. We provide a few different options, 

  - `squared_relu` - $\max(0, g)^2$
  - `huber` - $\begin{cases} \frac{1}{2} g^2 & \text{if } |g| \leq \delta \\ \delta (|g| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$
  - `smooth relu` - $\|\max(0, g)+c\|-c$

These can be set using the `penalty` flag. For example, if I want to use the huber penalty function, I can do the following:

```python

@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min, penalty="huber")

```

If you want to set your own penalty function, then you can provide a lambda function as follows 

```python
@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min, penalty=lambda x:  jnp.maximum(0, x)**2)
```

#### Node Specific Regions

If you onyl want to enforce the constraint in continuous time but only over a portion of the full trajectory, you can specify a 'nodes' argument as follows,

```python
@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min, nodes=(3, 8))
```

This will ensure that the constraint is only enforced between nodes 3 and 8. 

!!! tip
    Specifying this may be useful if, for example, you want the agent to occupy certain regions only at certain times.

By defualt, `ctcs` constraits will be assumed to be enforced over the entire trajectory. 


#### Multiple Augmented States

Constraints that are enforced over difference ranges will be associated with diffferent augmentes states, $y$. If you want to have seperate augmented states for each constraint, you can specify an `idx` argument as follows,

```python
@ctcs(lambda x, u: jnp.linalg.norm(x[r_idx]-r_c) - r_min, idx=0)
@ctcs(lambda x, u: x - max_state, idx = 1)
```
In this case, two augmented states will be used to enforce each constraint. 


### Nodal Constraints
You can additionally express constraints for discrete points along the trajectory. For exmaple if I wish to enforce that at each node, the agents velocity is less than a certain value, $v_{\max}$, I can do so as follows,

```python

@nodal(lambda x, u: cp.norm(x[v_idx]) <= v_max, convex=True)
```

#### Convexity
When specifying constraints if the constraint is convex, please set `convex = True` and use [```cvxpy``` atoms](https://www.cvxpy.org/tutorial/functions/index.html) to define the constraint as a boolean expression. By default `