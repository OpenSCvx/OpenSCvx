# Advanced Problem Setup

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
  - `huber` - $\frac{1}{2} \left( \sqrt{g^2 + \delta^2} - \delta \right)$
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
When specifying constraints if the constraint is convex, please set `convex = True` and use [```cvxpy``` atoms](https://www.cvxpy.org/tutorial/functions/index.html) to define the constraint as a boolean expression. By default `convex = False`. For example, 

```python   

@nodal(lambda x, u: cp.norm(x[v_idx]) <= v_max, convex=True)

```

!!! Tip
    The default solver `QOCO` can handle up to second-order conic constraints. If you wish to express exponential, power cones, or semi-definite programs, a good choice of solver would be `CLARABEL`. For mixed-interger programs, `Gurobi` is a good choice.

If the constraint is nonconvex, either set ```convex = False``` or leave it out as it defualts to ```False```. This will linearize the constraint and apply it at the nodes specified in the decorator. Nonconvex nodal constraints should be expressed as inequalities of the form $g(x,u) <= 0$ For example if I wish the agent to avoid a spherical obstacle of radius $r_{\min}$ at a point $r_c$ at all nodes,

```python
@nodal(lambda x, u: r_min - jnp.linalg.norm(x[r_idx] - r_c), convex=False)
```

#### Specific Node Ranges
Similar to the `ctcs` decorator, you can specify a range of nodes over which the constraint is enforced. By default this constraint will be applied at all nodes. To change this, you can add ```nodes = ["Insert a list of the desired nodes here"]``` to the decorator. For example if I wish for the agent to pass through a certain state, $x_\mathrm{wp}$, at the fourth node, this can be enforced as,

```python
@nodal(lambda x, u: x[r_idx] == r_wp, convex=True, nodes=[4])
```

