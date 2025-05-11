# Advanced Problem Setup

## Constraints Parameters

To allow for a large number of problems to be effectively solved with the same code, we have a number of parameters that can be set to modify the behavior of the constraints.

### Constraint Time Constraints

For a inequality constraint using the definition ($g(\cdot) \leq 0$) that you wish to be enforced in continuous time, the `@ctcs()` decorator can be applied. For example if I have a ball constraint, $\|x-r_c\|_2\leq r_{\min}$, meaning I want my agent to stay at least $r_{\min}$ distance to a point $r$ for the entire trajectory, I can do the following:

```python

@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min)

```
In order to enforce this constraint a penalty function is used. THis can be set using the `penalty` flag, which will defualt to `"squared_relu"`, $\max(0, g(x))^2$. However this may not be the best choice depending on the nature of your constraints. We provide a few different options, 

The type of penalty function to use. Options are:
  - `squared_relu` - $\max(0, g)^2$
  - `huber` - $\frac{1}{2} \left( \sqrt{g^2 + \delta^2} - \delta \right)$
  - `smooth relu` - $\|\max(0, g)+c\|-c$

These can be set using the `penalty` flag. For example, if I want to use the huber penalty function, I can do the following:

```python

@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min, penalty="huber")

```

If you want to set your own penalty function, then you can provide a lamda function as follows 

```python

@ctcs(lambda x, u: jnp.linalg.norm(x-r_c) - r_min, penalty=lambda x:  jnp.maximum(0, x)**2)
```
