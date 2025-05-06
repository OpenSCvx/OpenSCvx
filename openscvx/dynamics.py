import jax
import jax.numpy as jnp


def get_augmented_dynamics(dynamics: callable, g_func: dict[str, callable]):
    def dynamics_augmented(x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) only pass user-defined portion of x to dynamics in case user has `-1` or similar indexing in function
        x_dot = dynamics(x, u)
        # y_dot = g_func(x, u)
        # return jnp.hstack([x_dot, y_dot])

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for key, g in g_func.items():
            x_dot = jnp.hstack([x_dot, g(x, u)])
        
        return x_dot

    return dynamics_augmented


def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B