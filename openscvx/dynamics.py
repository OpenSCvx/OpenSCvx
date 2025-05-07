import jax
import jax.numpy as jnp


def get_augmented_dynamics(dynamics: callable, g_funcs: list[callable]):
    def dynamics_augmented(x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) only pass user-defined portion of x to dynamics in case user has `-1` or similar indexing in function
        x_dot = dynamics(x, u)

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for g in g_funcs:
            # TODO: (norrisg) don't do hacky -1 indexing!!!
            x_dot = jnp.hstack([x_dot, g(x[:-1], u[:-1])])
        
        return x_dot

    return dynamics_augmented


def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B