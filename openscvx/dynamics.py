import jax
import jax.numpy as jnp


def get_augmented_dynamics(dynamics: callable, g_func: callable):
    def dynamics_augmented(x: jnp.array, u: jnp.array) -> jnp.array:
        # TODO: (norrisg) handle varying lengths of x and u due to augmentation more elegantly
        x_dot = dynamics(x[:-1], u)
        y_dot = g_func(x, u)
        return jnp.hstack([x_dot, y_dot])

    return dynamics_augmented


def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B
