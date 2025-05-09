import jax
import jax.numpy as jnp


def get_augmented_dynamics(
    dynamics: callable, g_funcs: list[callable], idx_x_true: slice, idx_u_true: slice
) -> callable:
    def dynamics_augmented(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        x_dot = dynamics(x[idx_x_true], u[idx_u_true])

        # Iterate through the g_func dictionary and stack the output each function
        # to x_dot
        for g in g_funcs:
            x_dot = jnp.hstack([x_dot, g(x[idx_x_true], u[idx_u_true], node)])

        return x_dot

    return dynamics_augmented


def get_jacobians(dyn: callable):
    A = jax.jacfwd(dyn, argnums=0)
    B = jax.jacfwd(dyn, argnums=1)
    return A, B
