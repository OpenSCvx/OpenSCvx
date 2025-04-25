import jax.numpy as jnp

def get_g_func(constraints_ctcs: list[callable]):
    def g_func(x: jnp.array, u: jnp.array) -> jnp.array:
        g_sum = 0
        for g in constraints_ctcs:
            g_sum += g(x, u)
        return g_sum
    return g_func