from collections import defaultdict

import jax.numpy as jnp

def get_g_func(constraints_ctcs: list[callable, callable]):
    def g_func(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        g_sum = 0
        for g in constraints_ctcs:
            g_sum += g(x,u, node)
        return g_sum
    return g_func


def get_g_funcs(constraints_ctcs: list[callable]) -> list[callable]:
    # Bucket by idx
    groups: dict[int, list[callable]] = defaultdict(list)
    for c in constraints_ctcs:
        if c.idx is None:
            raise ValueError(f"CTCS constraint {c} has no .idx assigned")
        groups[c.idx].append(c)

    # Build and return a list of get_g_func(funcs) in idx order
    return [
        get_g_func(funcs)
        for idx, funcs in sorted(groups.items(), key=lambda kv: kv[0])
    ]
