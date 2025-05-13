from collections import defaultdict
from typing import List

import jax.numpy as jnp

from openscvx.constraints.ctcs import CTCSConstraint

def get_g_grad_x(constraints_ctcs: List[CTCSConstraint]):
    def g_grad_x(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        grads = [c.grad_f_x(x, u, node) for c in constraints_ctcs]
        if not grads:
            return None
        return sum(grads)
    return g_grad_x

def get_g_grad_u(constraints_ctcs: List[CTCSConstraint]):
    def g_grad_u(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        grads = [c.grad_f_u(x, u, node) for c in constraints_ctcs]
        if not grads:
            return None
        return sum(grads)
    return g_grad_u

def get_g_func(constraints_ctcs: List[CTCSConstraint]):
    def g_func(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        g_sum = 0
        for constraint in constraints_ctcs:
            g_sum += constraint(x,u, node)
        return g_sum
    return g_func


def get_g_funcs(constraints_ctcs: List[CTCSConstraint]) -> list[callable]:
    # Bucket by idx
    groups: dict[int, list[callable]] = defaultdict(list)
    for c in constraints_ctcs:
        if c.idx is None:
            raise ValueError(f"CTCS constraint {c} has no .idx assigned")
        groups[c.idx].append(c)

    # Build and return a list of get_g_func(funcs) in idx order
    g_funcs = [
        get_g_func(funcs)
        for idx, funcs in sorted(groups.items(), key=lambda kv: kv[0])
    ]

    g_grads_x = [
        get_g_grad_x(funcs)
        for idx, funcs in sorted(groups.items(), key=lambda kv: kv[0])
    ]

    g_grads_u = [
        get_g_grad_u(funcs)
        for idx, funcs in sorted(groups.items(), key=lambda kv: kv[0])
    ]

    return g_funcs, g_grads_x, g_grads_u
