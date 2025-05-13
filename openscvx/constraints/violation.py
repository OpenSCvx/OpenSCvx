from collections import defaultdict
from typing import List, Optional, Tuple, Callable

import jax.numpy as jnp

from openscvx.constraints.ctcs import CTCSConstraint


def get_g_grad_x(constraints_ctcs: List[CTCSConstraint]):
    def g_grad_x(x: jnp.ndarray, u: jnp.ndarray, node: int) -> jnp.ndarray:
        # only call those c.grad_f_x that are not None
        grads = [
            c.grad_f_x(x, u, node) for c in constraints_ctcs if c.grad_f_x is not None
        ]
        if not grads:
            return None
        return sum(grads)

    return g_grad_x


def get_g_grad_u(constraints_ctcs: List[CTCSConstraint]):
    def g_grad_u(x: jnp.ndarray, u: jnp.ndarray, node: int) -> jnp.ndarray:
        grads = [
            c.grad_f_u(x, u, node) for c in constraints_ctcs if c.grad_f_u is not None
        ]
        if not grads:
            return None
        return sum(grads)

    return g_grad_u


def get_g_func(constraints_ctcs: List[CTCSConstraint]):
    def g_func(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        return sum(c(x, u, node) for c in constraints_ctcs)

    return g_func


def get_g_funcs(constraints_ctcs: List[CTCSConstraint]) -> list[callable]:
    # Bucket by idx
    groups: dict[int, List[callable]] = defaultdict(list)
    for c in constraints_ctcs:
        if c.idx is None:
            raise ValueError(f"CTCS constraint {c} has no .idx assigned")
        groups[c.idx].append(c)

    g_funcs = []
    g_grads_x = []
    g_grads_u = []

    # For each bucket, build func + maybe-grad
    for idx, bucket in sorted(groups.items(), key=lambda kv: kv[0]):
        g_funcs.append(get_g_func(bucket))

        # only produce a grad-function if *all* c in bucket had one
        if all(c.grad_f_x for c in bucket):
            g_grads_x.append(get_g_grad_x(bucket))
        else:
            g_grads_x.append(None)

        if all(c.grad_f_u for c in bucket):
            g_grads_u.append(get_g_grad_u(bucket))
        else:
            g_grads_u.append(None)

    return g_funcs, g_grads_x, g_grads_u
