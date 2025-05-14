from collections import defaultdict
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

import jax.numpy as jnp

from openscvx.constraints.ctcs import CTCSConstraint


@dataclass
class CTCSViolation:
    g: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]
    g_grad_x: Optional[Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]] = None
    g_grad_u: Optional[Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]] = None


def get_g_grad_x(constraints_ctcs: List[CTCSConstraint]) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    def g_grad_x(x: jnp.ndarray, u: jnp.ndarray, node: int) -> jnp.ndarray:
        grads = [
            c.grad_f_x(x, u, node) for c in constraints_ctcs if c.grad_f_x is not None
        ]
        return sum(grads) if grads else None

    return g_grad_x


def get_g_grad_u(constraints_ctcs: List[CTCSConstraint]) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    def g_grad_u(x: jnp.ndarray, u: jnp.ndarray, node: int) -> jnp.ndarray:
        grads = [
            c.grad_f_u(x, u, node) for c in constraints_ctcs if c.grad_f_u is not None
        ]
        return sum(grads) if grads else None

    return g_grad_u


def get_g_func(constraints_ctcs: List[CTCSConstraint]) -> Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]:
    def g_func(x: jnp.array, u: jnp.array, node: int) -> jnp.array:
        return sum(c(x, u, node) for c in constraints_ctcs)

    return g_func


def get_g_funcs(constraints_ctcs: List[CTCSConstraint]) -> List[CTCSViolation]:
    # Bucket by idx
    groups: dict[int, List[CTCSConstraint]] = defaultdict(list)
    for c in constraints_ctcs:
        if c.idx is None:
            raise ValueError(f"CTCS constraint {c} has no .idx assigned")
        groups[c.idx].append(c)

    # For each bucket, build one CTCSViolation
    violations: List[CTCSViolation] = []
    for idx, bucket in sorted(groups.items(), key=lambda kv: kv[0]):
        g = get_g_func(bucket)
        g_grad_x = get_g_grad_u(bucket) if all(c.grad_f_x for c in bucket) else None
        g_grad_u = get_g_grad_x(bucket) if all(c.grad_f_u for c in bucket) else None

        violations.append(
            CTCSViolation(
                g=g,
                g_grad_x=g_grad_x,
                g_grad_u=g_grad_u,
            )
        )

    return violations
