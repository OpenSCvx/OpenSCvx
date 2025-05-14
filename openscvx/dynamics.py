from dataclasses import dataclass
from typing import Callable, Optional
import functools

import jax.numpy as jnp


@dataclass
class Dynamics:
    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    A: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    B: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

def dynamics(
    _func=None,
    *,
    A: Optional[Callable] = None,
    B: Optional[Callable] = None,):
    """Decorator to mark a function as defining the system dynamics.

    Use as:
    @dynamics(A=my_grad_f_x, B=my_grad_f_u)')
    def my_dynamics(x,u): ...
    """

    def decorator(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        # wrap so name, doc, signature stay on f
        wrapped = functools.wraps(f)(f)
        return Dynamics(
            f=wrapped,
            A=A,
            B=B,
        )

    # if called as @dynamics or @dynamics(...), _func will be None and we return decorator
    if _func is None:
        return decorator
    # if called as dynamics(func), we immediately decorate
    else:
        return decorator(_func)

