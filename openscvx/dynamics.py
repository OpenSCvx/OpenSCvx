from dataclasses import dataclass
from typing import Callable, Optional
import functools

import jax.numpy as jnp


@dataclass
class Dynamics:
    """
    Dataclass to hold dynamics function and (optionally) it's gradients.
    This class is intended to be instantiated using the `dynamics` decorator wrapped around a function defining the system dynamics.
    Note: the dynamics as well as the optional gradients should be composed of `jax` primitives to enable efficient computation.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): function defining the continuous time nonlinear system dynamics as x_dot = f(x, u)
        A (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]): Jacobian of `f` w.r.t. `x`. If not specified will be calculated using `jax.jacfwd`
        B (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]): Jacobian of `f` w.r.t. `u`. If not specified will be calculated using `jax.jacfwd`
    """
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

