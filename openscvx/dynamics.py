from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp


@dataclass
class Dynamics:
    """Dataclass to hold a system dynamics function and its Jacobians.

    This dataclass is used internally by openscvx to store the compiled dynamics
    function and its gradients after symbolic expressions are lowered to JAX.
    Users typically don't instantiate this class directly.

    Defining Dynamics with Symbolic Expressions
    --------------------------------------------
    Define dynamics as a dictionary mapping state names to symbolic expressions.
    This approach provides automatic differentiation, shape validation, and
    declarative problem specification.

    Example:
        3DoF rocket landing problem::

            import openscvx as ox
            import numpy as np

            # Define symbolic states
            position = ox.State("position", shape=(3,))
            velocity = ox.State("velocity", shape=(3,))
            mass = ox.State("mass", shape=(1,))

            # Define symbolic control
            thrust = ox.Control("thrust", shape=(3,))

            # Define parameters
            I_sp = ox.Parameter("I_sp", value=225.0)
            g = ox.Parameter("g", value=3.7114)
            theta = ox.Parameter("theta", value=27 * np.pi / 180)

            # Define dynamics using symbolic expressions
            g_vec = np.array([0, 0, 1]) * g
            dynamics = {
                "position": velocity,
                "velocity": thrust / mass[0] - g_vec,
                "mass": -ox.linalg.Norm(thrust) / (I_sp * 9.807 * ox.Cos(theta)),
            }

            # Create problem with symbolic dynamics
            problem = ox.TrajOptProblem(
                dynamics=dynamics,
                states=[position, velocity, mass],
                controls=[thrust],
                ...
            )

    Symbolic expressions support:
        - Arithmetic operations: ``+, -, *, /, @, **``
        - Comparison operations: ``==, <=, >=`` (for constraints)
        - Indexing and slicing: ``state[0]``, ``state[1:3]``
        - Functions: ``ox.Norm()``, ``ox.Sin()``, ``ox.Cos()``, etc.
        - Shape checking and validation at problem setup
        - Parameter updates without recompilation

    During problem initialization, the symbolic dynamics are automatically:
        1. Canonicalized (simplified algebraically)
        2. Shape-checked for correctness
        3. Lowered to JAX functions
        4. Differentiated to obtain Jacobians (A, B)
        5. Stored in this Dynamics dataclass

    Attributes:
        f (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
            Function defining the continuous time nonlinear system dynamics
            as x_dot = f(x, u, ...params).
            - x: 1D array (state at a single node), shape (n_x,)
            - u: 1D array (control at a single node), shape (n_u,)
            - Additional parameters: passed as keyword arguments with names
              matching the parameter name plus an underscore (e.g., g_ for
              Parameter('g')).
            If you use vectorized integration or batch evaluation, x and u
            may be 2D arrays (N, n_x) and (N, n_u).
        A (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of ``f`` w.r.t. ``x``. If not specified, will be calculated
            using ``jax.jacfwd``.
        B (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of ``f`` w.r.t. ``u``. If not specified, will be calculated
            using ``jax.jacfwd``.
    """

    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    A: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    B: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
