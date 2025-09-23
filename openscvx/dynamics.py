from dataclasses import dataclass
from typing import Callable, Optional, Union

import jax.numpy as jnp


@dataclass
class Dynamics:
    """
    Dataclass to hold a system dynamics function and (optionally) its gradients.
    This class is intended to be instantiated using the `dynamics` decorator
    wrapped around a function defining the system dynamics. Both the dynamics
    and optional gradients should be composed of `jax` primitives to enable
    efficient computation.

    Usage examples:

    ```python
    @dynamics
    def f(x_, u_):
        return x_ + u_
    # f is now a Dynamics object
    ```

    ```python
    @dynamics(A=grad_f_x, B=grad_f_u)
    def f(x_, u_):
        return x_ + u_
    ```

    Or, if a more lambda-function-style is desired, the function can be
    directly wrapped:

    ```python
    dyn = dynamics(lambda x_, u_: x_ + u_)
    ```

    ---
    **Using Parameters in Dynamics**

    You can use symbolic `Parameter` objects in your dynamics function to
    represent tunable or environment-dependent values. **The argument names
    for parameters must match the parameter name with an underscore suffix**
    (e.g., `I_sp_` for a parameter named `I_sp`). This is required for the
    parameter mapping to work correctly.

    Example (3DoF rocket landing):

    ```python
    from openscvx.backend.parameter import Parameter
    import jax.numpy as jnp

    I_sp = Parameter("I_sp")
    g = Parameter("g")
    theta = Parameter("theta")

    @dynamics
    def rocket_dynamics(x_, u_, I_sp_, g_, theta_):
        m = x_[6]
        T = u_
        r_dot = x_[3:6]
        g_vec = jnp.array([0, 0, g_])
        v_dot = T/m - g_vec
        m_dot = -jnp.linalg.norm(T) / (I_sp_ * 9.807 * jnp.cos(theta_))
        t_dot = 1
        return jnp.hstack([r_dot, v_dot, m_dot, t_dot])

    # Set parameter values before solving
    I_sp.value = 225
    g.value = 3.7114
    theta.value = 27 * jnp.pi / 180
    ```

    ---
    **Using Parameters in Nodal Constraints**

    You can also use symbolic `Parameter` objects in nodal constraints. As
    with dynamics, the argument names for parameters in the constraint
    function must match the parameter name with an underscore suffix
    (e.g., `g_` for a parameter named `g`).

    Example:

    ```python
    from openscvx.backend.parameter import Parameter
    from openscvx.constraints import nodal
    import jax.numpy as jnp

    g = Parameter("g")
    g.value = 3.7114

    @nodal
    def terminal_velocity_constraint(x_, u_, g_):
        # Enforce a terminal velocity constraint using the gravity parameter
        return x_[5] + g_ * x_[7]  # e.g., vz + g * t <= 0 at final node
    ```

    When building your problem, collect all parameters with
    `Parameter.get_all()` and pass them to your problem setup.

    Args:
        f (Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
            Function defining the continuous time nonlinear system dynamics
            as x_dot = f(x, u, ...params).
            - x: 1D array (state at a single node), shape (n_x,)
            - u: 1D array (control at a single node), shape (n_u,)
            - Additional parameters: passed as keyword arguments with names
              matching the parameter name plus an underscore (e.g., g_ for
              Parameter('g')).
            If you want to use parameters, include them as extra arguments
            with the underscore naming convention.
            If you use vectorized integration or batch evaluation, x and u
            may be 2D arrays (N, n_x) and (N, n_u).
        A (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of `f` w.r.t. `x`. If not specified, will be calculated
            using `jax.jacfwd`.
        B (Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]):
            Jacobian of `f` w.r.t. `u`. If not specified, will be calculated
            using `jax.jacfwd`.

    Returns:
        Dynamics: A dataclass bundling the system dynamics function and
        Jacobians.
    """

    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    A: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    B: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
