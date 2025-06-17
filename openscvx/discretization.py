import jax.numpy as jnp
import numpy as np

from openscvx.dynamics import Dynamics
from openscvx.integrators import solve_ivp_rk45, solve_ivp_diffrax


def dVdt(
    tau: float,
    V: jnp.ndarray,
    u_cur: np.ndarray,
    u_next: np.ndarray,
    state_dot: callable,
    A: callable,
    B: callable,
    n_x: int,
    n_u: int,
    N: int,
    dis_type: str,
    **params
) -> jnp.ndarray:
    """Compute the time derivative of the augmented state vector.
    
    This function computes the time derivative of the augmented state vector V,
    which includes the state, state transition matrix, and control influence matrix.
    
    Args:
        tau (float): Current normalized time in [0,1].
        V (jnp.ndarray): Augmented state vector.
        u_cur (np.ndarray): Control input at current node.
        u_next (np.ndarray): Control input at next node.
        state_dot (callable): Function computing state derivatives.
        A (callable): Function computing state Jacobian.
        B (callable): Function computing control Jacobian.
        n_x (int): Number of states.
        n_u (int): Number of controls.
        N (int): Number of nodes in trajectory.
        dis_type (str): Discretization type ("ZOH" or "FOH").
        **params: Additional parameters passed to state_dot, A, and B.
        
    Returns:
        jnp.ndarray: Time derivative of augmented state vector.
    """
    # Define the nodes
    nodes = jnp.arange(0, N-1)

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Unflatten V
    V = V.reshape(-1, i5)

    # Compute the interpolation factor based on the discretization type
    if dis_type == "ZOH":
        beta = 0.0
    elif dis_type == "FOH":
        beta = (tau) * N
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)
    s = u[:, -1]

    # Initialize the augmented Jacobians
    dfdx = jnp.zeros((V.shape[0], n_x, n_x))
    dfdu = jnp.zeros((V.shape[0], n_x, n_u))

    # Ensure x_seq and u have the same batch size
    x = V[:, :n_x]
    u = u[: x.shape[0]]

    # Compute the nonlinear propagation term
    f = state_dot(x, u[:, :-1], nodes, *params.items())
    F = s[:, None] * f

    # Evaluate the State Jacobian
    dfdx = A(x, u[:, :-1], nodes, *params.items())
    sdfdx = s[:, None, None] * dfdx

    # Evaluate the Control Jacobian
    dfdu_veh = B(x, u[:, :-1], nodes, *params.items())
    dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
    dfdu = dfdu.at[:, :, -1].set(f)

    # Compute the defect
    z = F - jnp.einsum("ijk,ik->ij", sdfdx, x) - jnp.einsum("ijk,ik->ij", dfdu, u)

    # Stack up the results into the augmented state vector
    # fmt: off
    dVdt = jnp.zeros_like(V)
    dVdt = dVdt.at[:, i0:i1].set(F)
    dVdt = dVdt.at[:, i1:i2].set(jnp.matmul(sdfdx, V[:, i1:i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x))
    dVdt = dVdt.at[:, i2:i3].set((jnp.matmul(sdfdx, V[:, i2:i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(-1, n_x * n_u))
    dVdt = dVdt.at[:, i3:i4].set((jnp.matmul(sdfdx, V[:, i3:i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(-1, n_x * n_u))
    dVdt = dVdt.at[:, i4:i5].set((jnp.matmul(sdfdx, V[:, i4:i5].reshape(-1, n_x)[..., None]).squeeze(-1) + z).reshape(-1, n_x))
    # fmt: on

    return dVdt.reshape(-1)


def calculate_discretization(
    x,
    u,
    state_dot: callable,
    A: callable,
    B: callable,
    n_x: int,
    n_u: int,
    N: int,
    custom_integrator: bool,
    debug: bool,
    solver: str,
    rtol,
    atol,
    dis_type: str,
    **kwargs
):
    """Calculate the discretized system matrices.
    
    This function computes the discretized system matrices (A_bar, B_bar, C_bar)
    and defect vector (z_bar) using numerical integration.
    
    Args:
        x: State trajectory.
        u: Control trajectory.
        state_dot (callable): Function computing state derivatives.
        A (callable): Function computing state Jacobian.
        B (callable): Function computing control Jacobian.
        n_x (int): Number of states.
        n_u (int): Number of controls.
        N (int): Number of nodes in trajectory.
        custom_integrator (bool): Whether to use custom RK45 integrator.
        debug (bool): Whether to use debug mode.
        solver (str): Name of the solver to use.
        rtol (float): Relative tolerance for integration.
        atol (float): Absolute tolerance for integration.
        dis_type (str): Discretization type ("ZOH" or "FOH").
        **kwargs: Additional parameters passed to state_dot, A, and B.
        
    Returns:
        tuple: (A_bar, B_bar, C_bar, z_bar, Vmulti) where:
            - A_bar: Discretized state transition matrix
            - B_bar: Discretized control influence matrix
            - C_bar: Discretized control influence matrix for next node
            - z_bar: Defect vector
            - Vmulti: Full augmented state trajectory
    """
    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Initial augmented state
    V0 = jnp.zeros((N - 1, i5))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    V0 = V0.at[:, n_x:n_x + n_x * n_x].set(
        jnp.eye(n_x).reshape(1, -1).repeat(N - 1, axis=0)
    )

    # Choose integrator
    integrator_args = dict(
        u_cur=u[:-1].astype(float),
        u_next=u[1:].astype(float),
        state_dot=state_dot,
        A=A,
        B=B,
        n_x=n_x,
        n_u=n_u,
        N=N,
        dis_type=dis_type,
        **kwargs  # <-- adds parameter values with names
    )

    # Define dVdt wrapper using named arguments
    def dVdt_wrapped(t, y):
        return dVdt(t, y, **integrator_args)

    # Choose integrator
    if custom_integrator:
        sol = solve_ivp_rk45(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0.reshape(-1),
            args=(),
            is_not_compiled=debug,
        )
    else:
        sol = solve_ivp_diffrax(
            dVdt_wrapped,
            1.0 / (N - 1),
            V0.reshape(-1),
            solver_name=solver,
            rtol=rtol,
            atol=atol,
            args=(),
            extra_kwargs=None,
        )

    Vend = sol[-1].T.reshape(-1, i5)
    Vmulti = sol.T

    A_bar = Vend[:, i1:i2].reshape(N - 1, n_x, n_x).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F').T
    B_bar = Vend[:, i2:i3].reshape(N - 1, n_x, n_u).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
    C_bar = Vend[:, i3:i4].reshape(N - 1, n_x, n_u).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
    z_bar = Vend[:, i4:i5]

    return A_bar, B_bar, C_bar, z_bar, Vmulti


def get_discretization_solver(dyn: Dynamics, settings, param_map):
    """Create a discretization solver function.
    
    This function creates a solver that computes the discretized system matrices
    using the specified dynamics and settings.
    
    Args:
        dyn (Dynamics): System dynamics object.
        settings: Configuration settings for discretization.
        param_map (dict): Mapping of parameter names to values.
        
    Returns:
        callable: A function that computes the discretized system matrices.
    """
    return lambda x, u, *params: calculate_discretization(
        x=x,
        u=u,
        state_dot=dyn.f,
        A=dyn.A,
        B=dyn.B,
        n_x=settings.sim.n_states,
        n_u=settings.sim.n_controls,
        N=settings.scp.n,
        custom_integrator=settings.dis.custom_integrator,
        debug=settings.dev.debug,
        solver=settings.dis.solver,
        rtol=settings.dis.rtol,
        atol=settings.dis.atol,
        dis_type=settings.dis.dis_type,
        **dict(zip(param_map.keys(), params))  # <--- Named keyword args
    )