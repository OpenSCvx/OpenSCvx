import jax.numpy as jnp
import numpy as np

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
) -> jnp.ndarray:
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
    f = state_dot(x, u[:, :-1], nodes)
    F = s[:, None] * f

    # Evaluate the State Jacobian
    dfdx = A(x, u[:, :-1], nodes)
    sdfdx = s[:, None, None] * dfdx

    # Evaluate the Control Jacobian
    dfdu_veh = B(x, u[:, :-1], nodes)
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
    return dVdt.flatten()


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
):

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # initial augmented state
    V0 = jnp.zeros((N - 1, i5))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    V0 = V0.at[:, n_x : n_x + n_x * n_x].set(
        jnp.eye(n_x).reshape(1, -1).repeat(N - 1, axis=0)
    )

    # choose integrator
    if custom_integrator:
        # fmt: off
        sol = solve_ivp_rk45(
            lambda t,y,*a: dVdt(t, y, *a),
            1.0/(N-1),
            V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, n_x, n_u, N, dis_type),
            is_not_compiled=debug,
        )
        # fmt: on
    else:
        # fmt: off
        sol = solve_ivp_diffrax(
            lambda t,y,*a: dVdt(t, y, *a),
            1.0/(N-1),
            V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, n_x, n_u, N, dis_type),
            solver_name=solver,
            rtol=rtol,
            atol=atol,
            extra_kwargs=None,
        )
        # fmt: on

    Vend = sol[-1].T.reshape(-1, i5)
    Vmulti = sol.T

    # fmt: off
    A_bar = Vend[:, i1:i2].reshape(N-1, n_x, n_x).transpose(1,2,0).reshape(n_x*n_x, -1, order='F').T
    B_bar = Vend[:, i2:i3].reshape(N-1, n_x, n_u).transpose(1,2,0).reshape(n_x*n_u, -1, order='F').T
    C_bar = Vend[:, i3:i4].reshape(N-1, n_x, n_u).transpose(1,2,0).reshape(n_x*n_u, -1, order='F').T
    z_bar = Vend[:, i4:i5]
    # fmt: on

    return A_bar, B_bar, C_bar, z_bar, Vmulti


def get_discretization_solver(state_dot, A, B, params):
    return lambda x, u: calculate_discretization(
        x=x,
        u=u,
        state_dot=state_dot,
        A=A,
        B=B,
        n_x=params.sim.n_states,
        n_u=params.sim.n_controls,
        N=params.scp.n,
        custom_integrator=params.dis.custom_integrator,
        debug=params.dev.debug,
        solver=params.dis.solver,
        rtol=params.dis.rtol,
        atol=params.dis.atol,
        dis_type=params.dis.dis_type,
    )
