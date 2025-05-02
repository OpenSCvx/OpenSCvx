import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import diffrax  as dfx

from openscvx.config import Config
from openscvx.integrators import solve_ivp_rk45, solve_ivp_diffrax, solve_ivp_diffrax_prop



class RK45_Custom:
    def __init__(self, params):
        self.params = params

    def solve_ivp(self, dVdt, tau_grid, V0, args, method='RK45', t_eval=None):
        if method != 'RK45':
            raise ValueError("Currently, only 'RK45' method is supported.")
        
        return solve_ivp_rk45(dVdt, tau_grid[1], V0, args, is_not_compiled=self.params.dev.debug)

    
class Diffrax_Prop:
    def __init__(self, state_dot, A, B, params):
        self.params = params
        self.func = ExactDis(state_dot, A, B, params).prop_aug_dy
    
    def solve_ivp(self, V0, tau_grid, u_cur, u_next, tau_init, idx_s):
        return solve_ivp_diffrax_prop(self.func, tau_grid[1], V0, args=(u_cur, u_next, tau_init, idx_s), tau_0=tau_grid[0])

class Diffrax:
    def __init__(self, params):
        self.params = params
    
    def solve_ivp(self, dVdt, tau_grid, V0, args, t_eval=None):
        return solve_ivp_diffrax(dVdt, tau_grid[1], V0, args, solver_name=self.params.dis.solver, rtol=self.params.dis.rtol, atol=self.params.dis.atol, extra_kwargs=self.params.dis.args)

def s_to_t(u, params: Config):
    t = [0]
    tau = np.linspace(0, 1, params.scp.n)
    for k in range(1, params.scp.n):
        s_kp = u[k-1,-1]
        s_k = u[k,-1]
        if params.dis.dis_type == 'ZOH':
            t.append(t[k-1] + (tau[k] - tau[k-1])*(s_kp))
        else:
            t.append(t[k-1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k-1]))
    return t

def t_to_tau(u, t, u_nodal, t_nodal, params: Config):
    u_lam = lambda new_t: np.array([np.interp(new_t, t_nodal, u[:, i]) for i in range(u.shape[1])]).T
    u = np.array([u_lam(t_i) for t_i in t])

    tau = np.zeros(len(t))
    tau_nodal = np.linspace(0, 1, params.scp.n)
    for k in range(1, len(t)):
        k_nodal = np.where(t_nodal < t[k])[0][-1]
        s_kp = u_nodal[k_nodal, -1]
        tp = t_nodal[k_nodal]
        tau_p = tau_nodal[k_nodal]

        s_k = u[k, -1]
        if params.dis.dis_type == 'ZOH':
            tau[k] = tau_p + (t[k] - tp) / s_kp
        else:
            tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
    return tau, u

class ExactDis:
    def __init__(self, state_dot, A, B, params: Config) -> None:
        self.state_dot = state_dot
        self.A = A
        self.B = B
        self.params = params

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Define indices for slicing the augmented state vector
        self.i0 = 0
        self.i1 = n_x
        self.i2 = self.i1 + n_x * n_x
        self.i3 = self.i2 + n_x * n_u
        self.i4 = self.i3 + n_x * n_u
        self.i5 = self.i4 + n_x

        
        if self.params.dis.custom_integrator:
            self.integrator = RK45_Custom(self.params)
        else:
            self.integrator = Diffrax(self.params)

        self.tau_grid = jnp.linspace(0, 1, self.params.scp.n)

        self.calculate_discretization = get_discretization_solver(state_dot, A, B, params)
    
    def prop_aug_dy(self,
                    tau: float,
                    x: np.ndarray,
                    u_current: np.ndarray,
                    u_next: np.ndarray,
                    tau_init: float,
                    idx_s: int) -> np.ndarray:
        x = x[None, :]
        
        if self.params.dis.dis_type == "ZOH":
            beta = 0.0
        elif self.params.dis.dis_type == "FOH":
            beta = (tau - tau_init) * self.params.scp.n
        u = u_current + beta * (u_next - u_current)
        
        return  u[:, idx_s] * self.state_dot(x, u[:,:-1]).squeeze()

    def simulate_nonlinear_time(self, x_0, u, tau_vals, t):
        params = self.params
        states = np.empty((x_0.shape[0], 0))  # Initialize states as a 2D array with shape (n, 0)

        tau = np.linspace(0, 1, params.scp.n)

        u_lam = lambda new_t: np.array([np.interp(new_t, t, u[:, i]) for i in range(u.shape[1])]).T

        # Bin the tau_vals into with respect to the uniform tau grid, tau
        tau_inds = np.digitize(tau_vals, tau) - 1
        # Force the last indice to be in the same bin as the previous ones
        tau_inds = np.where(tau_inds == params.scp.n - 1, params.scp.n - 2, tau_inds)

        prev_count = 0

        for k in range(params.scp.n - 1):
            controls_current = np.squeeze(u_lam(t[k]))[None, :]
            controls_next = np.squeeze(u_lam(t[k + 1]))[None, :]

            # Create a mask
            mask = (tau_inds >= k) & (tau_inds < k + 1)

            count = np.sum(mask)

            # Use count to grab the first count number of elements
            tau_cur = tau_vals[prev_count:prev_count + count]

            sol = self.params.prp.integrator(x_0, (tau[k], tau[k + 1]), controls_current, controls_next, np.array([[tau[k]]]), params.sim.idx_s.stop)

            x = sol.ys
            for tau_i in tau_cur:
                new_state = sol.evaluate(tau_i).reshape(-1, 1)  # Ensure new_state is 2D
                states = np.concatenate([states, new_state], axis=1)

            x_0 = x[-1]
            prev_count += count

        return states.T


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
    if dis_type == 'ZOH':
        beta = 0.
    elif dis_type == 'FOH':
        beta = (tau) * N
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)
    s = u[:,-1]

    # Initialize the augmented Jacobians
    dfdx = jnp.zeros((V.shape[0], n_x, n_x))
    dfdu = jnp.zeros((V.shape[0], n_x, n_u))

    # Ensure x_seq and u have the same batch size
    x = V[:,:n_x]
    u = u[:x.shape[0]]

    # Compute the nonlinear propagation term
    f = state_dot(x, u[:,:-1])
    F = s[:, None] * f

    # Evaluate the State Jacobian
    dfdx = A(x, u[:,:-1])
    sdfdx = s[:, None, None] * dfdx

    # Evaluate the Control Jacobian
    dfdu_veh = B(x, u[:,:-1])
    dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
    dfdu = dfdu.at[:, :, -1].set(f)
    
    # Compute the defect
    z = F - jnp.einsum('ijk,ik->ij', sdfdx, x) - jnp.einsum('ijk,ik->ij', dfdu, u)

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
    dis_type: str
):

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # initial augmented state
    V0 = jnp.zeros((N-1, i5))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    V0 = V0.at[:, n_x:n_x+n_x*n_x].set(
        jnp.eye(n_x).reshape(1,-1).repeat(N-1, axis=0)
    )

    # choose integrator
    if custom_integrator:
        sol = solve_ivp_rk45(
            lambda t,y,*a: dVdt(t, y, *a),
            1.0/N,
            V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, n_x, n_u, N, dis_type),
            is_not_compiled=debug,
        )
    else:
        sol = solve_ivp_diffrax(
            lambda t,y,*a: dVdt(t, y, *a),
            1.0/N,
            V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, n_x, n_u, N, dis_type),
            solver_name=solver,
            rtol=rtol,
            atol=atol,
            extra_kwargs=None,
        )

    Vend   = sol[-1].T.reshape(-1, i5)
    Vmulti = sol.T

    A_bar = Vend[:, i1:i2].reshape(N-1, n_x, n_x).transpose(1,2,0).reshape(n_x*n_x, -1, order='F').T
    B_bar = Vend[:, i2:i3].reshape(N-1, n_x, n_u).transpose(1,2,0).reshape(n_x*n_u, -1, order='F').T
    C_bar = Vend[:, i3:i4].reshape(N-1, n_x, n_u).transpose(1,2,0).reshape(n_x*n_u, -1, order='F').T
    z_bar = Vend[:, i4:i5]

    return A_bar, B_bar, C_bar, z_bar, Vmulti


def get_discretization_solver(state_dot, A, B, params):
    return lambda x, u: calculate_discretization(x, u, state_dot, A, B, params.sim.n_states, params.sim.n_controls, params.scp.n, params.dis.custom_integrator, params.dev.debug, params.dis.solver, params.dis.rtol, params.dis.atol, params.dis.dis_type)

