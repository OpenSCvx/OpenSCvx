import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import diffrax  as dfx

from openscvx.config import Config
from openscvx.integrators import solve_ivp_rk45, solve_ivp_diffrax, SOLVER_MAP



class RK45_Custom:
    def __init__(self, params):
        self.params = params

    def solve_ivp(self, dVdt, tau_grid, V0, args, method='RK45', t_eval=None):
        if method != 'RK45':
            raise ValueError("Currently, only 'RK45' method is supported.")
        
        return solve_ivp_rk45(dVdt, tau_grid, V0, args, self.params.dev.debug, t_eval)

    
class Diffrax_Prop:
    def __init__(self, state_dot, A, B, params):
        self.params = params
        self.func = ExactDis(state_dot, A, B, params).prop_aug_dy
    
    def solve_ivp(self, V0, tau_grid, u_cur, u_next, tau_init, idx_s):
        t_eval = jnp.linspace(tau_grid[0], tau_grid[1], 50)

        solver_class = SOLVER_MAP.get(self.params.prp.solver)
        if solver_class is None:
            raise ValueError(f"Unknown solver: {self.params.prp.solver}")
        solver = solver_class()

        args = (u_cur, u_next, tau_init, idx_s)

        term = dfx.ODETerm(lambda t, y, args: self.func(t, y, *args))
        stepsize_controller = dfx.PIDController(rtol=self.params.prp.rtol, atol=self.params.prp.atol)
        solution = dfx.diffeqsolve(
            term,
            solver = solver,
            t0=tau_grid[0],
            t1=tau_grid[1],
            dt0=(tau_grid[1] - tau_grid[0]) / (len(t_eval) - 1),
            y0=V0,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=dfx.SaveAt(dense=True, ts=t_eval),
            **self.params.prp.args
        )

        return solution

class Diffrax:
    def __init__(self, params):
        self.params = params
    
    def solve_ivp(self, dVdt, tau_grid, V0, args, t_eval=None):
        return solve_ivp_diffrax(dVdt, tau_grid, V0, args, t_eval=t_eval, solver_name=self.params.dis.solver, rtol=self.params.dis.rtol, atol=self.params.dis.atol, extra_kwargs=self.params.dis.args)


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

    def s_to_t(self, u, params: Config):
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
    
    def t_to_tau(self, u, t, u_nodal, t_nodal, params: Config):
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
    
    def calculate_discretization(self,
                                 x: jnp.ndarray,
                                 u: jnp.ndarray):
        """
        Calculate discretization for given states, inputs and total time.
        x: Matrix of states for all time points
        u: Matrix of inputs for all time points
        return: A_k, B_k, C_k, z_k
        """

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls
        
        # Initialize the augmented state vector
        V0 = jnp.zeros((x.shape[0]-1, self.i5))

        # Vectorized integration
        V0 = V0.at[:, self.i0:self.i1].set(x[:-1, :].astype(float))
        V0 = V0.at[:, self.i1:self.i2].set(np.eye(n_x).reshape(1, n_x * n_x).repeat(self.params.scp.n - 1, axis=0))
        
        int_result = self.integrator.solve_ivp(self.dVdt, (self.tau_grid[0], self.tau_grid[1]), V0.flatten(), args=(u[:-1, :].astype(float), u[1:, :].astype(float)), t_eval=self.tau_grid)
        

        V = int_result[-1].T.reshape(-1, self.i5)
        V_multi_shoot = int_result.T
    
        # Flatten matrices in column-major (Fortran) order for cvxpy
        A_bar = V[:, self.i1:self.i2].reshape((self.params.scp.n - 1, n_x, n_x)).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F').T
        B_bar = V[:, self.i2:self.i3].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        C_bar = V[:, self.i3:self.i4].reshape((self.params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F').T
        z_bar = V[:, self.i4:self.i5]
        return A_bar, B_bar, C_bar, z_bar, V_multi_shoot

    def dVdt(self,
             tau: float,
             V: jnp.ndarray,
             u_cur: np.ndarray,
             u_next: np.ndarray
             ) -> jnp.ndarray:
        """
        Computes the time derivative of the augmented state vector for the system for a sequence of states.

        Parameters:
        tau (float): Current time.
        V (np.ndarray): Sequence of augmented state vectors.
        u_cur (np.ndarray): Sequence of current control inputs.
        u_next (np.ndarray): Sequence of next control inputs.
        A: Function that computes the Jacobian of the system dynamics with respect to the state.
        B: Function that computes the Jacobian of the system dynamics with respect to the control input.
        obstacles: List of obstacles in the environment.
        params (dict): Parameters of the system.

        Returns:
        np.ndarray: Time derivatives of the augmented state vectors.
        """
        
        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Unflatten V
        V = V.reshape(-1, self.i5)

        # Compute the interpolation factor based on the discretization type
        if self.params.dis.dis_type == 'ZOH':
            beta = 0.
        elif self.params.dis.dis_type == 'FOH':
            beta = (tau) * self.params.scp.n
        alpha = 1 - beta

        # Interpolate the control input
        u = u_cur + beta * (u_next - u_cur)
        s = u[:,-1]

        # Initialize the augmented Jacobians
        dfdx = jnp.zeros((V.shape[0], n_x, n_x))
        dfdu = jnp.zeros((V.shape[0], n_x, n_u))

        # Ensure x_seq and u have the same batch size
        x = V[:,:self.params.sim.n_states]
        u = u[:x.shape[0]]

        # Compute the nonlinear propagation term
        f = self.state_dot(x, u[:,:-1])
        F = s[:, None] * f

        # Evaluate the State Jacobian
        dfdx = self.A(x, u[:,:-1])
        sdfdx = s[:, None, None] * dfdx

        # Evaluate the Control Jacobian
        dfdu_veh = self.B(x, u[:,:-1])
        dfdu = dfdu.at[:, :, :-1].set(s[:, None, None] * dfdu_veh)
        dfdu = dfdu.at[:, :, -1].set(f)
        
        # Compute the defect
        z = F - jnp.einsum('ijk,ik->ij', sdfdx, x) - jnp.einsum('ijk,ik->ij', dfdu, u)

        # Stack up the results into the augmented state vector
        dVdt = jnp.zeros_like(V)
        dVdt = dVdt.at[:, self.i0:self.i1].set(F)
        dVdt = dVdt.at[:, self.i1:self.i2].set(jnp.matmul(sdfdx, V[:, self.i1:self.i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x))
        dVdt = dVdt.at[:, self.i2:self.i3].set((jnp.matmul(sdfdx, V[:, self.i2:self.i3].reshape(-1, n_x, n_u)) + dfdu * alpha).reshape(-1, n_x * n_u))
        dVdt = dVdt.at[:, self.i3:self.i4].set((jnp.matmul(sdfdx, V[:, self.i3:self.i4].reshape(-1, n_x, n_u)) + dfdu * beta).reshape(-1, n_x * n_u))
        dVdt = dVdt.at[:, self.i4:self.i5].set((jnp.matmul(sdfdx, V[:, self.i4:self.i5].reshape(-1, n_x)[..., None]).squeeze(-1) + z).reshape(-1, n_x))
        return dVdt.flatten()
    
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
    params: Config,
) -> jnp.ndarray:
    
    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

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
    if params.dis.dis_type == 'ZOH':
        beta = 0.
    elif params.dis.dis_type == 'FOH':
        beta = (tau) * params.scp.n
    alpha = 1 - beta

    # Interpolate the control input
    u = u_cur + beta * (u_next - u_cur)
    s = u[:,-1]

    # Initialize the augmented Jacobians
    dfdx = jnp.zeros((V.shape[0], n_x, n_x))
    dfdu = jnp.zeros((V.shape[0], n_x, n_u))

    # Ensure x_seq and u have the same batch size
    x = V[:,:params.sim.n_states]
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

def calculate_discretization(x, u, state_dot, A, B, params):
    """
    x: (N, n_x) array of states
    u: (N, n_u+1) array of controls (+ slack)
    state_dot, A, B: callables matching your originals
    params: must have sim.n_states, sim.n_controls, scp.n,
            dis.custom_integrator (bool), dis.solver, dis.rtol, dis.atol, dis.args,
            dev.debug (bool)
    Returns A_bar, B_bar, C_bar, z_bar, Vmulti
    """
    n_x = params.sim.n_states
    n_u = params.sim.n_controls
    N   = params.scp.n

    # build tau grid
    tau_grid = jnp.linspace(0, 1, N)

    # initial augmented state
    aug_dim = n_x + n_x*n_x + 2*n_x*n_u + n_x
    V0 = jnp.zeros((N-1, aug_dim))
    V0 = V0.at[:, :n_x].set(x[:-1].astype(float))
    V0 = V0.at[:, n_x:n_x+n_x*n_x].set(
        jnp.eye(n_x).reshape(1,-1).repeat(N-1, axis=0)
    )

    # choose integrator
    if params.dis.custom_integrator:
        sol = solve_ivp_rk45(
            lambda t,y,*a: dVdt(t, y, *a),
            (0,1), V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, params),
            debug=params.dev.debug,
            # num_steps=N,
        )
    else:
        sol = solve_ivp_diffrax(
            lambda t,y,*a: dVdt(t, y, *a),
            (0,1), V0.reshape(-1),
            args=(u[:-1].astype(float), u[1:].astype(float),
                  state_dot, A, B, params),
            solver_name=params.dis.solver,
            rtol=params.dis.rtol,
            atol=params.dis.atol,
            extra_kwargs=params.dis.args,
            # num_steps=N,
        )

    Vend   = sol[-1].reshape(-1, aug_dim)
    Vmulti = sol.reshape(N, -1, aug_dim)

    A_bar = Vend[:, n_x:n_x+n_x*n_x] \
        .reshape(N-1,n_x,n_x).transpose(1,2,0) \
        .reshape(n_x*n_x, N-1, order='F').T
    B_bar = Vend[:, n_x+n_x*n_x:n_x+n_x*n_x+n_x*n_u] \
        .reshape(N-1,n_x,n_u).transpose(1,2,0) \
        .reshape(n_x*n_u, N-1, order='F').T
    C_bar = Vend[:, n_x+n_x*n_x+n_x*n_u:n_x+n_x*n_x+2*n_x*n_u] \
        .reshape(N-1,n_x,n_u).transpose(1,2,0) \
        .reshape(n_x*n_u, N-1, order='F').T
    z_bar = Vend[:, -n_x:]

    return A_bar, B_bar, C_bar, z_bar, Vmulti


def get_discretization_solver(state_dot, A, B, params):
    return lambda x, u: calculate_discretization(x, u, state_dot, A, B, params)

