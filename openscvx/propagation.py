import numpy as np

from openscvx.config import Config
from openscvx.integrators import solve_ivp_diffrax_prop


def prop_aug_dy(
    tau: float,
    x: np.ndarray,
    u_current: np.ndarray,
    u_next: np.ndarray,
    tau_init: float,
    node: int,
    idx_s: int,
    state_dot: callable,
    dis_type: str,
    N: int,
) -> np.ndarray:
    x = x[None, :]

    if dis_type == "ZOH":
        beta = 0.0
    elif dis_type == "FOH":
        beta = (tau - tau_init) * N
    u = u_current + beta * (u_next - u_current)

    return u[:, idx_s] * state_dot(x, u[:, :-1], node).squeeze()


def get_propagation_solver(state_dot, params):
    def propagation_solver(V0, tau_grid, u_cur, u_next, tau_init, node, idx_s):
        return solve_ivp_diffrax_prop(
            f=prop_aug_dy,
            tau_final=tau_grid[1],
            y_0=V0,
            args=(
                u_cur,
                u_next,
                tau_init,
                node,
                idx_s,
                state_dot,
                params.dis.dis_type,
                params.scp.n,
            ),
            tau_0=tau_grid[0],
        )

    return propagation_solver


def s_to_t(u, params: Config):
    t = [0]
    tau = np.linspace(0, 1, params.scp.n)
    for k in range(1, params.scp.n):
        s_kp = u[k - 1, -1]
        s_k = u[k, -1]
        if params.dis.dis_type == "ZOH":
            t.append(t[k - 1] + (tau[k] - tau[k - 1]) * (s_kp))
        else:
            t.append(t[k - 1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k - 1]))
    return t


def t_to_tau(u, t, u_nodal, t_nodal, params: Config):
    u_lam = lambda new_t: np.array(
        [np.interp(new_t, t_nodal, u[:, i]) for i in range(u.shape[1])]
    ).T
    u = np.array([u_lam(t_i) for t_i in t])

    tau = np.zeros(len(t))
    tau_nodal = np.linspace(0, 1, params.scp.n)
    for k in range(1, len(t)):
        k_nodal = np.where(t_nodal < t[k])[0][-1]
        s_kp = u_nodal[k_nodal, -1]
        tp = t_nodal[k_nodal]
        tau_p = tau_nodal[k_nodal]

        s_k = u[k, -1]
        if params.dis.dis_type == "ZOH":
            tau[k] = tau_p + (t[k] - tp) / s_kp
        else:
            tau[k] = tau_p + 2 * (t[k] - tp) / (s_k + s_kp)
    return tau, u


def simulate_nonlinear_time(x_0, u, tau_vals, t, params, propagation_solver):
    states = np.empty(
        (x_0.shape[0], 0)
    )  # Initialize states as a 2D array with shape (n, 0)

    tau = np.linspace(0, 1, params.scp.n)

    u_lam = lambda new_t: np.array(
        [np.interp(new_t, t, u[:, i]) for i in range(u.shape[1])]
    ).T

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
        tau_cur = tau_vals[prev_count : prev_count + count]

        sol = propagation_solver(
            x_0,
            (tau[k], tau[k + 1]),
            controls_current,
            controls_next,
            np.array([[tau[k]]]),
            np.array([[k]]),
            params.sim.idx_s.stop,
        )

        x = sol.ys
        for tau_i in tau_cur:
            new_state = sol.evaluate(tau_i).reshape(-1, 1)  # Ensure new_state is 2D
            states = np.concatenate([states, new_state], axis=1)

        x_0 = x[-1]
        prev_count += count

    return states.T
