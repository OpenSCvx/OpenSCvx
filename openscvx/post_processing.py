import numpy as np

from openscvx.propagation import s_to_t, t_to_tau, simulate_nonlinear_time
from openscvx.config import Config


def propagate_trajectory_results(params: Config, result: dict, propagation_solver: callable) -> dict:
    x = result["x"]
    u = result["u"]

    t = np.array(s_to_t(u, params))

    t_full = np.arange(0, t[-1], params.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, u, t, params)

    x_full = simulate_nonlinear_time(x[0], u, tau_vals, t, params, propagation_solver)

    print("Total CTCS Constraint Violation:", x_full[-1, params.sim.idx_y])
    i = 0
    cost = np.zeros_like(x[-1, i])
    for type in params.sim.initial_state.type:
        if type == "Minimize":
            cost += x[0, i]
        i += 1
    i = 0
    for type in params.sim.final_state.type:
        if type == "Minimize":
            cost += x[-1, i]
        i += 1
    print("Cost: ", cost)

    more_result = dict(t_full=t_full, x_full=x_full, u_full=u_full)

    result.update(more_result)
    return result
