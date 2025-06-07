import numpy as np
import jax.numpy as jnp

from openscvx.propagation import s_to_t, t_to_tau, simulate_nonlinear_time
from openscvx.config import Config


def propagate_trajectory_results(params: dict, settings: Config, result: dict, propagation_solver: callable) -> dict:
    x = result["x"]
    u = result["u"]

    t = np.array(s_to_t(x, u, settings)).squeeze()

    t_full = np.arange(t[0], t[-1], settings.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, t, settings)

    # Match free values from initial state to the initial value from the result
    mask = jnp.array([t == "Free" for t in x.initial_type], dtype=bool)
    settings.sim.x_prop.initial = jnp.where(mask, x.guess[0,:], settings.sim.x_prop.initial)

    x_full = simulate_nonlinear_time(params, x, u, tau_vals, t, settings, propagation_solver)

    print("Total CTCS Constraint Violation:", x_full[-1, settings.sim.idx_y_prop])
    i = 0
    cost = np.zeros_like(x.guess[-1,i])
    for type in x.initial_type:
        if type == "Minimize":
            cost += x.guess[0, i]
        i += 1
    i = 0
    for type in x.final_type:
        if type == "Minimize":
            cost += x.guess[-1, i]
        i += 1
    i=0
    for type in x.initial_type:
        if type == "Maximize":
            cost -= x.guess[0, i]
        i += 1
    i = 0
    for type in x.final_type:
        if type == "Maximize":
            cost -= x.guess[-1, i]
        i += 1
    print("Cost: ", cost)

    more_result = dict(t_full=t_full, x_full=x_full, u_full=u_full)

    result.update(more_result)
    return result
