import numpy as np
import jax.numpy as jnp

from openscvx.propagation import s_to_t, t_to_tau, simulate_nonlinear_time
from openscvx.config import Config


def propagate_trajectory_results(params: Config, result: dict, propagation_solver: callable) -> dict:
    x = result["x"]
    u = result["u"]

    t = np.array(s_to_t(u, params))

    t_full = np.arange(0, t[-1], params.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, u, t, params)

    # Match free values from initial state to the initial value from the result
    mask = jnp.array([t == "Free" for t in params.sim.initial_state_prop.types], dtype=bool)
    params.sim.initial_state_prop.value = jnp.where(mask, x[0], params.sim.initial_state_prop.value)

    x_full = simulate_nonlinear_time(params.sim.initial_state_prop.value, u, tau_vals, t, params, propagation_solver)

    print("Total CTCS Constraint Violation:", x_full[-1, params.sim.idx_y_prop])
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
    i=0
    for type in params.sim.initial_state.type:
        if type == "Maximize":
            cost -= x[0, i]
        i += 1
    i = 0
    for type in params.sim.final_state.type:
        if type == "Maximize":
            cost -= x[-1, i]
        i += 1
    print("Cost: ", cost)

    more_result = dict(t_full=t_full, x_full=x_full, u_full=u_full)

    result.update(more_result)
    return result
