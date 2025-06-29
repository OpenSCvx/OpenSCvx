import numpy as np
import jax.numpy as jnp

from openscvx.propagation import s_to_t, t_to_tau, simulate_nonlinear_time
from openscvx.config import Config
from openscvx.results import OptimizationResults


def propagate_trajectory_results(params: dict, settings: Config, result: OptimizationResults, propagation_solver: callable) -> OptimizationResults:
    """Propagate the optimal trajectory and compute additional results.
    
    This function takes the optimal control solution and propagates it through the
    nonlinear dynamics to compute the actual state trajectory and other metrics.
    
    Args:
        params (dict): System parameters.
        settings (Config): Configuration settings.
        result (OptimizationResults): Optimization results object.
        propagation_solver (callable): Function for propagating the system state.
        
    Returns:
        OptimizationResults: Updated results object containing:
            - t_full: Full time vector
            - x_full: Full state trajectory
            - u_full: Full control trajectory
            - cost: Computed cost
            - ctcs_violation: CTCS constraint violation
    """
    x = result.x
    u = result.u

    t = np.array(s_to_t(x, u, settings)).squeeze()

    t_full = np.arange(t[0], t[-1], settings.prp.dt)

    tau_vals, u_full = t_to_tau(u, t_full, t, settings)

    # Match free values from initial state to the initial value from the result
    mask = jnp.array([t == "Free" for t in x.initial_type], dtype=bool)
    settings.sim.x_prop.initial = jnp.where(mask, x.guess[0,:], settings.sim.x_prop.initial)

    x_full = simulate_nonlinear_time(params, x, u, tau_vals, t, settings, propagation_solver)

    # Calculate cost
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

    # Calculate CTCS constraint violation
    ctcs_violation = x_full[-1, settings.sim.idx_y_prop]

    # Update the results object with post-processing data
    result.t_full = t_full
    result.x_full = x_full
    result.u_full = u_full
    result.cost = cost
    result.ctcs_violation = ctcs_violation

    return result
