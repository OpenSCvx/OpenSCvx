"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

import numpy as np

from openscvx.config import Config

if TYPE_CHECKING:
    from .base import AlgorithmState


def update_scp_weights(state: "AlgorithmState", settings: Config, params: dict):
    """Update SCP weights and cost parameters based on iteration number.

    Args:
        state: Solver state containing current weight values (mutated in place)
        settings: Configuration object containing adaptation parameters
        scp_k: Current SCP iteration number
    """
    # Update trust region weight in state
    state.w_tr = min(state.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)

    # Update cost relaxation parameter after cost_drop iterations
    if state.k > settings.scp.cost_drop:
        state.lam_cost = state.lam_cost * settings.scp.cost_relax
    
    state.lam_vc_history.append(settings.scp.lam_vc)
    state.lam_cost_history.append(settings.scp.lam_cost)
    state.w_tr_history.append(settings.scp.w_tr)

    update_acceptance_ratio(state, settings, params)

def calculate_cost_from_state(x, settings: Config):
    """Calculate cost from state vector based on final_type.

    Args:
        state: Solver state containing current state vector

    Returns:
        float: Computed cost
    """
    cost = 0.0
    for i in range(settings.sim.n_states):
        if settings.sim.x.final_type[i] == "Minimize":
            cost += x[-1, i]
        if settings.sim.x.final_type[i] == "Maximize":
            cost -= x[-1, i]
        if settings.sim.x.initial_type[i] == "Minimize":
            cost += x[0, i]
        if settings.sim.x.initial_type[i] == "Maximize":
            cost -= x[0, i]
    return cost

def calculate_linear_penalty(x_bar, u_bar, x_bar_prev, u_bar_prev, A_d, B_d, C_d, x_prop, lam_vc, lam_cost, settings: Config):
    """Calculate linear penalty J_lin = x_bar_k(cost) + lam_vc * nu + lam_vb * nu_vb.

    Computes constraint violations by evaluating the dynamics constraint at x_bar_k and u_bar_k.
    The dynamics constraint violation (nu) is computed by evaluating the full constraint expression:
        x_nonscaled[i] = A_d[i-1] @ dx_nonscaled[i-1] + B_d[i-1] @ du_nonscaled[i-1] 
                        + C_d[i-1] @ du_nonscaled[i] + x_prop[i-1] + nu[i-1]
    where dx and du are computed from the difference between x_bar_k and the previous iteration
    from scp_trajs and scp_controls.
    The nodal constraint violations (nu_vb) are computed by evaluating the constraint
    functions at x_bar_k and u_bar_k.

    Args:
        x_bar: Current SCP iteration state (n_nodes, n_states)
        u_bar: Current SCP iteration control (n_nodes, n_controls)
        x_bar_prev: Previous SCP iteration state (n_nodes, n_states)
        u_bar_prev: Previous SCP iteration control (n_nodes, n_controls)
        A_d: State transition matrices (n_nodes-1, n_states*n_states) - flattened
        B_d: Control influence matrices (n_nodes-1, n_states*n_controls) - flattened
        C_d: Control influence matrices for next node (n_nodes-1, n_states*n_controls) - flattened
        x_prop: Propagated state from previous node (n_nodes-1, n_states)
        lam_vc: Virtual control weight (scalar or matrix)
        lam_cost: Cost relaxation parameter (scalar)
        settings: Configuration object containing adaptation parameters

    Returns:
        float: Computed linear penalty
    """
    cost = calculate_cost_from_state(x_bar, settings)

    n_nodes = x_bar.shape[0]
    n_states = settings.sim.n_states
    n_controls = settings.sim.n_controls

    dx = x_bar - x_bar_prev
    du = u_bar - u_bar_prev

    # Segment index seg = i-1 ranges from 0 to n_nodes-2
    nu = np.zeros((n_nodes - 1, n_states))
    for seg in range(n_nodes - 1):
        # Reshape matrices from flattened arrays
        # The matrices are stored in Fortran (column-major) order as per discretization.py
        # A_bar, B_bar, C_bar are reshaped with order="F" in discretization.py lines 214, 221, 228
        # So we need to use order='F' when reshaping back, or transpose after C-order reshape
        # CVXPy's reshape should handle this, but numpy defaults to C-order
        # To match CVXPy behavior, we reshape with Fortran order
        A_k = A_d[seg].reshape(n_states, n_states, order='F')
        B_k = B_d[seg].reshape(n_states, n_controls, order='F')
        C_k = C_d[seg].reshape(n_states, n_controls, order='F')
        
        # Compute matrix-vector products (matching ocp.py exactly)
        # A_d[seg] @ dx_nonscaled[seg]
        A_dx_seg = A_k @ dx[seg]
        
        # B_d[seg] @ du[seg]
        B_du_seg = B_k @ du[seg]
        
        # C_d[seg] @ du[seg+1]
        C_du_seg = C_k @ du[seg + 1]
        
        # Compute nu[seg] = x[seg+1] - (A_dx + B_du + C_du + x_prop[seg])
        # Note: seg corresponds to i-1, so x[seg+1] is x[i]
        nu[seg] = x_bar[seg + 1] - (A_dx_seg + B_du_seg + C_du_seg + x_prop[seg])
    
    # TODO: Implement nonconvex nodal constraint violations
    
    
    return lam_cost * cost + np.sum(lam_vc * np.abs(nu))

def calculate_nonlinear_penalty(x_prop: np.ndarray, 
                                x: np.ndarray, 
                                lam_vc: np.ndarray, 
                                lam_cost: float, 
                                params: dict, 
                                settings: Config):
    """Calculate nonlinear penalty J_nonlin = x_prop[cost] + lam_vc(x_prop-x_sol) + lam_vb(g(x_prop)).

    Args:
        x_prop: Propagated state (n_nodes-1, n_states)
        x: Previous iteration state (n_nodes, n_states)
        u: Solution control (n_nodes, n_controls)
        lam_vc: Virtual control weight (scalar or matrix)
        lam_cost: Cost relaxation parameter (scalar)
        param_dict: Dictionary of problem parameters
        settings: Configuration object

    Returns:
        float: Nonlinear penalty value
    """
    # TODO: Implement nonconvex nodal constraint violations

    cost = calculate_cost_from_state(x, settings)
    x_diff = x[1:, :] - x_prop

    return lam_cost * cost + np.sum(lam_vc * np.abs(x_diff))

def update_acceptance_ratio(state: "AlgorithmState", 
                               settings: Config, 
                               params: dict):
    """Calculate acceptance ratio for trust region method.
    
    Args:
        state: Solver state containing current weight values
        settings: Configuration object containing adaptation parameters
        params: Dictionary of problem parameters
    """

    x = state.x
    u = state.u
    x_prop = state.x_prop_history[-1]
    lam_vc = state.lam_vc
    lam_cost = state.lam_cost

    if state.k > 1:
        x_prev = state.X[-2]
        u_prev = state.U[-2]
        A_bar_prev = state.A_bar_history[-2]
        B_bar_prev = state.B_bar_history[-2]
        C_bar_prev = state.C_bar_history[-2]
        x_prop_prev = state.x_prop_history[-2]
        lam_vc_prev = state.lam_vc_history[-2]
        lam_cost_prev = state.lam_cost_history[-2]
    else:
        x_prev = state.x
        u_prev = state.u
    
    J_nonlin_current = calculate_nonlinear_penalty(x_prop,
                                                   x,
                                                   lam_vc,
                                                   lam_cost,
                                                   params,
                                                   settings)

    state.J_nonlin_history.append(J_nonlin_current)

    if state.k > 1:
        J_lin_current = calculate_linear_penalty(
            x,
            u,
            x_prev,
            u_prev,
            A_bar_prev,
            B_bar_prev,
            C_bar_prev,
            x_prop_prev,
            lam_vc_prev,
            lam_cost_prev,
            settings
        )

        J_lin_prev = calculate_linear_penalty(
            x_prev,
            u_prev,
            x_prev,
            u_prev,
            A_bar_prev,
            B_bar_prev,
            C_bar_prev,
            x_prop_prev,
            lam_vc_prev,
            lam_cost_prev,
            settings
        )

        return calculate_acceptance_ratio(J_nonlin_current, state.J_nonlin_history[-2], J_lin_current, J_lin_prev)
    else:
        return None
    
    

def calculate_acceptance_ratio(
    J_current_nonlin, J_prev_nonlin, J_current_lin, J_prev_lin
):
    """Calculate acceptance ratio for trust region method.

    The acceptance ratio measures how well the linearized model predicts the actual
    cost reduction. It is defined as:
        rho = (actual_reduction) / (predicted_reduction)
    where:
        actual_reduction = J_prev_nonlin - J_current_nonlin
        predicted_reduction = J_prev_lin - J_current_lin

    Args:
        J_current_nonlin: Current nonlinear penalty
        J_prev_nonlin: Previous nonlinear penalty
        J_current_lin: Current linear penalty
        J_prev_lin: Previous linear penalty
    Returns:
        float: Acceptance ratio. Returns None if predicted_reduction is zero or negative.
    """
    actual_reduction = J_current_nonlin - J_prev_nonlin
    predicted_reduction = J_current_lin - J_prev_lin

    if predicted_reduction >= 0:
        print(f"FUBAR: {predicted_reduction}")

    rho = actual_reduction / predicted_reduction
    return rho




