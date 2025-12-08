"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from typing import TYPE_CHECKING

from openscvx.config import Config

if TYPE_CHECKING:
    from .solver_state import SolverState


def update_scp_weights(state: "SolverState", settings: Config, scp_k: int):
    """Update SCP weights and cost parameters based on iteration number.

    Args:
        state: Solver state containing current weight values (mutated in place)
        settings: Configuration object containing adaptation parameters
        scp_k: Current SCP iteration number
    """
    # Update trust region weight in state
    state.w_tr = min(state.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)

    # Update cost relaxation parameter after cost_drop iterations
    if scp_k > settings.scp.cost_drop:
        state.lam_cost = state.lam_cost * settings.scp.cost_relax
