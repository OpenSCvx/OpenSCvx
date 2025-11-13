"""Autotuning functions for SCP (Successive Convex Programming) parameters."""

from openscvx.config import Config


def update_scp_weights(settings: Config, scp_k: int):
    """Update SCP weights and cost parameters based on iteration number.

    Args:
        settings: Configuration object containing SCP parameters
        scp_k: Current SCP iteration number
    """
    # Update trust region weight
    settings.scp.w_tr = min(settings.scp.w_tr * settings.scp.w_tr_adapt, settings.scp.w_tr_max)

    # Update cost relaxation parameter after cost_drop iterations
    if scp_k > settings.scp.cost_drop:
        settings.scp.lam_cost = settings.scp.lam_cost * settings.scp.cost_relax
