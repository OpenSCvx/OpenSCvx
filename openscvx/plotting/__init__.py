"""Trajectory visualization and plotting utilities.

!!! danger "Important"
    **THIS MODULE IS IN MAJOR NEED OF REFACTORING AND SHOULD NOT BE USED.**

    The plotting module is currently undergoing significant restructuring and
    should be considered unstable. The API is subject to change without notice.
    Use at your own risk.

This module provides visualization utilities for trajectory optimization results.
It includes functions for plotting state trajectories, control inputs, constraint
violations, and creating animations of the optimization process.
"""

from .plotting import (
    plot_control,
    plot_scp_iteration_animation,
    plot_state,
)

__all__ = [
    # Core plotting functions
    "plot_state",
    "plot_control",
    "plot_scp_iteration_animation",
]

# Mark module as unstable/deprecated
__deprecated__ = True
__status__ = "unstable"
