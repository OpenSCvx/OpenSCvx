"""Trajectory visualization and plotting utilities.

.. warning::
    **THIS MODULE IS IN MAJOR NEED OF REFACTORING AND SHOULD NOT BE USED.**

    The plotting module is currently undergoing significant restructuring and
    should be considered unstable. The API is subject to change without notice.
    Use at your own risk.

This module provides visualization utilities for trajectory optimization results.
It includes functions for plotting state trajectories, control inputs, constraint
violations, and creating animations of the optimization process.
"""

from .plotting import (
    frame_args,
    full_subject_traj_time,
    plot_constraint_violation,
    plot_control,
    plot_initial_guess,
    plot_losses,
    plot_scp_animation,
    plot_state,
    qdcm,
    save_gate_parameters,
    scp_traj_interp,
)

__all__ = [
    # Core plotting functions
    "plot_state",
    "plot_control",
    "plot_constraint_violation",
    "plot_losses",
    "plot_scp_animation",
    "plot_initial_guess",
    # Trajectory utilities
    "full_subject_traj_time",
    "scp_traj_interp",
    # Helper functions
    "qdcm",
    "save_gate_parameters",
    "frame_args",
]

# Mark module as unstable/deprecated
__deprecated__ = True
__status__ = "unstable"
