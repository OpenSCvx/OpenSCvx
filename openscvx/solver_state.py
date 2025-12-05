"""Solver state management for SCP iterations.

This module contains the SolverState dataclass that holds all mutable state
during successive convex programming iterations. By separating solver state
from problem definition, we enable clean reset() functionality and prevent
accidental mutation of initial conditions.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.config import Config


@dataclass
class SolverState:
    """Mutable state for SCP iterations.

    This dataclass holds all state that changes during the solve process.
    A fresh instance is created for each solve, enabling easy reset functionality.

    Attributes:
        k: Current iteration number (starts at 1)
        x_guess: Current state trajectory guess, shape (N, n_states)
        u_guess: Current control trajectory guess, shape (N, n_controls)
        J_tr: Current trust region cost
        J_vb: Current virtual buffer cost
        J_vc: Current virtual control cost
        w_tr: Current trust region weight (may adapt during solve)
        lam_cost: Current cost weight (may relax during solve)
        lam_vc: Current virtual control penalty weight
        x_history: List of state trajectory iterates
        u_history: List of control trajectory iterates
        V_history: List of discretization error history
    """

    k: int
    x_guess: np.ndarray
    u_guess: np.ndarray
    J_tr: float
    J_vb: float
    J_vc: float
    w_tr: float
    lam_cost: float
    lam_vc: Union[float, np.ndarray]
    x_history: List[np.ndarray] = field(default_factory=list)
    u_history: List[np.ndarray] = field(default_factory=list)
    V_history: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: "Config") -> "SolverState":
        """Create initial solver state from configuration.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters

        Returns:
            Fresh SolverState initialized from settings
        """
        x_guess = settings.sim.x.guess.copy()
        u_guess = settings.sim.u.guess.copy()

        return cls(
            k=1,
            x_guess=x_guess,
            u_guess=u_guess,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            w_tr=settings.scp.w_tr,
            lam_cost=settings.scp.lam_cost,
            lam_vc=settings.scp.lam_vc,
            x_history=[x_guess.copy()],
            u_history=[u_guess.copy()],
            V_history=[],
        )
