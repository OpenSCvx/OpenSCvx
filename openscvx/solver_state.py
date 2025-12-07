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
    It stores only the evolving trajectory arrays, not the full State/Control
    objects which contain immutable configuration metadata.

    Trajectory arrays are stored in history lists, with the current guess
    accessed via properties that return the latest entry.

    A fresh instance is created for each solve, enabling easy reset functionality.

    Attributes:
        k: Current iteration number (starts at 1)
        J_tr: Current trust region cost
        J_vb: Current virtual buffer cost
        J_vc: Current virtual control cost
        w_tr: Current trust region weight (may adapt during solve)
        lam_cost: Current cost weight (may relax during solve)
        lam_vc: Current virtual control penalty weight
        lam_vb: Current virtual buffer penalty weight
        X: List of state trajectory iterates
        U: List of control trajectory iterates
        V_sequence: List of discretization error history
    """

    k: int
    J_tr: float
    J_vb: float
    J_vc: float
    w_tr: float
    lam_cost: float
    lam_vc: Union[float, np.ndarray]
    lam_vb: float
    X: List[np.ndarray] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    V_sequence: List[np.ndarray] = field(default_factory=list)

    @property
    def x(self) -> np.ndarray:
        """Get current state trajectory array.

        Returns:
            Current state trajectory guess (latest entry in history), shape (N, n_states)
        """
        return self.X[-1]

    @property
    def u(self) -> np.ndarray:
        """Get current control trajectory array.

        Returns:
            Current control trajectory guess (latest entry in history), shape (N, n_controls)
        """
        return self.U[-1]

    @classmethod
    def from_settings(cls, settings: "Config") -> "SolverState":
        """Create initial solver state from configuration.

        Copies only the trajectory arrays from settings, leaving all metadata
        (bounds, boundary conditions, etc.) in the original settings object.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters

        Returns:
            Fresh SolverState initialized from settings with copied arrays
        """
        return cls(
            k=1,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            w_tr=settings.scp.w_tr,
            lam_cost=settings.scp.lam_cost,
            lam_vc=settings.scp.lam_vc,
            lam_vb=settings.scp.lam_vb,
            X=[settings.sim.x.guess.copy()],
            U=[settings.sim.u.guess.copy()],
            V_sequence=[],
        )
