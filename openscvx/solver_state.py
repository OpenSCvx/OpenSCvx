"""Solver state management for SCP iterations.

This module contains the SolverState dataclass that holds all mutable state
during successive convex programming iterations. By separating solver state
from problem definition, we enable clean reset() functionality and prevent
accidental mutation of initial conditions.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.symbolic.expr.control import Control
    from openscvx.symbolic.expr.state import State


@dataclass
class SolverState:
    """Mutable state for SCP iterations.

    This dataclass holds all state that changes during the solve process,
    including copies of State/Control objects that evolve during optimization.
    A fresh instance is created for each solve, enabling easy reset functionality.

    The State/Control objects stored here are shallow copies of the original
    problem definition, allowing the solver to mutate trajectories without
    affecting the initial problem settings.

    Attributes:
        k: Current iteration number (starts at 1)
        x: State object containing current trajectory and metadata (copied from settings)
        u: Control object containing current trajectory and metadata (copied from settings)
        J_tr: Current trust region cost
        J_vb: Current virtual buffer cost
        J_vc: Current virtual control cost
        w_tr: Current trust region weight (may adapt during solve)
        lam_cost: Current cost weight (may relax during solve)
        lam_vc: Current virtual control penalty weight
        lam_vb: Current virtual buffer penalty weight
        x_history: List of state trajectory iterates (arrays only)
        u_history: List of control trajectory iterates (arrays only)
        V_history: List of discretization error history
    """

    k: int
    x: "State"
    u: "Control"
    J_tr: float
    J_vb: float
    J_vc: float
    w_tr: float
    lam_cost: float
    lam_vc: Union[float, np.ndarray]
    lam_vb: float
    x_history: List[np.ndarray] = field(default_factory=list)
    u_history: List[np.ndarray] = field(default_factory=list)
    V_history: List[np.ndarray] = field(default_factory=list)

    @property
    def x_guess(self) -> np.ndarray:
        """Get current state trajectory array.

        Returns:
            Current state trajectory guess, shape (N, n_states)
        """
        return self.x.guess

    @property
    def u_guess(self) -> np.ndarray:
        """Get current control trajectory array.

        Returns:
            Current control trajectory guess, shape (N, n_controls)
        """
        return self.u.guess

    @classmethod
    def from_settings(cls, settings: "Config") -> "SolverState":
        """Create initial solver state from configuration.

        Creates shallow copies of State/Control objects from settings,
        allowing the solver to mutate trajectories without affecting
        the original problem definition.

        Args:
            settings: Configuration object containing initial guesses and SCP parameters

        Returns:
            Fresh SolverState initialized from settings with copied State/Control objects
        """
        # Deep copy ensures all arrays are independent
        x = copy.deepcopy(settings.sim.x)
        u = copy.deepcopy(settings.sim.u)

        return cls(
            k=1,
            x=x,
            u=u,
            J_tr=1e2,
            J_vb=1e2,
            J_vc=1e2,
            w_tr=settings.scp.w_tr,
            lam_cost=settings.scp.lam_cost,
            lam_vc=settings.scp.lam_vc,
            lam_vb=settings.scp.lam_vb,
            x_history=[x.guess.copy()],
            u_history=[u.guess.copy()],
            V_history=[],
        )
