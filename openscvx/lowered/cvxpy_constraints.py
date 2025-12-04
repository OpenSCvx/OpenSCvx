"""CVXPy-lowered constraint dataclass."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cvxpy as cp


@dataclass
class LoweredCvxpyConstraints:
    """CVXPy-lowered convex constraints.

    Contains constraints that have been lowered to CVXPy constraint objects.
    These are added directly to the optimal control problem without
    linearization.

    Attributes:
        constraints: List of CVXPy constraint objects (cp.Constraint).
            Includes both nodal and cross-node convex constraints.
    """

    constraints: list["cp.Constraint"] = field(default_factory=list)
