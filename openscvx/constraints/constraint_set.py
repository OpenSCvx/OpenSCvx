"""Container for categorized constraints in trajectory optimization.

This module provides a dataclass to hold all constraint types in a structured way,
replacing the previous pattern of passing multiple lists or storing separate fields.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from openscvx.symbolic.expr import CTCS, CrossNodeConstraint, NodalConstraint

    from .cross_node import CrossNodeConstraintLowered
    from .lowered import LoweredNodalConstraint


@dataclass
class ConstraintSet:
    """Container for categorized constraints.

    This dataclass holds all constraint types in a structured way, providing:
    - Type safety through named fields instead of tuple unpacking
    - Clear API for accessing constraint categories
    - Easy extensibility when adding new constraint types

    The constraint set can hold either symbolic constraints (before lowering)
    or lowered constraints (after lowering to JAX/CVXPy), depending on the
    stage of the pipeline.

    Attributes:
        ctcs: CTCS (continuous-time) constraints
        nodal: Non-convex nodal constraints (lowered to JAX for SCP linearization)
        nodal_convex: Convex nodal constraints (lowered to CVXPy for direct solving)
        cross_node: Non-convex cross-node constraints (lowered to JAX)
        cross_node_convex: Convex cross-node constraints (lowered to CVXPy)

    Example:
        Creating a constraint set from separate_constraints::

            constraints = ConstraintSet()
            constraints.ctcs.append(ctcs_constraint)
            constraints.nodal.append(nodal_constraint)

        Accessing constraints::

            for c in constraints.nodal:
                # Process non-convex nodal constraints
                pass

            if constraints.cross_node:
                # Handle cross-node constraints
                pass
    """

    ctcs: List["CTCS"] = field(default_factory=list)
    nodal: List["NodalConstraint | LoweredNodalConstraint"] = field(default_factory=list)
    nodal_convex: List["NodalConstraint"] = field(default_factory=list)
    cross_node: List["CrossNodeConstraint | CrossNodeConstraintLowered"] = field(
        default_factory=list
    )
    cross_node_convex: List["CrossNodeConstraint"] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return True if any constraint category is non-empty."""
        return bool(
            self.ctcs
            or self.nodal
            or self.nodal_convex
            or self.cross_node
            or self.cross_node_convex
        )

    def __len__(self) -> int:
        """Return total number of constraints across all categories."""
        return (
            len(self.ctcs)
            + len(self.nodal)
            + len(self.nodal_convex)
            + len(self.cross_node)
            + len(self.cross_node_convex)
        )
