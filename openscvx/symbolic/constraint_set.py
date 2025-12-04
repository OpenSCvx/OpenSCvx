"""Container for categorized symbolic constraints.

This module provides a dataclass to hold all symbolic constraint types in a
structured way before they are lowered to JAX/CVXPy.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from openscvx.symbolic.expr import CTCS, Constraint, CrossNodeConstraint, NodalConstraint


@dataclass
class ConstraintSet:
    """Container for categorized symbolic constraints.

    This dataclass holds all symbolic constraint types in a structured way,
    providing type safety and a clear API for accessing constraint categories.
    This is a pre-lowering container - after lowering, constraints live in
    LoweredJaxConstraints and LoweredCvxpyConstraints.

    The constraint set supports two lifecycle stages:

    1. **Before preprocessing**: Raw constraints live in `unsorted`
    2. **After preprocessing**: `unsorted` is empty, constraints are categorized

    Use `is_categorized` to check which stage the constraint set is in.

    Attributes:
        unsorted: Raw constraints before categorization. Empty after preprocessing.
        ctcs: CTCS (continuous-time) constraints.
        nodal: Non-convex nodal constraints (will be lowered to JAX).
        nodal_convex: Convex nodal constraints (will be lowered to CVXPy).
        cross_node: Non-convex cross-node constraints (will be lowered to JAX).
        cross_node_convex: Convex cross-node constraints (will be lowered to CVXPy).

    Example:
        Before preprocessing (raw constraints)::

            constraints = ConstraintSet(unsorted=[c1, c2, c3])
            assert not constraints.is_categorized

        After preprocessing (categorized)::

            # preprocess_symbolic_problem drains unsorted -> fills categories
            assert constraints.is_categorized
            for c in constraints.nodal:
                # Process non-convex nodal constraints
                pass
    """

    # Raw constraints before categorization (empty after preprocessing)
    unsorted: List[Union["Constraint", "CTCS"]] = field(default_factory=list)

    # Categorized symbolic constraints (populated by preprocessing)
    ctcs: List["CTCS"] = field(default_factory=list)
    nodal: List["NodalConstraint"] = field(default_factory=list)
    nodal_convex: List["NodalConstraint"] = field(default_factory=list)
    cross_node: List["CrossNodeConstraint"] = field(default_factory=list)
    cross_node_convex: List["CrossNodeConstraint"] = field(default_factory=list)

    @property
    def is_categorized(self) -> bool:
        """True if all constraints have been sorted into categories.

        After preprocessing, `unsorted` should be empty and all constraints
        should be in their appropriate category lists.
        """
        return len(self.unsorted) == 0

    def __bool__(self) -> bool:
        """Return True if any constraint list is non-empty."""
        return bool(
            self.unsorted
            or self.ctcs
            or self.nodal
            or self.nodal_convex
            or self.cross_node
            or self.cross_node_convex
        )

    def __len__(self) -> int:
        """Return total number of constraints across all lists."""
        return (
            len(self.unsorted)
            + len(self.ctcs)
            + len(self.nodal)
            + len(self.nodal_convex)
            + len(self.cross_node)
            + len(self.cross_node_convex)
        )
