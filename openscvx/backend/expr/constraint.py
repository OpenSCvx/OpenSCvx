from typing import Optional, Tuple

import numpy as np

from ..canonicalizer import canon_visitor, canonicalize
from .expr import Constraint, Expr, Sum


class NodalConstraint(Expr):
    """
    Wrapper for constraints that should only be enforced at specific discrete nodes.

    This separates nodal constraint logic from the base Constraint class,
    providing clean separation of concerns.
    """

    def __init__(self, constraint: Constraint, nodes: list[int]):
        if not isinstance(constraint, Constraint):
            raise TypeError("NodalConstraint must wrap a Constraint")
        if not isinstance(nodes, list):
            raise TypeError("nodes must be a list of integers")

        # Convert numpy integers to Python integers
        converted_nodes = []
        for n in nodes:
            if isinstance(n, np.integer):
                converted_nodes.append(int(n))
            elif isinstance(n, int):
                converted_nodes.append(n)
            else:
                raise TypeError("all node indices must be integers")

        self.constraint = constraint
        self.nodes = converted_nodes

    def children(self):
        return [self.constraint]

    def convex(self) -> "NodalConstraint":
        """Mark the underlying constraint as convex for CVXPy lowering.

        Returns:
            Self with underlying constraint's convex flag set to True (enables method chaining)
        """
        self.constraint.convex()
        return self

    def __repr__(self):
        return f"NodalConstraint({self.constraint!r}, nodes={self.nodes})"


@canon_visitor(NodalConstraint)
def canon_nodal_constraint(node: NodalConstraint) -> Expr:
    # Canonicalize the wrapped constraint and preserve the node specification
    canon_constraint = canonicalize(node.constraint)
    return NodalConstraint(canon_constraint, node.nodes)


# CTCS STUFF


class CTCS(Expr):
    """
    Marks a constraint for continuous-time constraint satisfaction.

    The constraint's left-hand side will be wrapped in a penalty function
    during lowering/compilation.
    """

    def __init__(
        self,
        constraint: Constraint,
        penalty: str = "squared_relu",
        nodes: Optional[Tuple[int, int]] = None,
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ):
        if not isinstance(constraint, Constraint):
            raise TypeError("CTCS must wrap a Constraint")

        # Validate nodes parameter for CTCS
        if nodes is not None:
            if not isinstance(nodes, tuple) or len(nodes) != 2:
                raise ValueError(
                    "CTCS constraints must specify nodes as a tuple of (start, end) or None for all nodes"
                )
            if not all(isinstance(n, int) for n in nodes):
                raise ValueError("CTCS node indices must be integers")
            if nodes[0] >= nodes[1]:
                raise ValueError("CTCS node range must have start < end")

        self.constraint = constraint
        self.penalty = penalty
        self.nodes = nodes  # (start, end) node range or None for all nodes
        self.idx = idx  # Optional grouping index for multiple augmented states
        # Whether to also enforce this constraint nodally for numerical stability
        self.check_nodally = check_nodally

    def children(self):
        return [self.constraint]

    def over(self, interval: tuple[int, int]) -> "CTCS":
        """Set the continuous interval for this CTCS constraint.

        Args:
            interval: Tuple of (start, end) node indices for the continuous interval

        Returns:
            New CTCS constraint with the specified interval
        """
        return CTCS(
            self.constraint,
            penalty=self.penalty,
            nodes=interval,
            idx=self.idx,
            check_nodally=self.check_nodally,
        )

    def __repr__(self):
        parts = [f"{self.constraint!r}", f"penalty={self.penalty!r}"]
        if self.nodes is not None:
            parts.append(f"nodes={self.nodes}")
        if self.idx is not None:
            parts.append(f"idx={self.idx}")
        if self.check_nodally:
            parts.append(f"check_nodally={self.check_nodally}")
        return f"CTCS({', '.join(parts)})"

    def penalty_expr(self) -> Expr:
        """
        Build the penalty expression for this CTCS constraint.
        This transforms the constraint's LHS into a penalized expression.
        """
        lhs = self.constraint.lhs

        if self.penalty == "squared_relu":
            from openscvx.backend.expr.math import PositivePart, Square

            penalty = Square(PositivePart(lhs))
        elif self.penalty == "huber":
            from openscvx.backend.expr.math import Huber, PositivePart

            penalty = Huber(PositivePart(lhs))
        elif self.penalty == "smooth_relu":
            from openscvx.backend.expr.math import SmoothReLU

            penalty = SmoothReLU(lhs)
        else:
            raise ValueError(f"Unknown penalty {self.penalty!r}")

        return Sum(penalty)


@canon_visitor(CTCS)
def canon_ctcs(node: CTCS) -> Expr:
    # Canonicalize the inner constraint but preserve CTCS parameters
    canon_constraint = canonicalize(node.constraint)
    return CTCS(canon_constraint, penalty=node.penalty, nodes=node.nodes)


def ctcs(
    constraint: Constraint,
    penalty: str = "squared_relu",
    nodes: Optional[Tuple[int, int]] = None,
    idx: Optional[int] = None,
    check_nodally: bool = False,
) -> CTCS:
    """Helper function to create CTCS constraints."""
    return CTCS(constraint, penalty, nodes, idx, check_nodally)
