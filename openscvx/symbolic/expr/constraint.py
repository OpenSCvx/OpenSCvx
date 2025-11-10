from typing import Optional, Tuple

import numpy as np

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

    def canonicalize(self) -> "Expr":
        """Canonicalize the wrapped constraint and preserve the node specification."""
        canon_constraint = self.constraint.canonicalize()
        return NodalConstraint(canon_constraint, self.nodes)

    def check_shape(self) -> Tuple[int, ...]:
        """
        NodalConstraint wraps a constraint but doesn't change its computational meaning,
        just specifies where it should be applied. Always produces a scalar.
        """
        # Validate the wrapped constraint's shape
        self.constraint.check_shape()

        # NodalConstraint produces a scalar like any constraint
        return ()

    def convex(self) -> "NodalConstraint":
        """Mark the underlying constraint as convex for CVXPy lowering.

        Returns:
            Self with underlying constraint's convex flag set to True (enables method chaining)
        """
        self.constraint.convex()
        return self

    def __repr__(self):
        return f"NodalConstraint({self.constraint!r}, nodes={self.nodes})"


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
                    "CTCS constraints must specify nodes as a tuple of (start, end) or None "
                    "for all nodes"
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

    def canonicalize(self) -> "Expr":
        """Canonicalize the inner constraint but preserve CTCS parameters."""
        canon_constraint = self.constraint.canonicalize()
        return CTCS(canon_constraint, penalty=self.penalty, nodes=self.nodes, idx=self.idx, check_nodally=self.check_nodally)

    def check_shape(self) -> Tuple[int, ...]:
        """
        CTCS wraps a constraint and transforms it into a penalty expression.
        The penalty expression is always summed, so CTCS always produces a scalar.
        """
        # First validate the wrapped constraint's shape
        self.constraint.check_shape()

        # Also validate the penalty expression that would be generated
        try:
            penalty_expr = self.penalty_expr()
            penalty_shape = penalty_expr.check_shape()

            # The penalty expression should always be scalar due to Sum wrapper
            if penalty_shape != ():
                raise ValueError(
                    f"CTCS penalty expression should be scalar, but got shape {penalty_shape}"
                )
        except Exception as e:
            # Re-raise with more context about which CTCS node failed
            raise ValueError(f"CTCS penalty expression validation failed: {e}") from e

        # CTCS always produces a scalar due to the Sum in penalty_expr
        return ()

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
            from openscvx.symbolic.expr.math import PositivePart, Square

            penalty = Square(PositivePart(lhs))
        elif self.penalty == "huber":
            from openscvx.symbolic.expr.math import Huber, PositivePart

            penalty = Huber(PositivePart(lhs))
        elif self.penalty == "smooth_relu":
            from openscvx.symbolic.expr.math import SmoothReLU

            penalty = SmoothReLU(lhs)
        else:
            raise ValueError(f"Unknown penalty {self.penalty!r}")

        return Sum(penalty)


def ctcs(
    constraint: Constraint,
    penalty: str = "squared_relu",
    nodes: Optional[Tuple[int, int]] = None,
    idx: Optional[int] = None,
    check_nodally: bool = False,
) -> CTCS:
    """Helper function to create CTCS constraints."""
    return CTCS(constraint, penalty, nodes, idx, check_nodally)
