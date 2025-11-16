"""Signal Temporal Logic (STL) operations for trajectory optimization.

This module provides symbolic expression nodes for Signal Temporal Logic (STL)
operations, enabling the specification of complex temporal and logical constraints
in optimization problems. STL is particularly useful for robotics and autonomous
systems where tasks involve temporal reasoning.
"""

from typing import Tuple

import numpy as np

from .expr import Expr, to_expr


class Or(Expr):
    """Logical OR operation for disjunctive constraints.

    Represents a logical disjunction (OR) between multiple constraint expressions.
    This is particularly useful in STL-based trajectory optimization for expressing
    choices or alternatives in task specifications. The Or operation is typically
    relaxed using smooth approximations (e.g., LogSumExp) during optimization.

    The Or operation allows expressing constraints like:

    - "Reach either goal A OR goal B"
    - "Avoid obstacle 1 OR obstacle 2" (at least one must be satisfied)
    - "Use path 1 OR path 2 OR path 3"

    During optimization, the disjunction is typically approximated using:
        Or(φ₁, φ₂, ..., φₙ) ≈ LSE(φ₁, φ₂, ..., φₙ) ≥ 0

    where LSE is the LogSumExp (smooth maximum) function.

    Attributes:
        operands: List of expressions representing the disjunctive clauses

    Example:
        Use Or STL operator to enforce that robot must reach either of two goal regions:

            import openscvx as ox
            x = ox.State("x", shape=(2,))
            goal_a = ox.Parameter("goal_a", shape=(2,), value=[1.0, 1.0])
            goal_b = ox.Parameter("goal_b", shape=(2,), value=[-1.0, -1.0])
            # Robot is within 0.5 units of either goal
            reach_a = 0.25 - ox.Norm(x - goal_a)**2
            reach_b = 0.25 - ox.Norm(x - goal_b)**2
            reach_either = ox.Or(reach_a, reach_b)

    Note:
        The Or operation produces a scalar result even when operands are vector
        expressions, as it represents a single logical proposition.

    See Also:
        LogSumExp: Common smooth approximation for OR operations
        Max: Hard maximum (non-smooth alternative)
    """

    def __init__(self, *operands):
        """Initialize a logical OR operation.

        Args:
            *operands: Two or more expressions to combine with logical OR.
                      Each operand typically represents a constraint or condition.

        Raises:
            ValueError: If fewer than two operands are provided
        """
        if len(operands) < 2:
            raise ValueError("Or requires at least two operands")
        self.operands = [to_expr(op) for op in operands]

    def children(self):
        return self.operands

    def canonicalize(self) -> "Expr":
        """Canonicalize by flattening nested Or expressions.

        Flattens nested Or operations into a single flat Or with all clauses
        at the same level. For example: Or(a, Or(b, c)) → Or(a, b, c).
        Also canonicalizes all operands recursively.

        Returns:
            Expr: Canonical form of the Or expression. If only one operand
                  remains after canonicalization, returns that operand directly.
        """
        operands = []

        for operand in self.operands:
            canonicalized = operand.canonicalize()
            if isinstance(canonicalized, Or):
                # Flatten nested Or: Or(a, Or(b, c)) -> Or(a, b, c)
                operands.extend(canonicalized.operands)
            else:
                operands.append(canonicalized)

        # Return simplified Or expression
        if len(operands) == 1:
            return operands[0]
        return Or(*operands)

    def check_shape(self) -> Tuple[int, ...]:
        """Validate operand shapes and return result shape.

        Checks that all operands have compatible (broadcastable) shapes. The Or
        operation supports broadcasting, allowing mixing of scalars and vectors.

        Returns:
            tuple: Empty tuple () indicating a scalar result, as Or represents
                   a single logical proposition

        Raises:
            ValueError: If fewer than two operands exist
            ValueError: If operand shapes are not broadcastable
        """
        if len(self.operands) < 2:
            raise ValueError("Or requires at least two operands")

        # Validate all operands and get their shapes
        operand_shapes = [operand.check_shape() for operand in self.operands]

        # For logical operations, all operands should be broadcastable
        # This allows mixing scalars with vectors for element-wise operations
        try:
            result_shape = operand_shapes[0]
            for shape in operand_shapes[1:]:
                result_shape = np.broadcast_shapes(result_shape, shape)
        except ValueError as e:
            raise ValueError(f"Or operands not broadcastable: {operand_shapes}") from e

        # Or produces a scalar result (like constraints)
        return ()

    def __repr__(self):
        operands_repr = " | ".join(repr(op) for op in self.operands)
        return f"Or({operands_repr})"
