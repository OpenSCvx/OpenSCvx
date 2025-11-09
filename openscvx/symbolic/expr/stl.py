from typing import Tuple

import numpy as np

from .expr import Expr, to_expr


class Or(Expr):
    """Logical OR operation for STL expressions"""

    def __init__(self, *operands):
        if len(operands) < 2:
            raise ValueError("Or requires at least two operands")
        self.operands = [to_expr(op) for op in operands]

    def children(self):
        return self.operands

    def canonicalize(self) -> "Expr":
        """Flatten nested Or expressions and canonicalize operands."""
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
        """Validates operand shapes and returns scalar."""
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
