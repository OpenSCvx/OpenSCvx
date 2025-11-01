import numpy as np

from ..canonicalizer import canon_visitor, canonicalize
from ..shape_checker import check_shape, shape_visitor
from .expr import Expr, to_expr


class Or(Expr):
    """Logical OR operation for STL expressions"""

    def __init__(self, *operands):
        if len(operands) < 2:
            raise ValueError("Or requires at least two operands")
        self.operands = [to_expr(op) for op in operands]

    def children(self):
        return self.operands

    def __repr__(self):
        operands_repr = " | ".join(repr(op) for op in self.operands)
        return f"Or({operands_repr})"


@canon_visitor(Or)
def canon_or(node: Or) -> Expr:
    # Flatten nested Or expressions and canonicalize operands
    operands: list[Expr] = []

    for operand in node.operands:
        canonicalized = canonicalize(operand)
        if isinstance(canonicalized, Or):
            # Flatten nested Or: Or(a, Or(b, c)) -> Or(a, b, c)
            operands.extend(canonicalized.operands)
        else:
            operands.append(canonicalized)

    # Return simplified Or expression
    if len(operands) == 1:
        return operands[0]
    return Or(*operands)


@shape_visitor(Or)
def check_shape_or(node: Or) -> tuple[int, ...]:
    """Logical OR operation for STL expressions - validates operand shapes and returns scalar"""
    if len(node.operands) < 2:
        raise ValueError("Or requires at least two operands")

    # Validate all operands and get their shapes
    operand_shapes = [check_shape(operand) for operand in node.operands]

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
