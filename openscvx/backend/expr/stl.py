from ..canonicalizer import canon_visitor, canonicalize
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
def canon_or(self, node: Or) -> Expr:
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
