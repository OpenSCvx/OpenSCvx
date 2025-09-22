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
