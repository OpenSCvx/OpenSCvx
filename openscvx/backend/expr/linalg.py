from .expr import Expr, to_expr


class Transpose(Expr):
    """Matrix transpose operation"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"({self.operand!r}).T"


class Diag(Expr):
    """Create diagonal matrix from vector or extract diagonal from matrix"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"diag({self.operand!r})"


class Norm(Expr):
    """Norm of an expression (reduction operation)"""

    def __init__(self, operand, ord="fro"):
        self.operand = to_expr(operand)
        self.ord = ord  # Can be "fro", "inf", 1, 2, etc.

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"norm({self.operand!r}, ord={self.ord!r})"


class Stack(Expr):
    """Stack expressions into a matrix - similar to jnp.array([[row1], [row2], ...])"""

    def __init__(self, rows):
        # rows should be a list of expressions representing each row
        self.rows = [to_expr(row) for row in rows]

    def children(self):
        return self.rows

    def __repr__(self):
        rows_repr = ", ".join(repr(row) for row in self.rows)
        return f"Stack([{rows_repr}])"


class Hstack(Expr):
    """Horizontal stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Hstack([{arrays_repr}])"


class Vstack(Expr):
    """Vertical stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Vstack([{arrays_repr}])"
