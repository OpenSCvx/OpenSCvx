from ..canonicalizer import canon_visitor, canonicalize
from .expr import Expr, to_expr


class Transpose(Expr):
    """Matrix transpose operation"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"({self.operand!r}).T"


@canon_visitor(Transpose)
def canon_transpose(node: Transpose) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)

    # Double transpose optimization: (A.T).T = A
    if isinstance(operand, Transpose):
        return operand.operand

    return Transpose(operand)


class Diag(Expr):
    """Create diagonal matrix from vector or extract diagonal from matrix"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"diag({self.operand!r})"


@canon_visitor(Diag)
def canon_diag(node: Diag) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Diag(operand)


class Norm(Expr):
    """Norm of an expression (reduction operation)"""

    def __init__(self, operand, ord="fro"):
        self.operand = to_expr(operand)
        self.ord = ord  # Can be "fro", "inf", 1, 2, etc.

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"norm({self.operand!r}, ord={self.ord!r})"


@canon_visitor(Norm)
def canon_norm(node: Norm) -> Expr:
    # Canonicalize the operand but preserve the ord parameter
    canon_operand = canonicalize(node.operand)
    return Norm(canon_operand, ord=node.ord)


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


@canon_visitor(Stack)
def canon_stack(node: Stack) -> Expr:
    # Canonicalize all rows
    rows = [canonicalize(row) for row in node.rows]
    return Stack(rows)


class Hstack(Expr):
    """Horizontal stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Hstack([{arrays_repr}])"


@canon_visitor(Hstack)
def canon_hstack(node: Hstack) -> Expr:
    # Canonicalize all arrays
    arrays = [canonicalize(arr) for arr in node.arrays]
    return Hstack(arrays)


class Vstack(Expr):
    """Vertical stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Vstack([{arrays_repr}])"


@canon_visitor(Vstack)
def canon_vstack(node: Vstack) -> Expr:
    # Canonicalize all arrays
    arrays = [canonicalize(arr) for arr in node.arrays]
    return Vstack(arrays)
