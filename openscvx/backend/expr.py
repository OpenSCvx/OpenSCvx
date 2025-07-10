from typing import Callable, Union

import numpy as np


class Expr:
    """
    Base class for symbolic expressions in optimization problems.

    Note: This class is currently not being used.
    """

    def __le__(self, other):
        return Inequality(self, to_expr(other))

    def __ge__(self, other):
        return Inequality(to_expr(other), self)

    def __eq__(self, other):
        return Equality(self, to_expr(other))

    def __add__(self, other):
        return Add(self, to_expr(other))

    def __sub__(self, other):
        return Sub(self, to_expr(other))

    def __rsub__(self, other):
        # e.g. 5 - a  ⇒ Sub(Constant(5), a)
        return Sub(to_expr(other), self)

    def __truediv__(self, other):
        return Div(self, to_expr(other))

    def __rtruediv__(self, other):
        # e.g. 10 / a
        return Div(to_expr(other), self)

    def __mul__(self, other):
        return Mul(self, to_expr(other))

    def __matmul__(self, other):
        return MatMul(self, to_expr(other))

    def __neg__(self):
        return Neg(self)

    def children(self):
        return []

    def pretty(self, indent=0):
        pad = "  " * indent
        pad = "  " * indent
        lines = [f"{pad}{self.__class__.__name__}"]
        for child in self.children():
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)


def to_expr(x: Union[Expr, float, int, np.ndarray]) -> Expr:
    return x if isinstance(x, Expr) else Constant(np.array(x))


def traverse(expr: Expr, visit: Callable[[Expr], None]):
    visit(expr)
    for child in expr.children():
        traverse(child, visit)


class Add(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} + {self.right!r})"


class Sub(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} * {self.right!r})"


class Div(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} / {self.right!r})"


class MatMul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} * {self.right!r})"


class Neg(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(-{self.operand!r})"


class Literal(Expr):
    """Represents a literal value in an expression."""

    def __init__(self, value):
        self.value = value

    def children(self):
        return []


# def to_expr(obj):
#     """Convert an object to an expression."""
#     if isinstance(obj, Expr):
#         return obj
#     return Literal(obj)


class Constant(Expr):
    def __init__(self, value: np.ndarray):
        self.value = value

    def __repr__(self):
        return f"Const({self.value!r})"


class Constraint(Expr):
    """
    Abstract base for all constraints.
    """
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

    def children(self):
        return [self.lhs, self.rhs]


class Equality(Constraint):
    """Represents lhs == rhs."""
    def __repr__(self):
        return f"{self.lhs!r} == {self.rhs!r}"


class Inequality(Constraint):
    """Represents lhs <= rhs"""
    def __repr__(self):
        return f"{self.lhs!r} <= {self.rhs!r}"
