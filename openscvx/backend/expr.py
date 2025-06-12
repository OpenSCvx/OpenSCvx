from typing import Union

import numpy as np


class Expr:
    """
    Base class for symbolic expressions in optimization problems.

    Note: This class is currently not being used.
    """


    def __le__(self, other):
        return Constraint(self, to_expr(other), op="<=")

    def __ge__(self, other):
        return Constraint(self, to_expr(other), op=">=")

    def __eq__(self, other):
        return Constraint(self, to_expr(other), op="==")

    def __add__(self, other):
        return Add(self, to_expr(other))

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


class Add(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class MatMul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class Neg(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]


class Literal(Expr):
    """Represents a literal value in an expression."""

    def __init__(self, value):
        self.value = value

    def children(self):
        return []


def to_expr(obj):
    """Convert an object to an expression."""
    if isinstance(obj, Expr):
        return obj
    return Literal(obj)



class Constant(Expr):
    def __init__(self, value: np.ndarray):
        self.value = value

    def __repr__(self):
        return f"Const({self.value!r})"


class Constraint(Expr):
    """
    A comparison node.  op is one of '<=', '>=', or '=='.
    """

    def __init__(self, lhs: Expr, rhs: Expr, op: str):
        assert op in ("<=", ">=", "=="), f"Invalid op {op}"
        self.lhs, self.rhs, self.op = lhs, rhs, op

    def __repr__(self):
        return f"({self.lhs!r} {self.op} {self.rhs!r})"
