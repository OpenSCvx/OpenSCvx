from typing import Callable, Optional, Tuple, Union

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

    def __radd__(self, other):
        return Add(to_expr(other), self)

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

    def __rmul__(self, other):
        return Mul(to_expr(other), self)

    def __matmul__(self, other):
        return MatMul(self, to_expr(other))

    def __neg__(self):
        return Neg(self)

    def __getitem__(self, idx):
        return Index(self, idx)

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
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Add requires two or more operands")
        self.terms = [to_expr(a) for a in args]

    def children(self):
        return list(self.terms)

    def __repr__(self):
        inner = " + ".join(repr(e) for e in self.terms)
        return f"({inner})"


class Sub(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class Mul(Expr):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Mul requires two or more operands")
        self.factors = [to_expr(a) for a in args]

    def children(self):
        return list(self.factors)

    def __repr__(self):
        inner = " * ".join(repr(e) for e in self.factors)
        return f"({inner})"


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


class Sum(Expr):
    """Sum all elements of an expression (reduction operation)"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"sum({self.operand!r})"


class Index(Expr):
    """Expr that means “take this Expr and index/slice it.”"""

    def __init__(self, base: Expr, index: Union[int, slice, tuple]):
        self.base = base
        self.index = index

    def children(self):
        return [self.base]

    def __repr__(self):
        return f"{self.base!r}[{self.index!r}]"


class Concat(Expr):
    """
    Concatenate a sequence of Exprs into one long vector.
    """

    def __init__(self, *exprs: Expr):
        # wrap raw values as Constant if needed
        self.exprs = [to_expr(e) for e in exprs]

    def children(self):
        return list(self.exprs)

    def __repr__(self):
        inner = ", ".join(repr(e) for e in self.exprs)
        return f"Concat({inner})"


class Sin(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(sin{self.operand!r})"


class Cos(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(sin{self.operand!r})"


# class Literal(Expr):
#     """Represents a literal value in an expression."""

#     def __init__(self, value):
#         self.value = value

#     def children(self):
#         return []


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


# CTCS STUFF
# TODO: (norrisg) move to a separate location


# Penalty function building blocks
class PositivePart(Expr):
    """pos(x) = max(x, 0)"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"pos({self.x!r})"


class Square(Expr):
    """x^2"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"({self.x!r})^2"


class Huber(Expr):
    """Huber penalty function"""

    def __init__(self, x, delta: float = 0.25):
        self.x = to_expr(x)
        self.delta = float(delta)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"huber({self.x!r}, delta={self.delta})"


class SmoothReLU(Expr):
    """sqrt(max(x, 0)^2 + c^2) - c"""

    def __init__(self, x, c: float = 1e-8):
        self.x = to_expr(x)
        self.c = float(c)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"smooth_relu({self.x!r}, c={self.c})"


# CTCS constraint wrapper
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
        self.constraint = constraint
        self.penalty = penalty
        self.nodes = nodes  # (start, end) node range or None for all nodes
        self.idx = idx  # Optional grouping index for multiple augmented states
        # Whether to also enforce this constraint nodally for numerical stability
        self.check_nodally = check_nodally

    def children(self):
        return [self.constraint]

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
            penalty = Square(PositivePart(lhs))
        elif self.penalty == "huber":
            penalty = Huber(PositivePart(lhs))
        elif self.penalty == "smooth_relu":
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
