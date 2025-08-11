from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

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

PenaltyKind = Literal["squared_relu", "huber", "smooth_relu"]


@dataclass(frozen=True)
class CTCSInfo:
    penalty: PenaltyKind = "squared_relu"
    delta: float = 0.25  # for huber
    c: float = 1e-8  # for smooth_relu
    scaling: float = 1.0
    check_at_nodes: bool = True
    nodes: Optional[Tuple[int, int]] = None  # keep as metadata for your integrator
    idx: Optional[int] = None


class CTCS(Expr):
    def __init__(self, constraint: Constraint, info: Optional[CTCSInfo] = None):
        if not isinstance(constraint, Constraint):
            raise TypeError("CTCS must wrap a Constraint")
        self.constraint = constraint
        self.info = info or CTCSInfo()

    def children(self):
        return [self.constraint]

    def __repr__(self):
        return f"CTCS({self.constraint!r}, info={self.info})"


def ctcs(c: Constraint, **kwargs) -> CTCS:
    return CTCS(c, CTCSInfo(**kwargs))


# TODO: (norrisg) move to more permanent location
class PositivePart(Expr):  # pos(x) = max(x, 0)
    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"pos({self.x!r})"


class Square(Expr):
    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"({self.x!r})^2"


class Huber(Expr):  # symmetric Huber on its input
    def __init__(self, x, delta: float = 0.25):
        self.x = to_expr(x)
        self.delta = float(delta)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"huber({self.x!r}; delta={self.delta})"


class SmoothReLU(Expr):  # sqrt(pos(x)^2 + c^2) - c
    def __init__(self, x, c: float = 1e-8):
        self.x = to_expr(x)
        self.c = float(c)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"smooth_relu({self.x!r}; c={self.c})"


# TODO: (norrisg) move these functions into separate `augmentation` step that occurs after
# preprocessing, canonicalization
def _penalty_expr(lhs: Expr, info: CTCSInfo) -> Expr:
    if info.penalty == "squared_relu":
        return Square(PositivePart(lhs))
    if info.penalty == "huber":
        return Huber(PositivePart(lhs), delta=info.delta)
    if info.penalty == "smooth_relu":
        return SmoothReLU(lhs, c=info.c)  # SmoothReLU uses pos(lhs) internally
    raise ValueError(f"Unknown penalty {info.penalty!r}")


def augment_dynamics_with_ctcs(
    xdot: Expr, constraints: List[Expr]
) -> tuple[Expr, List[Constraint]]:
    constraints_ctcs: List[CTCS] = []
    constraints_nodal: List[Constraint] = []

    for e in constraints:
        if isinstance(e, CTCS):
            constraints_ctcs.append(e)
        elif isinstance(e, Constraint):
            constraints_nodal.append(e)
        else:
            raise ValueError(f"Constraints must be `Constraint` or `CTCS`, got {e}")

    state_aug: List[Expr] = []
    node_checks: List[Constraint] = []

    # TODO: (norrisg) fix this so that state_aug are also added to the list of `Variable`s according
    # to their idx and node grouping
    for w in constraints_ctcs:
        lhs = w.constraint.lhs
        g = _penalty_expr(lhs, w.info)
        if w.info.scaling != 1.0:
            g = Mul(Constant(np.array(w.info.scaling)), g)
        state_aug.append(g)
        if w.info.check_at_nodes:
            node_checks.append(w.constraint)

    # non-CTCS constraints are always checked at nodes
    for c in constraints_nodal:
        if all(c is not w.constraint for w in constraints_ctcs):
            node_checks.append(c)

    xdot_aug = Concat(xdot, *state_aug) if state_aug else xdot
    return xdot_aug, node_checks
