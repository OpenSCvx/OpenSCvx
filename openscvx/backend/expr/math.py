from ..canonicalizer import canon_visitor, canonicalize
from .expr import Expr, to_expr


class Sin(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(sin{self.operand!r})"


@canon_visitor(Sin)
def canon_sin(node: Sin) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Sin(operand)


class Cos(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(cos{self.operand!r})"


@canon_visitor(Cos)
def canon_cos(node: Cos) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Cos(operand)


class Square(Expr):
    """x^2"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"({self.x!r})^2"


@canon_visitor(Square)
def canon_square(node: Square) -> Expr:
    # Canonicalize the operand
    x = canonicalize(node.x)
    return Square(x)


class Sqrt(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"sqrt({self.operand!r})"


@canon_visitor(Sqrt)
def canon_sqrt(node: Sqrt) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Sqrt(operand)


class Exp(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"exp({self.operand!r})"


@canon_visitor(Exp)
def canon_exp(node: Exp) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Exp(operand)


class Log(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"log({self.operand!r})"


@canon_visitor(Log)
def canon_log(node: Log) -> Expr:
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Log(operand)


# Penalty function building blocks
class PositivePart(Expr):
    """pos(x) = max(x, 0)"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"pos({self.x!r})"


@canon_visitor(PositivePart)
def canon_positive_part(node: PositivePart) -> Expr:
    # Canonicalize the operand
    x = canonicalize(node.x)
    return PositivePart(x)


class Huber(Expr):
    """Huber penalty function"""

    def __init__(self, x, delta: float = 0.25):
        self.x = to_expr(x)
        self.delta = float(delta)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"huber({self.x!r}, delta={self.delta})"


@canon_visitor(Huber)
def canon_huber(node: Huber) -> Expr:
    # Canonicalize the operand but preserve delta parameter
    x = canonicalize(node.x)
    return Huber(x, delta=node.delta)


class SmoothReLU(Expr):
    """sqrt(max(x, 0)^2 + c^2) - c"""

    def __init__(self, x, c: float = 1e-8):
        self.x = to_expr(x)
        self.c = float(c)

    def children(self):
        return [self.x]

    def __repr__(self):
        return f"smooth_relu({self.x!r}, c={self.c})"


@canon_visitor(SmoothReLU)
def canon_smooth_relu(node: SmoothReLU) -> Expr:
    # Canonicalize the operand but preserve c parameter
    x = canonicalize(node.x)
    return SmoothReLU(x, c=node.c)
