from typing import Tuple

from .expr import Expr, to_expr


class Sin(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Sin(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sin preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"(sin{self.operand!r})"


class Cos(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Cos(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Cos preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"(cos{self.operand!r})"


class Square(Expr):
    """x^2"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        x = self.x.canonicalize()
        return Square(x)

    def check_shape(self) -> Tuple[int, ...]:
        """x^2 preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self):
        return f"({self.x!r})^2"


class Sqrt(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Sqrt(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sqrt preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"sqrt({self.operand!r})"


class Exp(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Exp(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Exp preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"exp({self.operand!r})"


class Log(Expr):
    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Log(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Log preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"log({self.operand!r})"


# Penalty function building blocks
class PositivePart(Expr):
    """pos(x) = max(x, 0)"""

    def __init__(self, x):
        self.x = to_expr(x)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        x = self.x.canonicalize()
        return PositivePart(x)

    def check_shape(self) -> Tuple[int, ...]:
        """pos(x) = max(x, 0) preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self):
        return f"pos({self.x!r})"


class Huber(Expr):
    """Huber penalty function"""

    def __init__(self, x, delta: float = 0.25):
        self.x = to_expr(x)
        self.delta = float(delta)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve delta parameter."""
        x = self.x.canonicalize()
        return Huber(x, delta=self.delta)

    def check_shape(self) -> Tuple[int, ...]:
        """Huber penalty preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self):
        return f"huber({self.x!r}, delta={self.delta})"


class SmoothReLU(Expr):
    """sqrt(max(x, 0)^2 + c^2) - c"""

    def __init__(self, x, c: float = 1e-8):
        self.x = to_expr(x)
        self.c = float(c)

    def children(self):
        return [self.x]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve c parameter."""
        x = self.x.canonicalize()
        return SmoothReLU(x, c=self.c)

    def check_shape(self) -> Tuple[int, ...]:
        """Smooth ReLU preserves the shape of x."""
        return self.x.check_shape()

    def __repr__(self):
        return f"smooth_relu({self.x!r}, c={self.c})"
