from typing import Tuple

import numpy as np

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


class Max(Expr):
    """Maximum of two or more operands: max(a, b, c, ...)"""

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Max requires two or more operands")
        self.operands = [to_expr(a) for a in args]

    def children(self):
        return list(self.operands)

    def canonicalize(self) -> "Expr":
        """Canonicalize max: flatten nested Max, fold constants."""
        from .expr import Constant

        operands = []
        const_vals = []

        for op in self.operands:
            c = op.canonicalize()
            if isinstance(c, Max):
                operands.extend(c.operands)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                operands.append(c)

        # If we have constants, compute their max and keep it
        if const_vals:
            max_const = np.maximum.reduce(const_vals)
            operands.append(Constant(max_const))

        if not operands:
            raise ValueError("Max must have at least one operand after canonicalization")
        if len(operands) == 1:
            return operands[0]
        return Max(*operands)

    def check_shape(self) -> Tuple[int, ...]:
        """Max broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Max shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        inner = ", ".join(repr(op) for op in self.operands)
        return f"max({inner})"


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


class Linterp(Expr):
    """Linear interpolation node supporting 1D and 2D interpolation.

    1D interpolation: Linterp(xp, fp, x)
        - xp: x-coordinates of data points (constant array)
        - fp: y-coordinates of data points (constant array)
        - x: query point(s) (can be symbolic expression)

    2D interpolation: Linterp(xp, yp, fp, x, y)
        - xp: x-coordinates of data points (constant 1D array)
        - yp: y-coordinates of data points (constant 1D array)
        - fp: values at grid points (constant 2D array)
        - x: x query point(s) (can be symbolic expression)
        - y: y query point(s) (can be symbolic expression)
    """

    def __init__(self, *args):
        if len(args) == 3:
            # 1D interpolation: Linterp(xp, fp, x)
            self.xp = to_expr(args[0])
            self.fp = to_expr(args[1])
            self.x = to_expr(args[2])
            self.yp = None
            self.y = None
            self.ndim = 1
        elif len(args) == 5:
            # 2D interpolation: Linterp(xp, yp, fp, x, y)
            self.xp = to_expr(args[0])
            self.yp = to_expr(args[1])
            self.fp = to_expr(args[2])
            self.x = to_expr(args[3])
            self.y = to_expr(args[4])
            self.ndim = 2
        else:
            raise ValueError(
                f"Linterp requires either 3 arguments (1D) or 5 arguments (2D), got {len(args)}"
            )

    def children(self):
        if self.ndim == 1:
            return [self.xp, self.fp, self.x]
        else:
            return [self.xp, self.yp, self.fp, self.x, self.y]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operands."""
        if self.ndim == 1:
            xp = self.xp.canonicalize()
            fp = self.fp.canonicalize()
            x = self.x.canonicalize()
            return Linterp(xp, fp, x)
        else:
            xp = self.xp.canonicalize()
            yp = self.yp.canonicalize()
            fp = self.fp.canonicalize()
            x = self.x.canonicalize()
            y = self.y.canonicalize()
            return Linterp(xp, yp, fp, x, y)

    def check_shape(self) -> Tuple[int, ...]:
        """Output has the same shape as the query point(s)."""
        if self.ndim == 1:
            return self.x.check_shape()
        else:
            x_shape = self.x.check_shape()
            y_shape = self.y.check_shape()
            try:
                return np.broadcast_shapes(x_shape, y_shape)
            except ValueError as e:
                raise ValueError(
                    f"Linterp query shapes not broadcastable: {x_shape} vs {y_shape}"
                ) from e

    def __repr__(self):
        if self.ndim == 1:
            return f"linterp({self.xp!r}, {self.fp!r}, {self.x!r})"
        else:
            return f"linterp({self.xp!r}, {self.yp!r}, {self.fp!r}, {self.x!r}, {self.y!r})"
