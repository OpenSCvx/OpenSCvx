from typing import Tuple

from .expr import Expr, to_expr


# Efficient 6DOF utility functions that directly map to JAX implementations
class QDCM(Expr):
    """Quaternion to Direction Cosine Matrix conversion"""

    def __init__(self, q):
        self.q = to_expr(q)

    def children(self):
        return [self.q]

    def canonicalize(self) -> "Expr":
        """Canonicalize the quaternion operand."""
        q = self.q.canonicalize()
        return QDCM(q)

    def check_shape(self) -> Tuple[int, ...]:
        """QDCM takes a quaternion (4,) and produces a 3x3 DCM."""
        q_shape = self.q.check_shape()
        if q_shape != (4,):
            raise ValueError(f"QDCM expects quaternion with shape (4,), got {q_shape}")
        return (3, 3)

    def __repr__(self):
        return f"qdcm({self.q!r})"


class SSMP(Expr):
    """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""

    def __init__(self, w):
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def canonicalize(self) -> "Expr":
        """Canonicalize the angular velocity operand."""
        w = self.w.canonicalize()
        return SSMP(w)

    def check_shape(self) -> Tuple[int, ...]:
        """SSMP takes angular velocity (3,) and produces a 4x4 matrix."""
        w_shape = self.w.check_shape()
        if w_shape != (3,):
            raise ValueError(f"SSMP expects angular velocity with shape (3,), got {w_shape}")
        return (4, 4)

    def __repr__(self):
        return f"ssmp({self.w!r})"


class SSM(Expr):
    """Angular rate to 3x3 skew symmetric matrix"""

    def __init__(self, w):
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def canonicalize(self) -> "Expr":
        """Canonicalize the angular velocity operand."""
        w = self.w.canonicalize()
        return SSM(w)

    def check_shape(self) -> Tuple[int, ...]:
        """SSM takes angular velocity (3,) and produces a 3x3 matrix."""
        w_shape = self.w.check_shape()
        if w_shape != (3,):
            raise ValueError(f"SSM expects angular velocity with shape (3,), got {w_shape}")
        return (3, 3)

    def __repr__(self):
        return f"ssm({self.w!r})"
