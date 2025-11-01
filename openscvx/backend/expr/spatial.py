from ..canonicalizer import canon_visitor, canonicalize
from ..shape_checker import check_shape, shape_visitor
from .expr import Expr, to_expr


# Efficient 6DOF utility functions that directly map to JAX implementations
class QDCM(Expr):
    """Quaternion to Direction Cosine Matrix conversion"""

    def __init__(self, q):
        self.q = to_expr(q)

    def children(self):
        return [self.q]

    def __repr__(self):
        return f"qdcm({self.q!r})"


@canon_visitor(QDCM)
def canon_qdcm(node: QDCM) -> Expr:
    # Canonicalize the quaternion operand
    q = canonicalize(node.q)
    return QDCM(q)


@shape_visitor(QDCM)
def check_shape_qdcm(node: QDCM) -> tuple[int, ...]:
    """QDCM takes a quaternion (4,) and produces a 3x3 DCM"""
    q_shape = check_shape(node.q)
    if q_shape != (4,):
        raise ValueError(f"QDCM expects quaternion with shape (4,), got {q_shape}")
    return (3, 3)


class SSMP(Expr):
    """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""

    def __init__(self, w):
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def __repr__(self):
        return f"ssmp({self.w!r})"


@canon_visitor(SSMP)
def canon_ssmp(node: SSMP) -> Expr:
    # Canonicalize the angular velocity operand
    w = canonicalize(node.w)
    return SSMP(w)


@shape_visitor(SSMP)
def check_shape_ssmp(node: SSMP) -> tuple[int, ...]:
    """SSMP takes angular velocity (3,) and produces a 4x4 matrix"""
    w_shape = check_shape(node.w)
    if w_shape != (3,):
        raise ValueError(f"SSMP expects angular velocity with shape (3,), got {w_shape}")
    return (4, 4)


class SSM(Expr):
    """Angular rate to 3x3 skew symmetric matrix"""

    def __init__(self, w):
        self.w = to_expr(w)

    def children(self):
        return [self.w]

    def __repr__(self):
        return f"ssm({self.w!r})"


@canon_visitor(SSM)
def canon_ssm(node: SSM) -> Expr:
    # Canonicalize the angular velocity operand
    w = canonicalize(node.w)
    return SSM(w)


@shape_visitor(SSM)
def check_shape_ssm(node: SSM) -> tuple[int, ...]:
    """SSM takes angular velocity (3,) and produces a 3x3 matrix"""
    w_shape = check_shape(node.w)
    if w_shape != (3,):
        raise ValueError(f"SSM expects angular velocity with shape (3,), got {w_shape}")
    return (3, 3)
