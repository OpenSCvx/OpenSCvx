from ..canonicalizer import canon_visitor, canonicalize
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
