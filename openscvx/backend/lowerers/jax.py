import jax.numpy as jnp

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Constant,
    Constraint,
    Expr,
    MatMul,
    Mul,
    Neg,
)
from openscvx.backend.state import State

# openscvx/backend/lowerers/jax.py


def lower(expr: Expr, lowerer: "JaxLowerer"):
    """
    Look up `lowerer.visit_<nodename>` and call it.
    """
    method = getattr(lowerer, f"visit_{expr.__class__.__name__.lower()}")
    return method(expr)


class JaxLowerer:  # openscvx/backend/lowerers/jax.py
    def visit_constant(self, node: Constant):
        # capture the constant value once
        value = jnp.array(node.value)
        return lambda x, u: value

    def visit_state(self, node: State):
        sl = node._slice
        if sl is None:
            raise ValueError(f"State {node.name!r} has no slice assigned")
        return lambda x, u: x[sl]

    def visit_control(self, node: Control):
        sl = node._slice
        if sl is None:
            raise ValueError(f"Control {node.name!r} has no slice assigned")
        return lambda x, u: u[sl]

    def visit_add(self, node: Add):
        fL = lower(node.left, self)
        fR = lower(node.right, self)
        return lambda x, u: fL(x, u) + fR(x, u)

    def visit_sub(self, node: Add):
        fL = lower(node.left, self)
        fR = lower(node.right, self)
        return lambda x, u: fL(x, u) - fR(x, u)

    def visit_mul(self, node: Mul):
        fL = lower(node.left, self)
        fR = lower(node.right, self)
        return lambda x, u: fL(x, u) * fR(x, u)

    def visit_div(self, node: Mul):
        fL = lower(node.left, self)
        fR = lower(node.right, self)
        return lambda x, u: fL(x, u) / fR(x, u)

    def visit_matmul(self, node: MatMul):
        fL = lower(node.left, self)
        fR = lower(node.right, self)
        return lambda x, u: jnp.matmul(fL(x, u), fR(x, u))

    def visit_neg(self, node: Neg):
        fO = lower(node.operand, self)
        return lambda x, u: -fO(x, u)

    def visit_constraint(self, node: Constraint):
        fL = lower(node.lhs, self)
        fR = lower(node.rhs, self)

        if node.op == "<=":
            return lambda x, u: fL(x, u) <= fR(x, u)
        elif node.op == ">=":
            return lambda x, u: fL(x, u) >= fR(x, u)
        else:  # "=="
            return lambda x, u: fL(x, u) == fR(x, u)
