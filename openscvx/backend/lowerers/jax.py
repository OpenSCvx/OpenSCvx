from typing import Any, Callable, Dict, Type

import jax.numpy as jnp
from jax.lax import cond

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    CTCS,
    Add,
    Concat,
    Constant,
    Constraint,
    Cos,
    Div,
    Equality,
    Expr,
    Huber,
    Index,
    Inequality,
    MatMul,
    Mul,
    Neg,
    PositivePart,
    Sin,
    SmoothReLU,
    Square,
    Sub,
    Sum,
)
from openscvx.backend.state import State

_JAX_VISITORS: Dict[Type[Expr], Callable] = {}


def visitor(expr_cls: Type[Expr]):
    def register(fn: Callable[[Any, Expr], Callable]):
        _JAX_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: Any, expr: Expr):
    fn = _JAX_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{lowerer.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(lowerer, expr)


class JaxLowerer:
    def lower(self, expr: Expr):
        return dispatch(self, expr)

    @visitor(Constant)
    def visit_constant(self, node: Constant):
        # capture the constant value once
        value = jnp.array(node.value)
        return lambda x, u, **kwargs: value

    @visitor(State)
    def visit_state(self, node: State):
        sl = node._slice
        if sl is None:
            raise ValueError(f"State {node.name!r} has no slice assigned")
        return lambda x, u, **kwargs: x[sl]

    @visitor(Control)
    def visit_control(self, node: Control):
        sl = node._slice
        if sl is None:
            raise ValueError(f"Control {node.name!r} has no slice assigned")
        return lambda x, u, **kwargs: u[sl]

    @visitor(Add)
    def visit_add(self, node: Add):
        fs = [self.lower(term) for term in node.terms]

        def fn(x, u, **kwargs):
            acc = fs[0](x, u, **kwargs)
            for f in fs[1:]:
                acc = acc + f(x, u, **kwargs)
            return acc

        return fn

    @visitor(Sub)
    def visit_sub(self, node: Sub):
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, **kwargs: fL(x, u, **kwargs) - fR(x, u, **kwargs)

    @visitor(Mul)
    def visit_mul(self, node: Mul):
        fs = [self.lower(factor) for factor in node.factors]

        def fn(x, u, **kwargs):
            acc = fs[0](x, u, **kwargs)
            for f in fs[1:]:
                acc = acc * f(x, u, **kwargs)
            return acc

        return fn

    @visitor(Div)
    def visit_div(self, node: Div):
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, **kwargs: fL(x, u, **kwargs) / fR(x, u, **kwargs)

    @visitor(MatMul)
    def visit_matmul(self, node: MatMul):
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, **kwargs: jnp.matmul(fL(x, u, **kwargs), fR(x, u, **kwargs))

    @visitor(Neg)
    def visit_neg(self, node: Neg):
        fO = self.lower(node.operand)
        return lambda x, u, **kwargs: -fO(x, u, **kwargs)

    @visitor(Sum)
    def visit_sum(self, node: Sum):
        f = self.lower(node.operand)
        return lambda x, u, **kwargs: jnp.sum(f(x, u, **kwargs))

    @visitor(Index)
    def visit_index(self, node: Index):
        # lower the "base" expr into a fn(x,u), then index it
        f_base = self.lower(node.base)
        idx = node.index
        return lambda x, u, **kwargs: jnp.atleast_1d(f_base(x, u, **kwargs))[idx]

    @visitor(Concat)
    def visit_concat(self, node: Concat):
        # lower each child
        fn_list = [self.lower(child) for child in node.exprs]

        # wrapper that promotes scalars to 1-D and concatenates
        def concat_fn(x, u, **kwargs):
            parts = [jnp.atleast_1d(fn(x, u, **kwargs)) for fn in fn_list]
            return jnp.concatenate(parts, axis=0)

        return concat_fn

    @visitor(Sin)
    def visit_sin(self, node: Sin):
        fO = self.lower(node.operand)
        return lambda x, u, **kwargs: jnp.sin(fO(x, u, **kwargs))

    @visitor(Cos)
    def visit_cos(self, node: Cos):
        fO = self.lower(node.operand)
        return lambda x, u, **kwargs: jnp.cos(fO(x, u, **kwargs))

    @visitor(Equality)
    @visitor(Inequality)
    def visit_constraint(self, node: Constraint):
        """Lower equality constraint: lhs == rhs or lhs <= rhs becomes lhs - rhs"""
        fL = self.lower(node.lhs)
        fR = self.lower(node.rhs)
        return lambda x, u, **kwargs: fL(x, u, **kwargs) - fR(x, u, **kwargs)

    @visitor(CTCS)
    def visit_ctcs(self, node: CTCS):
        raise RuntimeError(
            "CTCS constraint should not be lowered directly. "
            "It should be processed during augmentation phase."
        )

    @visitor(PositivePart)
    def visit_pos(self, node):
        f = self.lower(node.x)
        return lambda x, u, **kwargs: jnp.maximum(f(x, u, **kwargs), 0.0)

    @visitor(Square)
    def visit_square(self, node):
        f = self.lower(node.x)
        return lambda x, u, **kwargs: f(x, u, **kwargs) * f(x, u, **kwargs)

    @visitor(Huber)
    def visit_huber(self, node):
        f = self.lower(node.x)
        delta = node.delta
        return lambda x, u, **kwargs: jnp.where(
            jnp.abs(f(x, u, **kwargs)) <= delta,
            0.5 * f(x, u, **kwargs) ** 2,
            delta * (jnp.abs(f(x, u, **kwargs)) - 0.5 * delta),
        )

    @visitor(SmoothReLU)
    def visit_srelu(self, node):
        f = self.lower(node.x)
        c = node.c
        # smooth_relu(pos(x)) = sqrt(pos(x)^2 + c^2) - c ; here f already includes pos inside node
        return lambda x, u, **kwargs: jnp.sqrt(jnp.maximum(f(x, u, **kwargs), 0.0) ** 2 + c**2) - c
