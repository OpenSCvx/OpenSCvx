from typing import Any, Callable, Dict, Type

import jax.numpy as jnp

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Constraint,
    Div,
    Expr,
    Index,
    MatMul,
    Mul,
    Neg,
    Sub,
)
from openscvx.backend.state import State

_VISITORS: Dict[Type[Expr], Callable] = {}


def visitor(expr_cls: Type[Expr]):
    def register(fn: Callable[[Any, Expr], Callable]):
        _VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: Any, expr: Expr):
    fn = _VISITORS.get(type(expr))
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
        return lambda x, u: value

    @visitor(State)
    def visit_state(self, node: State):
        sl = node._slice
        if sl is None:
            raise ValueError(f"State {node.name!r} has no slice assigned")
        return lambda x, u: x[sl]

    @visitor(Control)
    def visit_control(self, node: Control):
        sl = node._slice
        if sl is None:
            raise ValueError(f"Control {node.name!r} has no slice assigned")
        return lambda x, u: u[sl]

    @visitor(Add)
    def visit_add(self, node: Add):
        fs = [self.lower(term) for term in node.terms]

        def fn(x, u):
            acc = fs[0](x, u)
            for f in fs[1:]:
                acc = acc + f(x, u)
            return acc

        return fn

    @visitor(Sub)
    def visit_sub(self, node: Sub):
        fs = [self.lower(term) for term in node.terms]

        def fn(x, u):
            acc = fs[0](x, u)
            for f in fs[1:]:
                acc = acc - f(x, u)
            return acc

        return fn

    @visitor(Mul)
    def visit_mul(self, node: Mul):
        fs = [self.lower(factor) for factor in node.factors]

        def fn(x, u):
            acc = fs[0](x, u)
            for f in fs[1:]:
                acc = acc * f(x, u)
            return acc

        return fn

    @visitor(Div)
    def visit_div(self, node: Div):
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u: fL(x, u) / fR(x, u)

    @visitor(MatMul)
    def visit_matmul(self, node: MatMul):
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u: jnp.matmul(fL(x, u), fR(x, u))

    @visitor(Neg)
    def visit_neg(self, node: Neg):
        fO = self.lower(node.operand)
        return lambda x, u: -fO(x, u)

    @visitor(Index)
    def visit_index(self, node: Index):
        # lower the “base” expr into a fn(x,u), then index it
        f_base = self.lower(node.base)
        idx = node.index
        return lambda x, u: jnp.atleast_1d(f_base(x, u))[idx]

    @visitor(Concat)
    def visit_concat(self, node: Concat):
        # lower each child
        fn_list = [self.lower(child) for child in node.exprs]

        # wrapper that promotes scalars to 1-D and concatenates
        def concat_fn(x, u):
            parts = [jnp.atleast_1d(fn(x, u)) for fn in fn_list]
            return jnp.concatenate(parts, axis=0)

        return concat_fn

    @visitor(Constraint)
    def visit_constraint(self, node: Constraint):
        fL = self.lower(node.lhs)
        fR = self.lower(node.rhs)

        if node.op == "<=":
            return lambda x, u: fL(x, u) <= fR(x, u)
        elif node.op == ">=":
            return lambda x, u: fL(x, u) >= fR(x, u)
        else:  # "=="
            return lambda x, u: fL(x, u) == fR(x, u)
