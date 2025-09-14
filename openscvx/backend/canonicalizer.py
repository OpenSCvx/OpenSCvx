from typing import Any, Callable, Dict, Type, Union

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    CTCS,
    Add,
    Concat,
    Constant,
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
    NodalConstraint,
    Norm,
    PositivePart,
    Sin,
    SmoothReLU,
    Square,
    Sub,
    Sum,
)
from openscvx.backend.state import State

_CANON_VISITORS: Dict[Type[Expr], Callable] = {}


def canonicalize(expr: Expr) -> Expr:
    """
    Dispatch an Expr node (or tree) through the visitor-based
    Canonicalizer.  If you don’t pass in your own Canonicalizer
    instance, a fresh one will be used.
    """
    canonizer = Canonicalizer()
    return canonizer.canonicalize(expr)


def visitor(expr_cls: Type[Expr]):
    def register(fn: Callable[[Any, Expr], Expr]):
        _CANON_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(canon: "Canonicalizer", expr: Expr) -> Expr:
    fn = _CANON_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{canon.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(canon, expr)


class Canonicalizer:
    """Visitor-based canonicalizer for Expr trees."""

    def canonicalize(self, expr: Expr) -> Expr:
        return dispatch(self, expr)

    @visitor(State)
    @visitor(Control)
    @visitor(Constant)
    def visit_leaf(self, node: Union[State, Control]) -> Expr:
        # leaf nodes are already canonical
        return node

    @visitor(Add)
    def visit_add(self, node: Add) -> Expr:
        # flatten, recurse, fold, eliminate zero, collapse singleton
        terms: list[Expr] = []
        const_vals: list[np.ndarray] = []

        for t in node.terms:
            c = self.canonicalize(t)
            if isinstance(c, Add):
                terms.extend(c.terms)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                terms.append(c)

        if const_vals:
            total = sum(const_vals)
            # if not all-zero, keep it
            if not (isinstance(total, np.ndarray) and np.all(total == 0)):
                terms.append(Constant(total))

        if not terms:
            return Constant(np.array(0))
        if len(terms) == 1:
            return terms[0]
        return Add(*terms)

    @visitor(Mul)
    def visit_mul(self, node: Mul) -> Expr:
        factors: list[Expr] = []
        const_vals: list[np.ndarray] = []

        for f in node.factors:
            c = self.canonicalize(f)
            if isinstance(c, Mul):
                factors.extend(c.factors)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                factors.append(c)

        if const_vals:
            prod = np.prod(const_vals)
            # if prod != 1, keep it
            if not (isinstance(prod, np.ndarray) and np.all(prod == 1)):
                factors.append(Constant(prod))

        if not factors:
            return Constant(np.array(1))
        if len(factors) == 1:
            return factors[0]
        return Mul(*factors)

    @visitor(Sub)
    def visit_sub(self, node: Sub) -> Expr:
        # canonicalize children, but keep as binary
        left = self.canonicalize(node.left)
        right = self.canonicalize(node.right)
        # maybe special-case Constant-Constant?
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value - right.value)
        return Sub(left, right)

    @visitor(Div)
    def visit_div(self, node: Div) -> Expr:
        lhs = self.canonicalize(node.left)
        rhs = self.canonicalize(node.right)
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            return Constant(lhs.value / rhs.value)
        return Div(lhs, rhs)

    @visitor(Neg)
    def visit_neg(self, node: Neg) -> Expr:
        o = self.canonicalize(node.operand)
        if isinstance(o, Constant):
            return Constant(-o.value)
        return Neg(o)

    @visitor(Concat)
    def visit_concat(self, node: Concat) -> Expr:
        exprs = [self.canonicalize(e) for e in node.exprs]
        return Concat(*exprs)

    @visitor(Index)
    def visit_index(self, node: Index) -> Expr:
        base = self.canonicalize(node.base)
        return Index(base, node.index)

    @visitor(Inequality)
    def visit_inequality(self, node: Inequality) -> Expr:
        diff = Sub(node.lhs, node.rhs)
        canon_diff = self.canonicalize(diff)
        return Inequality(canon_diff, Constant(np.array(0)))

    @visitor(Equality)
    def visit_equality(self, node: Equality) -> Expr:
        diff = Sub(node.lhs, node.rhs)
        canon_diff = self.canonicalize(diff)
        return Equality(canon_diff, Constant(np.array(0)))

    @visitor(Norm)
    def visit_norm(self, node: Norm) -> Expr:
        # Canonicalize the operand but preserve the ord parameter
        canon_operand = self.canonicalize(node.operand)
        return Norm(canon_operand, ord=node.ord)

    @visitor(NodalConstraint)
    def visit_nodal_constraint(self, node: NodalConstraint) -> Expr:
        # Canonicalize the wrapped constraint and preserve the node specification
        canon_constraint = self.canonicalize(node.constraint)
        return NodalConstraint(canon_constraint, node.nodes)

    @visitor(MatMul)
    def visit_matmul(self, node: MatMul) -> Expr:
        # Canonicalize operands but preserve the operation
        left = self.canonicalize(node.left)
        right = self.canonicalize(node.right)
        return MatMul(left, right)

    @visitor(Sum)
    def visit_sum(self, node: Sum) -> Expr:
        # Canonicalize the operand
        operand = self.canonicalize(node.operand)
        return Sum(operand)

    @visitor(Sin)
    def visit_sin(self, node: Sin) -> Expr:
        # Canonicalize the operand
        operand = self.canonicalize(node.operand)
        return Sin(operand)

    @visitor(Cos)
    def visit_cos(self, node: Cos) -> Expr:
        # Canonicalize the operand
        operand = self.canonicalize(node.operand)
        return Cos(operand)

    @visitor(PositivePart)
    def visit_positive_part(self, node: PositivePart) -> Expr:
        # Canonicalize the operand
        x = self.canonicalize(node.x)
        return PositivePart(x)

    @visitor(Square)
    def visit_square(self, node: Square) -> Expr:
        # Canonicalize the operand
        x = self.canonicalize(node.x)
        return Square(x)

    @visitor(Huber)
    def visit_huber(self, node: Huber) -> Expr:
        # Canonicalize the operand but preserve delta parameter
        x = self.canonicalize(node.x)
        return Huber(x, delta=node.delta)

    @visitor(SmoothReLU)
    def visit_smooth_relu(self, node: SmoothReLU) -> Expr:
        # Canonicalize the operand but preserve c parameter
        x = self.canonicalize(node.x)
        return SmoothReLU(x, c=node.c)

    @visitor(CTCS)
    def visit_ctcs(self, node: CTCS) -> Expr:
        # Canonicalize the inner constraint but preserve CTCS parameters
        canon_constraint = self.canonicalize(node.constraint)
        return CTCS(canon_constraint, penalty=node.penalty, nodes=node.nodes)
