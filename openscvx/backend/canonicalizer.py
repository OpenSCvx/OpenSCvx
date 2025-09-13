from typing import Any, Callable, Dict, Type, Union

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Div,
    Equality,
    Expr,
    Index,
    Inequality,
    Mul,
    Neg,
    Sub,
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
        # by default, just recurse into children and rebuild
        rebuilt_children = [canon.canonicalize(child) for child in expr.children()]
        # assumes each Expr has a constructor matching its fields
        return expr.__class__(*rebuilt_children)
    return fn(canon, expr)


class Canonicalizer:
    """Visitor-based canonicalizer for Expr trees."""

    def canonicalize(self, expr: Expr) -> Expr:
        return dispatch(self, expr)

    @visitor(State)
    @visitor(Control)
    @visitor(Constant)
    def visit_leaf(self, node: Union[State, Control, Constant]) -> Expr:
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
        return Inequality(canon_diff, Constant(np.array(0)), nodes=node.nodes)

    @visitor(Equality)
    def visit_equality(self, node: Equality) -> Expr:
        diff = Sub(node.lhs, node.rhs)
        canon_diff = self.canonicalize(diff)
        return Equality(canon_diff, Constant(np.array(0)), nodes=node.nodes)
