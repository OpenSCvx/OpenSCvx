from typing import Any, Callable, Dict, Type

import cvxpy as cp

from openscvx.symbolic.expr import (
    CTCS,
    Add,
    Concat,
    Constant,
    Cos,
    Div,
    Equality,
    Exp,
    Expr,
    Huber,
    Index,
    Inequality,
    Linterp,
    Log,
    MatMul,
    Max,
    Mul,
    Neg,
    Norm,
    Parameter,
    PositivePart,
    Power,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    Stack,
    Sub,
    Sum,
    Transpose,
)
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State

_CVXPY_VISITORS: Dict[Type[Expr], Callable] = {}


def visitor(expr_cls: Type[Expr]):
    def register(fn: Callable[[Any, Expr], cp.Expression]):
        _CVXPY_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: Any, expr: Expr):
    fn = _CVXPY_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{lowerer.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(lowerer, expr)


class CvxpyLowerer:
    """
    Lowers symbolic expressions to CVXPy expressions.

    CVXPy variables must be created externally and passed in during initialization.
    The lowerer assumes variables are already properly shaped and indexed.
    """

    def __init__(self, variable_map: Dict[str, cp.Expression] = None):
        """
        Initialize the CVXPy lowerer.

        Args:
            variable_map: Dictionary mapping variable names to CVXPy expressions.
                         For State/Control objects, keys should match their names.
        """
        self.variable_map = variable_map or {}

    def lower(self, expr: Expr) -> cp.Expression:
        """Lower a symbolic expression to a CVXPy expression."""
        return dispatch(self, expr)

    def register_variable(self, name: str, cvx_expr: cp.Expression):
        """Register a CVXPy variable/expression for use in lowering."""
        self.variable_map[name] = cvx_expr

    @visitor(Constant)
    def visit_constant(self, node: Constant) -> cp.Expression:
        return cp.Constant(node.value)

    @visitor(State)
    def visit_state(self, node: State) -> cp.Expression:
        if "x" not in self.variable_map:
            raise ValueError("State vector 'x' not found in variable_map.")

        cvx_var = self.variable_map["x"]

        # If the state has a slice assigned, apply it
        if node._slice is not None:
            return cvx_var[node._slice]
        return cvx_var

    @visitor(Control)
    def visit_control(self, node: Control) -> cp.Expression:
        if "u" not in self.variable_map:
            raise ValueError("Control vector 'u' not found in variable_map.")

        cvx_var = self.variable_map["u"]

        # If the control has a slice assigned, apply it
        if node._slice is not None:
            return cvx_var[node._slice]
        return cvx_var

    @visitor(Parameter)
    def visit_parameter(self, node: Parameter) -> cp.Expression:
        param_name = node.name
        if param_name in self.variable_map:
            return self.variable_map[param_name]
        else:
            raise ValueError(
                f"Parameter '{param_name}' not found in variable_map. "
                f"Add it during CVXPy lowering or use cp.Parameter for parameter sweeps."
            )

    @visitor(Add)
    def visit_add(self, node: Add) -> cp.Expression:
        terms = [self.lower(term) for term in node.terms]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    @visitor(Sub)
    def visit_sub(self, node: Sub) -> cp.Expression:
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left - right

    @visitor(Mul)
    def visit_mul(self, node: Mul) -> cp.Expression:
        factors = [self.lower(factor) for factor in node.factors]
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result

    @visitor(Div)
    def visit_div(self, node: Div) -> cp.Expression:
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left / right

    @visitor(MatMul)
    def visit_matmul(self, node: MatMul) -> cp.Expression:
        left = self.lower(node.left)
        right = self.lower(node.right)
        return left @ right

    @visitor(Neg)
    def visit_neg(self, node: Neg) -> cp.Expression:
        operand = self.lower(node.operand)
        return -operand

    @visitor(Sum)
    def visit_sum(self, node: Sum) -> cp.Expression:
        operand = self.lower(node.operand)
        return cp.sum(operand)

    @visitor(Norm)
    def visit_norm(self, node: Norm) -> cp.Expression:
        operand = self.lower(node.operand)
        return cp.norm(operand, node.ord)

    @visitor(Index)
    def visit_index(self, node: Index) -> cp.Expression:
        base = self.lower(node.base)
        return base[node.index]

    @visitor(Concat)
    def visit_concat(self, node: Concat) -> cp.Expression:
        exprs = [self.lower(child) for child in node.exprs]
        # Ensure all expressions are at least 1D for concatenation
        exprs_1d = []
        for expr in exprs:
            if expr.ndim == 0:  # scalar
                exprs_1d.append(cp.reshape(expr, (1,), order="C"))
            else:
                exprs_1d.append(expr)
        return cp.hstack(exprs_1d)

    @visitor(Sin)
    def visit_sin(self, node: Sin) -> cp.Expression:
        # CVXPy doesn't support trigonometric functions in DCP form
        raise NotImplementedError(
            "Trigonometric functions like Sin are not DCP-compliant in CVXPy. "
            "Consider using piecewise-linear approximations or handle these constraints "
            "in the dynamics (JAX) layer instead."
        )

    @visitor(Cos)
    def visit_cos(self, node: Cos) -> cp.Expression:
        # CVXPy doesn't support trigonometric functions in DCP form
        raise NotImplementedError(
            "Trigonometric functions like Cos are not DCP-compliant in CVXPy. "
            "Consider using piecewise-linear approximations or handle these constraints "
            "in the dynamics (JAX) layer instead."
        )

    @visitor(Exp)
    def visit_exp(self, node: Exp) -> cp.Expression:
        operand = self.lower(node.operand)
        # Exponential is convex, so it's DCP-compliant when used appropriately
        return cp.exp(operand)

    @visitor(Log)
    def visit_log(self, node: Log) -> cp.Expression:
        operand = self.lower(node.operand)
        # Logarithm is concave, so it's DCP-compliant when used appropriately
        return cp.log(operand)

    @visitor(Equality)
    def visit_equality(self, node: Equality) -> cp.Constraint:
        left = self.lower(node.lhs)
        right = self.lower(node.rhs)
        return left == right

    @visitor(Inequality)
    def visit_inequality(self, node: Inequality) -> cp.Constraint:
        left = self.lower(node.lhs)
        right = self.lower(node.rhs)
        return left <= right

    @visitor(CTCS)
    def visit_ctcs(self, node: CTCS) -> cp.Expression:
        raise NotImplementedError(
            "CTCS constraints are for continuous-time constraint satisfaction and "
            "should be handled through dynamics augmentation with JAX lowering, "
            "not CVXPy lowering. CTCS constraints represent non-convex dynamics "
            "augmentation."
        )

    @visitor(PositivePart)
    def visit_pos(self, node: PositivePart) -> cp.Expression:
        operand = self.lower(node.x)
        return cp.maximum(operand, 0.0)

    @visitor(Square)
    def visit_square(self, node: Square) -> cp.Expression:
        operand = self.lower(node.x)
        return cp.square(operand)

    @visitor(Huber)
    def visit_huber(self, node: Huber) -> cp.Expression:
        operand = self.lower(node.x)
        return cp.huber(operand, M=node.delta)

    @visitor(SmoothReLU)
    def visit_srelu(self, node: SmoothReLU) -> cp.Expression:
        operand = self.lower(node.x)
        c = node.c
        # smooth_relu(x) = sqrt(max(x, 0)^2 + c^2) - c
        pos_part = cp.maximum(operand, 0.0)
        # For SmoothReLU, we use the 2-norm formulation
        return cp.sqrt(cp.sum_squares(pos_part) + c**2) - c

    @visitor(Sqrt)
    def visit_sqrt(self, node: Sqrt) -> cp.Expression:
        operand = self.lower(node.operand)
        return cp.sqrt(operand)

    @visitor(Max)
    def visit_max(self, node: Max) -> cp.Expression:
        operands = [self.lower(op) for op in node.operands]
        # CVXPy's maximum can take multiple arguments
        if len(operands) == 2:
            return cp.maximum(operands[0], operands[1])
        else:
            # For more than 2 operands, chain maximum calls
            result = cp.maximum(operands[0], operands[1])
            for op in operands[2:]:
                result = cp.maximum(result, op)
            return result

    @visitor(Transpose)
    def visit_transpose(self, node: Transpose) -> cp.Expression:
        operand = self.lower(node.operand)
        return operand.T

    @visitor(Power)
    def visit_power(self, node: Power) -> cp.Expression:
        base = self.lower(node.base)
        exponent = self.lower(node.exponent)
        return cp.power(base, exponent)

    @visitor(Stack)
    def visit_stack(self, node: Stack) -> cp.Expression:
        rows = [self.lower(row) for row in node.rows]
        # Stack rows vertically
        return cp.vstack(rows)

    @visitor(Linterp)
    def visit_linterp(self, node: Linterp) -> cp.Expression:
        raise NotImplementedError(
            "Linear interpolation (Linterp) is not DCP-compliant in CVXPy. "
            "Interpolation should be handled in the dynamics (JAX) layer. "
            "If you need interpolated values in constraints, consider linearizing "
            "around the current trajectory in the SCP iteration."
        )


def lower_to_cvxpy(expr: Expr, variable_map: Dict[str, cp.Expression] = None) -> cp.Expression:
    """
    Convenience function to lower a single expression to CVXPy.

    Args:
        expr: Expression to lower
        variable_map: Dictionary mapping variable names to CVXPy expressions

    Returns:
        CVXPy expression or constraint

    Example:
        >>> import cvxpy as cp
        >>> from openscvx.backend.expr import *
        >>> from openscvx.backend.state import State
        >>>
        >>> # Create CVXPy variables
        >>> x_var = cp.Variable((10, 3), name="x")  # 10 time steps, 3 states
        >>>
        >>> # Create symbolic state
        >>> x = State("x", shape=(3,))
        >>>
        >>> # Create expression: x + 1
        >>> expr = x + 1
        >>>
        >>> # Lower to CVXPy
        >>> cvx_expr = lower_to_cvxpy(expr, {"x": x_var})
    """
    lowerer = CvxpyLowerer(variable_map)
    return lowerer.lower(expr)
