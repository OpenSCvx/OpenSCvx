from typing import Any, Callable, Dict, Type

import numpy as np

_SHAPE_VISITORS: Dict[Type[Any], Callable[[Any], tuple[int, ...]]] = {}


def shape_visitor(expr_cls: Type[Any]):
    """Decorator to register a shape visitor for an AST node type."""

    def register(fn: Callable[[Any], tuple[int, ...]]):
        _SHAPE_VISITORS[expr_cls] = fn
        return fn

    return register


def check_shape(expr: Any) -> tuple[int, ...]:
    """Check the shape of an expression using the registered visitors."""
    fn = _SHAPE_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(f"No shape rule for {type(expr).__name__}")
    return fn(expr)


def validate_shapes(exprs):
    """Validate shapes for a single expression or list of expressions."""
    exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    for e in exprs:
        check_shape(e)  # will raise ValueError if anything's wrong


def _broadcast_shape_for(node: Any) -> tuple[int, ...]:
    """Helper function to broadcast shapes for nodes with multiple children."""
    # gather all child shapes
    shapes = [check_shape(child) for child in node.children()]
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError as e:
        op = type(node).__name__
        raise ValueError(f"{op} shapes not broadcastable: {shapes}") from e


# Legacy support - kept for backward compatibility
def dispatch(expr: Any) -> tuple[int, ...]:
    return check_shape(expr)
