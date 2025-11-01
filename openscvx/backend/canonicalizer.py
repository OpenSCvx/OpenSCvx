from typing import Any, Callable, Dict, Type

_CANON_VISITORS: Dict[Type[Any], Callable[[Any], Any]] = {}


def canon_visitor(expr_cls: Type[Any]):
    """Decorator to register a canonicalization visitor for an AST node type."""

    def register(fn: Callable[[Any], Any]):
        _CANON_VISITORS[expr_cls] = fn
        return fn

    return register


def canonicalize(expr: Any) -> Any:
    """Canonicalize an expression using the registered visitors."""
    fn = _CANON_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(f"No canonicalization rule for {type(expr).__name__}")
    return fn(expr)
