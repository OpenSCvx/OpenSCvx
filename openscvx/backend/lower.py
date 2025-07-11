from typing import Any, Sequence, Union

from openscvx.backend.expr import Expr


def lower(expr: Expr, lowerer: Any):
    """
    Dispatch an Expr node to the appropriate visit_* method on the lowerer.
    """
    method_name = f"visit_{expr.__class__.__name__.lower()}"
    if not hasattr(lowerer, method_name):
        raise AttributeError(f"{lowerer.__class__.__name__!r} has no method {method_name}")
    method = getattr(lowerer, method_name)
    return method(expr)


# --- Convenience wrappers for common backends ---


def lower_to_jax(exprs: Union[Expr, Sequence[Expr]]) -> Union[callable, list[callable]]:
    """
    If `exprs` is a single Expr, returns a single callable.
    Otherwise (a list/tuple of Exprs), returns a list of callables.
    """
    from openscvx.backend.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    if isinstance(exprs, Expr):
        return lower(exprs, jl)
    fns = [lower(e, jl) for e in exprs]
    return fns
