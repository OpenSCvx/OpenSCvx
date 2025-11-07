from typing import Any, Sequence, Union

from openscvx.symbolic.expr import Expr


def lower(expr: Expr, lowerer: Any):
    """
    Dispatch an Expr node to the appropriate visit_* method on the lowerer.
    """
    return lowerer.lower(expr)


# --- Convenience wrappers for common backends ---


def lower_to_jax(exprs: Union[Expr, Sequence[Expr]]) -> Union[callable, list[callable]]:
    """
    If `exprs` is a single Expr, returns a single callable.
    Otherwise (a list/tuple of Exprs), returns a list of callables.
    """
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    if isinstance(exprs, Expr):
        return lower(exprs, jl)
    fns = [lower(e, jl) for e in exprs]
    return fns
