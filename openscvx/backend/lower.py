# openscvx/backend/lower.py

from typing import Any, Sequence

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


def lower_to_jax(
    exprs: Sequence[Expr],
):
    """
    Lower one or more Exprs into callable JAX functions (x,u)->...
    """
    from openscvx.backend.lowerers.jax import JaxLowerer

    jl = JaxLowerer()
    return [lower(expr, jl) for expr in exprs]
