"""Linear algebra operations for symbolic expressions."""

from openscvx.backend.expr import (
    Diag,
    Hstack,
    MatMul,  # Also include MatMul here since it's linear algebra
    Norm,
    Stack,
    Transpose,
    Vstack,
)

__all__ = ["Transpose", "Stack", "Hstack", "Vstack", "Norm", "Diag", "MatMul"]
