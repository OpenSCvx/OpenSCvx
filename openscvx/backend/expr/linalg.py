from typing import Tuple

import numpy as np

from .expr import Expr, to_expr


class Transpose(Expr):
    """Matrix transpose operation"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand with double transpose optimization."""
        operand = self.operand.canonicalize()

        # Double transpose optimization: (A.T).T = A
        if isinstance(operand, Transpose):
            return operand.operand

        return Transpose(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Matrix transpose operation swaps the last two dimensions."""
        operand_shape = self.operand.check_shape()

        if len(operand_shape) == 0:
            # Scalar transpose is the scalar itself
            return ()
        elif len(operand_shape) == 1:
            # Vector transpose is the vector itself (row vector remains row vector)
            return operand_shape
        elif len(operand_shape) == 2:
            # Matrix transpose: (m,n) -> (n,m)
            return (operand_shape[1], operand_shape[0])
        else:
            # Higher-dimensional array: transpose last two dimensions
            # (..., m, n) -> (..., n, m)
            return operand_shape[:-2] + (operand_shape[-1], operand_shape[-2])

    def __repr__(self):
        return f"({self.operand!r}).T"


class Diag(Expr):
    """Create diagonal matrix from vector or extract diagonal from matrix"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand."""
        operand = self.operand.canonicalize()
        return Diag(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Diag converts a vector (n,) to a diagonal matrix (n,n)."""
        operand_shape = self.operand.check_shape()
        if len(operand_shape) != 1:
            raise ValueError(f"Diag expects a 1D vector, got shape {operand_shape}")
        n = operand_shape[0]
        return (n, n)

    def __repr__(self):
        return f"diag({self.operand!r})"


class Norm(Expr):
    """Norm of an expression (reduction operation)"""

    def __init__(self, operand, ord="fro"):
        self.operand = to_expr(operand)
        self.ord = ord  # Can be "fro", "inf", 1, 2, etc.

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize the operand but preserve the ord parameter."""
        canon_operand = self.operand.canonicalize()
        return Norm(canon_operand, ord=self.ord)

    def check_shape(self) -> Tuple[int, ...]:
        """Norm reduces any shape to a scalar."""
        # Validate that the operand has a valid shape
        self.operand.check_shape()
        # Norm always produces a scalar regardless of input shape
        return ()

    def __repr__(self):
        return f"norm({self.operand!r}, ord={self.ord!r})"


class Stack(Expr):
    """Stack expressions into a matrix - similar to jnp.array([[row1], [row2], ...])"""

    def __init__(self, rows):
        # rows should be a list of expressions representing each row
        self.rows = [to_expr(row) for row in rows]

    def children(self):
        return self.rows

    def canonicalize(self) -> "Expr":
        """Canonicalize all rows."""
        rows = [row.canonicalize() for row in self.rows]
        return Stack(rows)

    def check_shape(self) -> Tuple[int, ...]:
        """Stack creates a 2D matrix from 1D rows."""
        if not self.rows:
            raise ValueError("Stack requires at least one row")

        # All rows should have the same shape
        row_shapes = [row.check_shape() for row in self.rows]

        # Verify all rows have the same shape
        first_shape = row_shapes[0]
        for i, shape in enumerate(row_shapes[1:], 1):
            if shape != first_shape:
                raise ValueError(
                    f"Stack row {i} has shape {shape}, but row 0 has shape {first_shape}"
                )

        # Result shape is (num_rows, *row_shape)
        return (len(self.rows),) + first_shape

    def __repr__(self):
        rows_repr = ", ".join(repr(row) for row in self.rows)
        return f"Stack([{rows_repr}])"


class Hstack(Expr):
    """Horizontal stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def canonicalize(self) -> "Expr":
        """Canonicalize all arrays."""
        arrays = [arr.canonicalize() for arr in self.arrays]
        return Hstack(arrays)

    def check_shape(self) -> Tuple[int, ...]:
        """Horizontal stack concatenates arrays along the second axis (columns)."""
        if not self.arrays:
            raise ValueError("Hstack requires at least one array")

        array_shapes = [arr.check_shape() for arr in self.arrays]

        # All arrays must have the same number of dimensions
        first_ndim = len(array_shapes[0])
        for i, shape in enumerate(array_shapes[1:], 1):
            if len(shape) != first_ndim:
                raise ValueError(
                    f"Hstack array {i} has {len(shape)} dimensions, but array 0 has {first_ndim}"
                )

        # For 1D arrays, hstack concatenates along axis 0
        if first_ndim == 1:
            total_length = sum(shape[0] for shape in array_shapes)
            return (total_length,)

        # For 2D+ arrays, all dimensions except the second must match
        first_shape = array_shapes[0]
        for i, shape in enumerate(array_shapes[1:], 1):
            if shape[0] != first_shape[0]:
                raise ValueError(
                    f"Hstack array {i} has {shape[0]} rows, but array 0 has {first_shape[0]} rows"
                )
            if shape[2:] != first_shape[2:]:
                raise ValueError(
                    f"Hstack array {i} has trailing dimensions {shape[2:]}, but array 0 has {first_shape[2:]}"
                )

        # Result shape: concatenate along axis 1 (columns)
        total_cols = sum(shape[1] for shape in array_shapes)
        return (first_shape[0], total_cols) + first_shape[2:]

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Hstack([{arrays_repr}])"


class Vstack(Expr):
    """Vertical stack"""

    def __init__(self, arrays):
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def canonicalize(self) -> "Expr":
        """Canonicalize all arrays."""
        arrays = [arr.canonicalize() for arr in self.arrays]
        return Vstack(arrays)

    def check_shape(self) -> Tuple[int, ...]:
        """Vertical stack concatenates arrays along the first axis (rows)."""
        if not self.arrays:
            raise ValueError("Vstack requires at least one array")

        array_shapes = [arr.check_shape() for arr in self.arrays]

        # All arrays must have the same number of dimensions
        first_ndim = len(array_shapes[0])
        for i, shape in enumerate(array_shapes[1:], 1):
            if len(shape) != first_ndim:
                raise ValueError(
                    f"Vstack array {i} has {len(shape)} dimensions, but array 0 has {first_ndim}"
                )

        # All dimensions except the first must match
        first_shape = array_shapes[0]
        for i, shape in enumerate(array_shapes[1:], 1):
            if shape[1:] != first_shape[1:]:
                raise ValueError(
                    f"Vstack array {i} has trailing dimensions {shape[1:]}, but array 0 has {first_shape[1:]}"
                )

        # Result shape: concatenate along axis 0 (rows)
        total_rows = sum(shape[0] for shape in array_shapes)
        return (total_rows,) + first_shape[1:]

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Vstack([{arrays_repr}])"
