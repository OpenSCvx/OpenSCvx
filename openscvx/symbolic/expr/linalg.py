"""Linear algebra operations for symbolic expressions.

This module provides essential linear algebra operations for matrix and vector
manipulation in optimization problems. Operations follow NumPy/JAX conventions
for shapes and broadcasting behavior.

Key Operations:

- **Matrix Operations:**
    - `Transpose` - Matrix/tensor transposition (swaps last two dimensions)
    - `Diag` - Construct diagonal matrix from vector
- **Stacking and Concatenation:**
    - `Stack` - Stack expressions along a new dimension
    - `Hstack` - Horizontally stack matrices/vectors
    - `Vstack` - Vertically stack matrices/vectors
- **Norms:**
    - `Norm` - Euclidean (L2) norm of vectors/matrices

Example:
    Building rotation matrices and transformations::

        import openscvx as ox
        import numpy as np

        # Create a rotation matrix from angle
        theta = ox.Variable("theta", shape=(1,))
        R = ox.Stack(
            ox.Hstack(ox.Cos(theta), -ox.Sin(theta)),
            ox.Hstack(ox.Sin(theta), ox.Cos(theta))
        )

        # Transform a point
        point = ox.Variable("p", shape=(2,))
        rotated = R @ point

    Computing kinetic energy::

        v = ox.State("v", shape=(3,))  # Velocity vector
        m = 10.0  # Mass
        kinetic_energy = 0.5 * m * ox.Norm(v)**2
"""

from typing import Tuple

from .expr import Expr, to_expr


class Transpose(Expr):
    """Matrix transpose operation for symbolic expressions.

    Transposes the last two dimensions of an expression. For matrices, this swaps
    rows and columns. For higher-dimensional arrays, it swaps the last two axes.
    Scalars and vectors are unchanged by transposition.

    The canonicalization includes an optimization that eliminates double transposes:
    (A.T).T simplifies to A.

    Attributes:
        operand: Expression to transpose

    Example:
        Define Tranpose expressions:

            A = Variable("A", shape=(3, 4))
            A_T = Transpose(A)  # or A.T, result shape (4, 3)
            v = Variable("v", shape=(5,))
            v_T = Transpose(v)  # result shape (5,) - vectors unchanged
    """

    def __init__(self, operand):
        """Initialize a transpose operation.

        Args:
            operand: Expression to transpose
        """
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
    """Diagonal matrix construction from a vector.

    Creates a square diagonal matrix from a 1D vector. The vector elements become
    the diagonal entries, with all off-diagonal entries set to zero. This is
    analogous to numpy.diag() or jax.numpy.diag().

    Note:
        Currently only supports creating diagonal matrices from vectors.
        Extracting diagonals from matrices is not yet implemented.

    Attributes:
        operand: 1D vector expression to place on the diagonal

    Example:
        Define a Diag:

            v = Variable("v", shape=(3,))
            D = Diag(v)  # Creates a (3, 3) diagonal matrix
    """

    def __init__(self, operand):
        """Initialize a diagonal matrix operation.

        Args:
            operand: 1D vector expression to place on the diagonal
        """
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
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
    """Norm operation for symbolic expressions (reduction to scalar).

    Computes the norm of an expression according to the specified order parameter.
    This is a reduction operation that always produces a scalar result regardless
    of the input shape. Supports various norm types following NumPy/SciPy conventions.

    Attributes:
        operand: Expression to compute norm of
        ord: Norm order specification (default: "fro" for Frobenius norm)
            - "fro": Frobenius norm (default)
            - "inf": Infinity norm
            - 1: L1 norm (sum of absolute values)
            - 2: L2 norm (Euclidean norm)
            - Other values as supported by the backend

    Example:
        Define Norms:

            x = Variable("x", shape=(3,))
            euclidean_norm = Norm(x, ord=2)  # L2 norm, result is scalar
            A = Variable("A", shape=(3, 4))
            frobenius_norm = Norm(A)  # Frobenius norm, result is scalar
    """

    def __init__(self, operand, ord="fro"):
        """Initialize a norm operation.

        Args:
            operand: Expression to compute norm of
            ord: Norm order specification (default: "fro")
        """
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
    """Stack expressions vertically to create a higher-dimensional array.

    Stacks a list of expressions along a new first dimension. All input expressions
    must have the same shape. The result has shape (num_rows, *row_shape).

    This is similar to numpy.array([row1, row2, ...]) or jax.numpy.stack(rows, axis=0).

    Attributes:
        rows: List of expressions to stack, each representing a "row"

    Example:
        Leverage stack to combine expressions:

            x = Variable("x", shape=(3,))
            y = Variable("y", shape=(3,))
            z = Variable("z", shape=(3,))
            stacked = Stack([x, y, z])  # Creates shape (3, 3)
            # Equivalent to: [[x[0], x[1], x[2]],
            #                 [y[0], y[1], y[2]],
            #                 [z[0], z[1], z[2]]]
    """

    def __init__(self, rows):
        """Initialize a stack operation.

        Args:
            rows: List of expressions to stack along a new first dimension.
                  All expressions must have the same shape.
        """
        # rows should be a list of expressions representing each row
        self.rows = [to_expr(row) for row in rows]

    def children(self):
        return self.rows

    def canonicalize(self) -> "Expr":
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
    """Horizontal stacking operation for symbolic expressions.

    Concatenates expressions horizontally (along columns for 2D arrays).
    This is analogous to numpy.hstack() or jax.numpy.hstack().

    Behavior depends on input dimensionality:
    - 1D arrays: Concatenates along axis 0 (making a longer vector)
    - 2D arrays: Concatenates along axis 1 (columns), rows must match
    - Higher-D: Concatenates along axis 1, all other dimensions must match

    Attributes:
        arrays: List of expressions to stack horizontally

    Example:
        1D case: concatenate vectors:

            x = Variable("x", shape=(3,))
            y = Variable("y", shape=(2,))
            h = Hstack([x, y])  # Result shape (5,)

        2D case: concatenate matrices horizontally:

            A = Variable("A", shape=(3, 4))
            B = Variable("B", shape=(3, 2))
            C = Hstack([A, B])  # Result shape (3, 6)
    """

    def __init__(self, arrays):
        """Initialize a horizontal stack operation.

        Args:
            arrays: List of expressions to concatenate horizontally
        """
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def canonicalize(self) -> "Expr":
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
                    f"Hstack array {i} has trailing dimensions {shape[2:]}, "
                    f"but array 0 has {first_shape[2:]}"
                )

        # Result shape: concatenate along axis 1 (columns)
        total_cols = sum(shape[1] for shape in array_shapes)
        return (first_shape[0], total_cols) + first_shape[2:]

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Hstack([{arrays_repr}])"


class Vstack(Expr):
    """Vertical stacking operation for symbolic expressions.

    Concatenates expressions vertically (along rows for 2D arrays).
    This is analogous to numpy.vstack() or jax.numpy.vstack().

    All input expressions must have the same number of dimensions, and all
    dimensions except the first must match. The result concatenates along
    axis 0 (rows).

    Attributes:
        arrays: List of expressions to stack vertically

    Example:
        Stack vectors to create a matrix:

            x = Variable("x", shape=(3,))
            y = Variable("y", shape=(3,))
            v = Vstack([x, y])  # Result shape (2, 3)

        Stack matrices vertically:

            A = Variable("A", shape=(3, 4))
            B = Variable("B", shape=(2, 4))
            C = Vstack([A, B])  # Result shape (5, 4)
    """

    def __init__(self, arrays):
        """Initialize a vertical stack operation.

        Args:
            arrays: List of expressions to concatenate vertically.
                    All must have matching dimensions except the first.
        """
        self.arrays = [to_expr(arr) for arr in arrays]

    def children(self):
        return self.arrays

    def canonicalize(self) -> "Expr":
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
                    f"Vstack array {i} has trailing dimensions {shape[1:]}, "
                    f"but array 0 has {first_shape[1:]}"
                )

        # Result shape: concatenate along axis 0 (rows)
        total_rows = sum(shape[0] for shape in array_shapes)
        return (total_rows,) + first_shape[1:]

    def __repr__(self):
        arrays_repr = ", ".join(repr(arr) for arr in self.arrays)
        return f"Vstack([{arrays_repr}])"
