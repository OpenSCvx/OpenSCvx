"""Array manipulation operations for symbolic expressions.

This module provides operations for indexing, slicing, concatenating, and stacking
symbolic expressions. These are structural operations that manipulate array shapes
and combine or extract array elements, as opposed to mathematical transformations.

Key Operations:

- **Indexing and Slicing:**
    - `Index` - NumPy-style indexing and slicing to extract subarrays

- **Concatenation:**
    - `Concat` - Concatenate expressions along the first dimension (axis 0)

- **Stacking:**
    - `Stack` - Stack expressions along a new first dimension
    - `Hstack` - Horizontal stacking (along columns for 2D arrays)
    - `Vstack` - Vertical stacking (along rows for 2D arrays)

All operations follow NumPy conventions for shapes and indexing behavior, enabling
familiar array manipulation patterns in symbolic optimization problems.

Example:
    Indexing and slicing arrays::

        import openscvx as ox

        x = ox.State("x", shape=(10,))
        first_half = x[0:5]      # Slice: Index(x, slice(0, 5))
        element = x[3]           # Single element: Index(x, 3)

        A = ox.State("A", shape=(5, 4))
        row = A[2, :]            # Extract row
        col = A[:, 1]            # Extract column

    Concatenating expressions::

        from openscvx.symbolic.expr.array import Concat

        x = ox.State("x", shape=(3,))
        y = ox.State("y", shape=(4,))
        combined = Concat(x, y)  # Result shape (7,)

    Stacking to build matrices::

        from openscvx.symbolic.expr.array import Stack, Hstack, Vstack

        # Stack vectors into a matrix
        v1 = ox.State("v1", shape=(3,))
        v2 = ox.State("v2", shape=(3,))
        v3 = ox.State("v3", shape=(3,))
        matrix = Stack([v1, v2, v3])  # Result shape (3, 3)

        # Horizontal stacking (concatenate along columns)
        A = ox.State("A", shape=(3, 4))
        B = ox.State("B", shape=(3, 2))
        wide = Hstack([A, B])    # Result shape (3, 6)

        # Vertical stacking (concatenate along rows)
        C = ox.State("C", shape=(2, 4))
        tall = Vstack([A, C])    # Result shape (5, 4)

    Building rotation matrices with stacking::

        import openscvx as ox
        from openscvx.symbolic.expr.array import Stack, Hstack

        theta = ox.Variable("theta", shape=(1,))
        R = Stack([
            Hstack([ox.Cos(theta), -ox.Sin(theta)]),
            Hstack([ox.Sin(theta), ox.Cos(theta)])
        ])  # 2D rotation matrix, shape (2, 2)
"""

from typing import Tuple, Union

import numpy as np

from .expr import Expr, to_expr


class Index(Expr):
    """Indexing and slicing operation for symbolic expressions.

    Represents indexing or slicing of an expression using NumPy-style indexing.
    Can be created using square bracket notation on Expr objects.

    Attributes:
        base: Expression to index into
        index: Index specification (int, slice, or tuple of indices/slices)

    Example:
        Define an Index expression:

            x = ox.State("x", shape=(10,))
            y = x[0:5]  # Creates Index(x, slice(0, 5))
            z = x[3]    # Creates Index(x, 3)
    """

    def __init__(self, base: Expr, index: Union[int, slice, tuple]):
        """Initialize an indexing operation.

        Args:
            base: Expression to index into
            index: NumPy-style index (int, slice, or tuple of indices/slices)
        """
        self.base = base
        self.index = index

    def children(self):
        return [self.base]

    def canonicalize(self) -> "Expr":
        """Canonicalize index by canonicalizing the base expression.

        Returns:
            Expr: Canonical form of the indexing expression
        """
        base = self.base.canonicalize()
        return Index(base, self.index)

    def check_shape(self) -> Tuple[int, ...]:
        """Compute the shape after indexing."""
        base_shape = self.base.check_shape()
        dummy = np.zeros(base_shape)
        try:
            result = dummy[self.index]
        except Exception as e:
            raise ValueError(f"Bad index {self.index} for shape {base_shape}") from e
        return result.shape

    def __repr__(self):
        return f"{self.base!r}[{self.index!r}]"


class Concat(Expr):
    """Concatenation operation for symbolic expressions.

    Concatenates a sequence of expressions along their first dimension. All inputs
    must have the same rank and matching dimensions except for the first dimension.

    Attributes:
        exprs: Tuple of expressions to concatenate

    Example:
        Define a Concat expression:

            x = ox.State("x", shape=(3,))
            y = ox.State("y", shape=(4,))
            z = Concat(x, y)  # Creates Concat(x, y), result shape (7,)
    """

    def __init__(self, *exprs: Expr):
        """Initialize a concatenation operation.

        Args:
            *exprs: Expressions to concatenate along the first dimension
        """
        # wrap raw values as Constant if needed
        self.exprs = [to_expr(e) for e in exprs]

    def children(self):
        return list(self.exprs)

    def canonicalize(self) -> "Expr":
        """Canonicalize concatenation by canonicalizing all operands.

        Returns:
            Expr: Canonical form of the concatenation expression
        """
        exprs = [e.canonicalize() for e in self.exprs]
        return Concat(*exprs)

    def check_shape(self) -> Tuple[int, ...]:
        """Check concatenation shape compatibility and return result shape."""
        shapes = [e.check_shape() for e in self.exprs]
        shapes = [(1,) if len(s) == 0 else s for s in shapes]
        rank = len(shapes[0])
        if any(len(s) != rank for s in shapes):
            raise ValueError(f"Concat rank mismatch: {shapes}")
        if any(s[1:] != shapes[0][1:] for s in shapes[1:]):
            raise ValueError(f"Concat non-0 dims differ: {shapes}")
        return (sum(s[0] for s in shapes),) + shapes[0][1:]

    def __repr__(self):
        inner = ", ".join(repr(e) for e in self.exprs)
        return f"Concat({inner})"


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
