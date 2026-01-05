"""Vmap expression for data-parallel operations.

This module provides symbolic support for JAX's vmap (vectorized map) operation,
enabling efficient data-parallel computations over batched data within the
symbolic expression framework.

Vmap supports two modes based on the type of `batch`:

- **Constant/array**: Values baked into the compiled function at trace time,
  equivalent to closure-captured values in BYOF. Use for static data.
- **Parameter**: Values looked up from params dict at runtime, allowing
  updates between SCP iterations. Use for data that may change.

Example:
    Compute distances from a position to multiple reference points::

        import openscvx as ox
        import numpy as np

        position = ox.State("position", shape=(3,))
        init_poses = np.random.randn(10, 3)  # 10 reference points

        # Option 1: Baked-in data (closure-equivalent)
        distances = ox.Vmap(
            lambda pose: ox.linalg.Norm(position - pose),
            batch=init_poses  # or batch=ox.Constant(init_poses)
        )

        # Option 2: Runtime-updateable Parameter
        refs = ox.Parameter("refs", shape=(10, 3), value=init_poses)
        distances = ox.Vmap(
            lambda pose: ox.linalg.Norm(position - pose),
            batch=refs
        )
"""

import uuid
from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np

from .expr import Constant, Expr, Leaf

if TYPE_CHECKING:
    from .expr import Parameter


class _Placeholder(Leaf):
    """Placeholder variable for use inside Vmap expressions.

    Placeholder is a symbolic leaf node that represents a single element from
    a batched array during vmap execution. It is created automatically by
    Vmap.__init__ and should not be instantiated directly by users.

    During lowering, the Vmap visitor injects the current batch element into
    the params dict, and Placeholder retrieves it via params lookup.

    Attributes:
        name (str): Unique identifier for params lookup (auto-generated)
        _shape (tuple): Shape of a single element from the batched data

    Note:
        Users should not create Placeholder instances directly. Instead, use
        ox.Vmap with a lambda that receives the placeholder as an argument.
    """

    def __init__(self, shape: Tuple[int, ...]):
        """Initialize a Placeholder.

        Args:
            shape: Shape of a single element from the batched data.
                   For example, if vmapping over data with shape (10, 3),
                   the placeholder shape would be (3,).
        """
        # Generate unique name for params lookup
        name = f"_vmap_placeholder_{uuid.uuid4().hex[:8]}"
        super().__init__(name, shape)

    def _hash_into(self, hasher):
        """Hash Placeholder by its unique name.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Placeholder")
        hasher.update(self.name.encode())


class Vmap(Expr):
    """Vectorized map over batched data in symbolic expressions.

    Vmap enables data-parallel operations by applying a symbolic expression
    to each element of a batched array. This is the symbolic equivalent of
    JAX's jax.vmap, allowing efficient vectorized computation without
    explicit loops.

    The expression is defined via a lambda that receives a Placeholder
    representing a single element from the batch. During lowering, this
    becomes a jax.vmap call.

    The behavior depends on the type of `batch`:

    - **numpy array or Constant**: Data is baked into the compiled function
      at trace time, equivalent to closure-captured values in BYOF.
    - **Parameter**: Data is looked up from the params dict at runtime,
      allowing the same compiled code to be reused with different values.

    Attributes:
        _batch: The data source (Constant or Parameter)
        _axis (int): The axis to vmap over (default: 0)
        _placeholder (Placeholder): The placeholder used in the expression
        _child (Expr): The expression tree built from the user's lambda
        _is_parameter (bool): Whether _batch is a Parameter (runtime lookup)

    Example:
        Compute distances to multiple reference points (baked-in)::

            position = ox.State("position", shape=(3,))
            init_poses = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            distances = ox.Vmap(
                lambda pose: ox.linalg.Norm(position - pose),
                batch=init_poses
            )
            # distances has shape (3,)

        With runtime-updateable Parameter::

            refs = ox.Parameter("refs", shape=(10, 3), value=init_poses)
            dist_state = ox.State("dist_state", shape=(10,))

            dynamics["dist_state"] = ox.Vmap(
                lambda pose: ox.linalg.Norm(position - pose),
                batch=refs
            )

            # Later, change the parameter value without recompiling:
            problem.parameters["refs"] = new_poses

    Note:
        - For static data that won't change, pass a numpy array or Constant
          to get closure-equivalent behavior (numerically identical to BYOF).
        - For data that needs to be updated between iterations, use Parameter.

    !!! warning "Prefer Constants over Parameters"
        **Use a raw numpy array or Constant unless you specifically need to
        update the vmap data between solves without recompiling.**

        Using a Parameter (runtime lookup) may produce **different numerical
        results** compared to using a Constant (baked-in), even when the
        underlying data is identical. This can manifest as:

        - Different SCP iteration counts
        - Different convergence behavior
        - In unlucky cases, convergence to a different local solution

        This is likely due to JAX/XLA trace and compilation differences between
        the two code paths. When data is baked in, JAX sees concrete values at
        trace time. When data is looked up from a params dict at runtime, JAX
        traces through the dictionary access, potentially producing different
        XLA compilation or floating-point operation ordering.
    """

    def __init__(
        self,
        fn: Callable[[_Placeholder], Expr],
        batch: Union[np.ndarray, Constant, "Parameter"],
        axis: int = 0,
    ):
        """Initialize a Vmap expression.

        Args:
            fn: A callable (typically a lambda) that takes a Placeholder and
                returns a symbolic expression. The Placeholder represents a
                single element from the batched data.
            batch: The batched data to vmap over. Can be:
                  - numpy array: baked into compiled function (closure-equivalent)
                  - Constant: baked into compiled function (closure-equivalent)
                  - Parameter: looked up from params dict at runtime
            axis: The axis to vmap over. Default is 0 (first axis).

        Example:
            Baked-in data::

                ox.Vmap(lambda x: ox.linalg.Norm(x), batch=points)

            With Parameter::

                refs = ox.Parameter("refs", shape=(10, 3), value=points)
                ox.Vmap(lambda ref: ox.linalg.Norm(position - ref), batch=refs)
        """
        from .expr import Parameter

        # Normalize input: wrap raw arrays in Constant
        if isinstance(batch, np.ndarray):
            batch = Constant(batch)
        elif not isinstance(batch, (Constant, Parameter)):
            # Try to convert to array then Constant
            batch = Constant(np.asarray(batch))

        self._batch = batch
        self._axis = axis
        self._is_parameter = isinstance(batch, Parameter)

        # Get shape from the appropriate source
        if self._is_parameter:
            batch_shape = batch.shape
        else:
            # Constant
            batch_shape = batch.value.shape

        # Compute per-element shape by removing the vmap axis
        if axis < 0 or axis >= len(batch_shape):
            raise ValueError(f"Vmap axis {axis} out of bounds for data with shape {batch_shape}")
        per_elem_shape = tuple(s for i, s in enumerate(batch_shape) if i != axis)

        # Create placeholder and build expression tree
        self._placeholder = _Placeholder(shape=per_elem_shape)
        self._child = fn(self._placeholder)

    @property
    def batch(self):
        """The batched data source being vmapped over."""
        return self._batch

    @property
    def axis(self) -> int:
        """The axis being vmapped over."""
        return self._axis

    @property
    def placeholder(self) -> _Placeholder:
        """The placeholder used in the inner expression."""
        return self._placeholder

    @property
    def is_parameter(self) -> bool:
        """Whether the data source is a Parameter (runtime lookup)."""
        return self._is_parameter

    def children(self):
        """Return child expressions.

        Returns:
            list: The vmapped expression and (if Parameter) the data source.
                  Parameter is included so traverse() finds it for parameter
                  collection in preprocessing.
        """
        if self._is_parameter:
            return [self._child, self._batch]
        else:
            return [self._child]

    def canonicalize(self) -> "Expr":
        """Canonicalize by canonicalizing the child expression.

        Returns:
            Vmap: A new Vmap with canonicalized child expression
        """
        canon_child = self._child.canonicalize()
        # Create new Vmap with the canonicalized child
        new_vmap = Vmap.__new__(Vmap)
        new_vmap._batch = self._batch
        new_vmap._axis = self._axis
        new_vmap._placeholder = self._placeholder
        new_vmap._child = canon_child
        new_vmap._is_parameter = self._is_parameter
        return new_vmap

    def check_shape(self) -> Tuple[int, ...]:
        """Compute the output shape of the vmapped expression.

        The output shape is (batch_size,) + inner_shape, where batch_size
        is the size of the vmap axis and inner_shape is the shape of the
        child expression.

        Returns:
            tuple: Output shape after vmapping

        Example:
            If data has shape (10, 3) and the inner expression produces a
            scalar (shape ()), the output shape is (10,).
        """
        inner_shape = self._child.check_shape()

        if self._is_parameter:
            batch_size = self._batch.shape[self._axis]
        else:
            batch_size = self._batch.value.shape[self._axis]

        return (batch_size,) + inner_shape

    def _hash_into(self, hasher):
        """Hash Vmap including data source, axis, and child expression.

        Args:
            hasher: A hashlib hash object to update
        """
        hasher.update(b"Vmap")
        hasher.update(str(self._axis).encode())
        hasher.update(str(self._is_parameter).encode())

        if self._is_parameter:
            # Hash Parameter by name and shape (not value - value can change)
            self._batch._hash_into(hasher)
        else:
            # Hash Constant by value (baked in, won't change)
            hasher.update(self._batch.value.tobytes())

        self._child._hash_into(hasher)

    def __repr__(self):
        """String representation of the Vmap expression.

        Returns:
            str: Description of the Vmap
        """
        if self._is_parameter:
            return f"Vmap(batch=Parameter({self._batch.name!r}), axis={self._axis})"
        else:
            return f"Vmap(batch=Constant(shape={self._batch.value.shape}), axis={self._axis})"
