"""Tests for array operation nodes.

This module tests array operation nodes: Index, Concat, Stack, Hstack, Vstack.
Tests cover:

- Node creation and indexing/slicing operations
- Concatenation and stacking operations
- Shape inference
- Lowering to JAX
- Lowering to CVXPY
- Canonicalization patterns

Tests are organized by node type, with each section containing:

1. Node creation and tree structure tests
2. Shape/dimension tests (if applicable)
3. Canonicalization tests
4. JAX lowering tests
5. CVXPY lowering tests
6. Integration tests (if applicable)
"""

import numpy as np
import pytest

from openscvx.symbolic.expr import Concat, Constant, Hstack, Index, Vstack

# =============================================================================
# Index
# =============================================================================

# --- Index: Shape Checking ---


def test_index_valid_passes():
    a = Constant(np.zeros((5,)))
    index = a[2:4]
    index.check_shape()


def test_index_out_of_bounds_raises():
    a = Constant(np.zeros((3,)))
    index = a[5]
    with pytest.raises(ValueError):
        index.check_shape()


# --- Index: Canonicalization ---


def test_index_canonicalize():
    """Test that Index canonicalizes its children recursively."""
    idx = Index(Constant([5, 6, 7]), 1)
    result = idx.canonicalize()
    assert isinstance(result, Index)
    assert result.index == 1
    assert isinstance(result.base, Constant)


# --- Index: JAX Lowering ---


def test_index_and_slice():
    """Test JAX lowering of index and slice operations."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    # make a 4-vector state
    x = jnp.array([10.0, 20.0, 30.0, 40.0])
    s = State("s", (4,))
    s._slice = slice(0, 4)

    # index it and slice it
    expr_elem = s[2]
    expr_slice = s[1:3]

    # lower → callables
    fn_elem = lower_to_jax(expr_elem)
    fn_slice = lower_to_jax(expr_slice)

    # check results
    out_elem = fn_elem(x, None, None, None)
    out_slice = fn_slice(x, None, None, None)

    assert out_elem.shape == () or out_elem.shape == ()  # scalar or 0-D
    assert out_elem == x[2]

    assert out_slice.shape == (2,)
    assert jnp.allclose(out_slice, x[1:3])


# --- Index: CVXPy Lowering ---


def test_cvxpy_index():
    """Test CVXPY lowering of indexing."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Index(x, 0)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Concat
# =============================================================================

# --- Concat: Shape Checking ---


def test_concat_1d_passes():
    a = Constant(np.zeros((2,)))
    b = Constant(np.ones((3,)))
    concat = Concat(a, b)
    concat.check_shape()


def test_concat_rank_mismatch_raises():
    a = Constant(np.zeros((2, 2)))
    b = Constant(np.ones((3, 2, 2)))  # Changed to (3, 2, 2) to avoid squeeze collapsing dimensions
    concat = Concat(a, b)
    with pytest.raises(ValueError):
        concat.check_shape()


def test_concat_nonzero_axes_mismatch_raises():
    a = Constant(np.zeros((2, 3)))
    b = Constant(np.ones((3, 4)))
    # shapes (2,3) vs (3,4) agree on rank but not on axis>0
    concat = Concat(a, b)
    with pytest.raises(ValueError):
        concat.check_shape()


# --- Concat: Canonicalization ---


def test_concat_canonicalize():
    """Test that Concat canonicalizes its children recursively."""
    # Concat should simply rebuild with canonical children
    x = Constant([1, 2])
    y = Constant([3, 4])
    concat = Concat(x, y)
    result = concat.canonicalize()
    assert isinstance(result, Concat)
    # both children are still Constant
    assert all(isinstance(c, Constant) for c in result.exprs)


# --- Concat: JAX Lowering ---


def test_concat_simple():
    """Test JAX lowering of simple concatenation."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.arange(5.0)
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(2, 4)
    c = Constant(9.0)
    expr = Concat(a, b, c)

    fn = lower_to_jax(expr)
    out = fn(x, None, None, None)
    expected = jnp.concatenate([x[0:2], x[2:4], jnp.array([9.0])], axis=0)
    assert jnp.allclose(out, expected)
    assert out.shape == (5,)


# --- Concat: CVXPy Lowering ---


def test_cvxpy_concat():
    """Test CVXPY lowering of concatenation."""
    import cvxpy as cp

    from openscvx.symbolic.expr import Control, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    u_cvx = cp.Variable(2, name="u")
    variable_map = {"x": x_cvx, "u": u_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    u = Control("u", shape=(2,))
    expr = Concat(x, u)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Hstack & Vstack
# =============================================================================

# --- Hstack & Vstack: Shape Checking ---


def test_hstack_basic_passes():
    """Test basic horizontal stacking functionality"""
    a = Constant(np.array([1.0, 2.0]))  # (2,)
    b = Constant(np.array([3.0, 4.0, 5.0]))  # (3,)

    stacked = Hstack([a, b])

    result_shape = stacked.check_shape()
    assert result_shape == (5,)  # 2 + 3 = 5


def test_hstack_dimension_mismatch_raises():
    """Test that arrays with different numbers of dimensions raise error"""
    a = Constant(np.zeros((2,)))  # 1D
    b = Constant(np.ones((2, 3)))  # 2D

    with pytest.raises(ValueError) as exc:
        (Hstack([a, b])).check_shape()
    assert "dimensions" in str(exc.value)


def test_vstack_basic_passes():
    """Test basic vertical stacking functionality"""
    a = Constant(np.zeros((2, 3)))  # (2, 3)
    b = Constant(np.ones((4, 3)))  # (4, 3)

    stacked = Vstack([a, b])
    stacked.check_shape()

    result_shape = stacked.check_shape()
    assert result_shape == (6, 3)  # 2 + 4 = 6 rows


def test_vstack_trailing_dimension_mismatch_raises():
    """Test that arrays with mismatched trailing dimensions raise error"""
    a = Constant(np.zeros((2, 3)))  # (2, 3)
    b = Constant(np.ones((4, 5)))  # (4, 5) - different second dim

    with pytest.raises(ValueError) as exc:
        (Vstack([a, b])).check_shape()
    assert "trailing dimensions" in str(exc.value)


# --- Hstack & Vstack: Canonicalization --- TODO: (norrisg)

# --- Hstack & Vstack: JAX Lowering ---


def test_hstack_constants():
    """Test JAX lowering of Hstack with constant arrays."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Hstack
    from openscvx.symbolic.lower import lower_to_jax

    arr1 = Constant(np.array([1.0, 2.0]))
    arr2 = Constant(np.array([3.0, 4.0, 5.0]))
    expr = Hstack([arr1, arr2])

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert jnp.allclose(result, expected)
    assert result.shape == (5,)


def test_hstack_states_and_controls():
    """Test JAX lowering of Hstack with state and control variables."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control, Hstack, State
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([40.0, 50.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Stack: [state[0:2], control, constant]
    const = Constant(np.array([60.0]))
    expr = Hstack([state[0:2], control, const])

    fn = lower_to_jax(expr)
    result = fn(x, u, None, None)

    expected = jnp.array([10.0, 20.0, 40.0, 50.0, 60.0])
    assert jnp.allclose(result, expected)
    assert result.shape == (5,)


def test_vstack_constants():
    """Test JAX lowering of Vstack with constant arrays."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Vstack
    from openscvx.symbolic.lower import lower_to_jax

    arr1 = Constant(np.array([[1.0, 2.0]]))  # (1, 2)
    arr2 = Constant(np.array([[3.0, 4.0], [5.0, 6.0]]))  # (2, 2)
    expr = Vstack([arr1, arr2])

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert jnp.allclose(result, expected)
    assert result.shape == (3, 2)


def test_vstack_vectors():
    """Test JAX lowering of Vstack with vector arrays (promotes to 2D)."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State, Vstack
    from openscvx.symbolic.lower import lower_to_jax

    x = jnp.array([10.0, 20.0, 30.0, 40.0])

    state = State("x", (4,))
    state._slice = slice(0, 4)

    # Split state into two parts and stack vertically
    part1 = state[0:2]  # [10, 20]
    part2 = state[2:4]  # [30, 40]
    expr = Vstack([part1, part2])

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # vstack promotes 1D arrays to 2D: [[10, 20], [30, 40]]
    expected = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    assert jnp.allclose(result, expected)
    assert result.shape == (2, 2)


# --- Hstack & Vstack: CVXPy Lowering --- TODO: (norrisg)
