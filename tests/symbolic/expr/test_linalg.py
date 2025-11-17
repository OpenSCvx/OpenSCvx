"""Tests for linear algebra operation nodes.

This module tests linear algebra operation nodes: Transpose, Diag, Sum, Norm.

Tests are organized by node/node-group, with each section containing:

1. Node creation and properties
2. Shape Checking
3. Canonicalization
4. JAX lowering tests
5. CVXPY lowering tests (where applicable)
6. Integration tests (where applicable)
"""

import numpy as np

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Diag,
    Norm,
    Sum,
    Transpose,
    Variable,
)

# =============================================================================
# Transpose
# =============================================================================

# --- Transpose: Creation & Tree Structure ---


def test_transpose_creation_and_children():
    """Test Transpose node creation and tree structure."""
    A = Variable("A", shape=(3, 4))
    A_T = Transpose(A)

    assert isinstance(A_T, Transpose)
    assert A_T.children() == [A]
    assert repr(A_T) == "(Var('A')).T"


def test_transpose_with_property_accessor():
    """Test that .T property creates Transpose node."""
    A = Variable("A", shape=(3, 4))
    A_T = A.T

    assert isinstance(A_T, Transpose)
    assert A_T.operand is A


def test_transpose_wraps_constants():
    """Test Transpose with constant matrices."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    T = Transpose(arr)

    assert isinstance(T.operand, Constant)
    assert np.array_equal(T.operand.value, arr)


# --- Transpose: Shape Checking ---


def test_transpose_shape_scalar():
    """Test that transpose of scalar is unchanged."""
    x = Variable("x", shape=())
    x_T = Transpose(x)

    assert x_T.check_shape() == ()


def test_transpose_shape_vector():
    """Test that transpose of vector is unchanged."""
    v = Variable("v", shape=(5,))
    v_T = Transpose(v)

    assert v_T.check_shape() == (5,)


def test_transpose_shape_matrix():
    """Test that transpose of matrix swaps dimensions."""
    A = Variable("A", shape=(3, 4))
    A_T = Transpose(A)

    assert A_T.check_shape() == (4, 3)


def test_transpose_shape_tensor():
    """Test that transpose of tensor swaps last two dimensions."""
    T = Variable("T", shape=(2, 3, 4, 5))
    T_T = Transpose(T)

    assert T_T.check_shape() == (2, 3, 5, 4)


# --- Transpose: Canonicalization ---


def test_transpose_canonicalize_recurses():
    """Test that canonicalize recurses into the operand."""
    x = Variable("x", shape=(3, 4))
    expr = Transpose(x + 0)  # x + 0 should simplify

    canonical = expr.canonicalize()

    # The Add(x, 0) should have been canonicalized
    assert isinstance(canonical, Transpose)
    # After canonicalization, x + 0 should become x
    assert canonical.operand == x


def test_transpose_double_transpose_eliminates():
    """Test that (A.T).T simplifies to A."""
    A = Variable("A", shape=(3, 4))
    double_T = Transpose(Transpose(A))

    canonical = double_T.canonicalize()

    # Double transpose should be eliminated
    assert canonical is A


def test_transpose_triple_transpose_becomes_single():
    """Test that ((A.T).T).T simplifies to A.T."""
    A = Variable("A", shape=(3, 4))
    triple_T = Transpose(Transpose(Transpose(A)))

    canonical = triple_T.canonicalize()

    # Should reduce to single transpose
    assert isinstance(canonical, Transpose)
    assert canonical.operand is A


# --- Transpose: JAX Lowering ---


def test_transpose_jax_matrix():
    """Test JAX lowering of matrix transpose."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    # Test with a 2x3 matrix
    A = Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    A_T = Transpose(A)
    fn = lower_to_jax(A_T)

    result = fn(None, None, None, None)

    # Should be transposed to 3x2
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert jnp.allclose(result, expected, atol=1e-12)


def test_transpose_jax_vector():
    """Test JAX lowering of vector transpose (no-op)."""
    import jax.numpy as jnp

    from openscvx.symbolic.lower import lower_to_jax

    v = Constant(np.array([1.0, 2.0, 3.0]))
    v_T = Transpose(v)
    fn = lower_to_jax(v_T)

    result = fn(None, None, None, None)

    # Vector transpose is identity
    expected = np.array([1.0, 2.0, 3.0])
    assert jnp.allclose(result, expected, atol=1e-12)


# --- Transpose: CVXPy Lowering ---


def test_transpose_cvxpy():
    """Test CVXPy lowering of matrix transpose."""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((2, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(6,))  # 2x3 flattened
    expr = Transpose(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Sum
# =============================================================================

# --- Sum: Creation & Tree Structure ---


def test_sum_node_creation_and_children():
    """Test Sum node creation and tree structure."""
    x = Variable("x", shape=(3,))
    sum_expr = Sum(x)

    assert isinstance(sum_expr, Sum)
    assert sum_expr.children() == [x]
    assert repr(sum_expr) == "sum(Var('x'))"


def test_sum_wraps_constants_and_expressions():
    """Test Sum node with various input types."""
    # Sum of a constant array
    arr = np.array([1.0, 2.0, 3.0])
    sum1 = Sum(arr)
    assert isinstance(sum1.operand, Constant)
    assert np.array_equal(sum1.operand.value, arr)
    assert repr(sum1) == "sum(Const([1.0, 2.0, 3.0]))"

    # Sum of an arithmetic expression
    x = Variable("x", shape=(2,))
    y = Variable("y", shape=(2,))
    sum2 = Sum(x + y)
    assert isinstance(sum2.operand, Add)
    assert len(sum2.operand.children()) == 2
    assert repr(sum2) == "sum((Var('x') + Var('y')))"


# --- Sum: Shape Checking ---


def test_sum_shape_vector():
    """Test that sum of vector produces scalar."""
    x = Variable("x", shape=(5,))
    sum_expr = Sum(x)

    assert sum_expr.check_shape() == ()


def test_sum_shape_matrix():
    """Test that sum of matrix produces scalar."""
    A = Variable("A", shape=(3, 4))
    sum_expr = Sum(A)

    assert sum_expr.check_shape() == ()


def test_sum_shape_scalar():
    """Test that sum of scalar produces scalar."""
    x = Variable("x", shape=())
    sum_expr = Sum(x)

    assert sum_expr.check_shape() == ()


# --- Sum: Canonicalization ---


def test_sum_canonicalize_recurses():
    """Test that canonicalize recurses into the operand."""
    x = Variable("x", shape=(3,))
    expr = Sum(x + 0)  # x + 0 should simplify

    canonical = expr.canonicalize()

    # The Add(x, 0) should have been canonicalized
    assert isinstance(canonical, Sum)
    # After canonicalization, x + 0 should become x
    assert canonical.operand == x


# --- Sum: JAX Lowering ---


def test_sum_jax_vector():
    """Test JAX lowering of sum over vector."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(4,))
    x._slice = slice(0, 4)

    sum_expr = Sum(x)
    fn = lower_to_jax(sum_expr)

    x_val = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = fn(x_val, None, None, None)

    expected = jnp.sum(x_val)
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == ()  # Should be scalar


def test_sum_jax_matrix():
    """Test JAX lowering of sum over matrix."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    A = State("A", shape=(6,))  # 2x3 flattened
    A._slice = slice(0, 6)

    sum_expr = Sum(A)
    fn = lower_to_jax(sum_expr)

    A_val = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = fn(A_val, None, None, None)

    expected = jnp.sum(A_val)
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == ()  # Should be scalar


# --- Sum: CVXPy Lowering ---


def test_cvxpy_sum():
    """Test sum operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Sum(x)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


# =============================================================================
# Diag
# =============================================================================

# --- Diag: Creation & Tree Structure ---


def test_diag_creation_and_children():
    """Test Diag node creation and tree structure."""
    v = Variable("v", shape=(3,))
    diag_expr = Diag(v)

    assert isinstance(diag_expr, Diag)
    assert diag_expr.children() == [v]
    assert repr(diag_expr) == "diag(Var('v'))"


def test_diag_wraps_constants():
    """Test Diag with constant vectors."""
    arr = np.array([1.0, 2.0, 3.0])
    diag_expr = Diag(arr)

    assert isinstance(diag_expr.operand, Constant)
    assert np.array_equal(diag_expr.operand.value, arr)


# --- Diag: Shape Checking ---


def test_diag_shape_from_vector():
    """Test that diag of vector (n,) produces matrix (n, n)."""
    v = Variable("v", shape=(5,))
    diag_expr = Diag(v)

    assert diag_expr.check_shape() == (5, 5)


def test_diag_shape_requires_vector():
    """Test that Diag raises error for non-vector inputs."""
    import pytest

    # Try with a scalar
    x = Variable("x", shape=())
    diag_expr = Diag(x)
    with pytest.raises(ValueError, match="Diag expects a 1D vector"):
        diag_expr.check_shape()

    # Try with a matrix
    A = Variable("A", shape=(3, 3))
    diag_expr2 = Diag(A)
    with pytest.raises(ValueError, match="Diag expects a 1D vector"):
        diag_expr2.check_shape()


# --- Diag: Canonicalization ---


def test_diag_canonicalize_recurses():
    """Test that canonicalize recurses into the operand."""
    v = Variable("v", shape=(3,))
    expr = Diag(v + 0)  # v + 0 should simplify

    canonical = expr.canonicalize()

    # The Add(v, 0) should have been canonicalized
    assert isinstance(canonical, Diag)
    # After canonicalization, v + 0 should become v
    assert canonical.operand == v


# --- Diag: JAX Lowering ---


def test_diag():
    """Test the Diag compact node individually."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Diag, State
    from openscvx.symbolic.lower import lower_to_jax

    # Test with different vectors
    test_vectors = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([0.5, -1.0, 2.5]),
        jnp.array([0.0, 0.0, 0.0]),
    ]

    for v_val in test_vectors:
        # Create vector state
        v = State("v", (3,))
        v._slice = slice(0, 3)

        # Test Diag node
        diag_expr = Diag(v)
        fn = lower_to_jax(diag_expr)
        result = fn(v_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be diagonal
        expected = jnp.diag(v_val)
        assert jnp.allclose(result, expected, atol=1e-12)

        # Off-diagonal elements should be zero
        off_diag_mask = ~jnp.eye(3, dtype=bool)
        assert jnp.allclose(result[off_diag_mask], 0.0, atol=1e-12)


# --- Diag: CVXPy Lowering ---

# TODO: Implement Diag CVXPy lowering
# def test_diag_cvxpy():
#     """Test CVXPy lowering of Diag."""
#     import cvxpy as cp
#
#     from openscvx.symbolic.expr import State
#     from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer
#
#     v_cvx = cp.Variable(3, name="v")
#     variable_map = {"v": v_cvx}
#     lowerer = CvxpyLowerer(variable_map)
#
#     v = State("v", shape=(3,))
#     expr = Diag(v)
#
#     result = lowerer.lower(expr)
#     assert isinstance(result, cp.Expression)


# =============================================================================
# Norm
# =============================================================================

# --- Norm: Creation & Tree Structure ---


def test_norm_creation_and_children():
    """Test Norm node creation and tree structure."""
    x = Variable("x", shape=(3,))
    norm_expr = Norm(x)

    assert isinstance(norm_expr, Norm)
    assert norm_expr.children() == [x]
    assert norm_expr.ord == "fro"  # Default order
    assert repr(norm_expr) == "norm(Var('x'), ord='fro')"


def test_norm_with_different_orders():
    """Test Norm with different order parameters."""
    x = Variable("x", shape=(3,))

    # L2 norm
    norm_l2 = Norm(x, ord=2)
    assert norm_l2.ord == 2
    assert repr(norm_l2) == "norm(Var('x'), ord=2)"

    # L1 norm
    norm_l1 = Norm(x, ord=1)
    assert norm_l1.ord == 1

    # Infinity norm
    norm_inf = Norm(x, ord="inf")
    assert norm_inf.ord == "inf"

    # Frobenius norm (explicit)
    norm_fro = Norm(x, ord="fro")
    assert norm_fro.ord == "fro"


def test_norm_wraps_constants():
    """Test Norm with constant arrays."""
    arr = np.array([1.0, 2.0, 3.0])
    norm_expr = Norm(arr, ord=2)

    assert isinstance(norm_expr.operand, Constant)
    assert np.array_equal(norm_expr.operand.value, arr)


# --- Norm: Shape Checking ---


def test_norm_shape_vector():
    """Test that norm of vector produces scalar."""
    x = Variable("x", shape=(5,))
    norm_expr = Norm(x, ord=2)

    assert norm_expr.check_shape() == ()


def test_norm_shape_matrix():
    """Test that norm of matrix produces scalar."""
    A = Variable("A", shape=(3, 4))
    norm_expr = Norm(A, ord="fro")

    assert norm_expr.check_shape() == ()


def test_norm_shape_scalar():
    """Test that norm of scalar produces scalar."""
    x = Variable("x", shape=())
    norm_expr = Norm(x)

    assert norm_expr.check_shape() == ()


# --- Norm: Canonicalization ---


def test_norm_canonicalize_recurses():
    """Test that canonicalize recurses into the operand."""
    x = Variable("x", shape=(3,))
    expr = Norm(x + 0, ord=2)  # x + 0 should simplify

    canonical = expr.canonicalize()

    # The Add(x, 0) should have been canonicalized
    assert isinstance(canonical, Norm)
    assert canonical.ord == 2  # ord parameter preserved
    # After canonicalization, x + 0 should become x
    assert canonical.operand == x


def test_norm_canonicalize_preserves_ord():
    """Test that canonicalization preserves the ord parameter."""
    x = Variable("x", shape=(3,))

    for ord_val in [1, 2, "inf", "fro"]:
        expr = Norm(x, ord=ord_val)
        canonical = expr.canonicalize()

        assert isinstance(canonical, Norm)
        assert canonical.ord == ord_val


# --- Norm: JAX Lowering ---


def test_norm_jax_l2_vector():
    """Test JAX lowering of L2 norm of vector."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    norm_expr = Norm(x, ord=2)
    fn = lower_to_jax(norm_expr)

    x_val = jnp.array([3.0, 4.0, 0.0])
    result = fn(x_val, None, None, None)

    expected = jnp.linalg.norm(x_val, ord=2)
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == ()  # Should be scalar


def test_norm_jax_l1_vector():
    """Test JAX lowering of L1 norm of vector."""
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lower import lower_to_jax

    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    norm_expr = Norm(x, ord=1)
    fn = lower_to_jax(norm_expr)

    x_val = jnp.array([1.0, -2.0, 3.0])
    result = fn(x_val, None, None, None)

    expected = jnp.linalg.norm(x_val, ord=1)
    assert jnp.allclose(result, expected, atol=1e-12)
    assert result.shape == ()


# --- Norm: CVXPy Lowering ---


def test_cvxpy_norm_l2():
    """Test L2 norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord=2)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_l1():
    """Test L1 norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord=1)

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_inf():
    """Test infinity norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable(3, name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(3,))
    expr = Norm(x, ord="inf")

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)


def test_cvxpy_norm_fro():
    """Test Frobenius norm operation"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Norm, State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((2, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    x = State("x", shape=(6,))  # Flattened 2x3 matrix
    expr = Norm(x, ord="fro")

    result = lowerer.lower(expr)
    assert isinstance(result, cp.Expression)
