"""Tests for variable nodes.

This module tests variable node types: Variable, State, Control.
Tests cover:
- Node creation and properties
- Bounds and constraints
- Slice assignment and usage
- Lowering to JAX (with slices)
- Lowering to CVXPY (with variable mapping)
"""

import numpy as np
import pytest

# =============================================================================
# JAX Lowering Tests
# =============================================================================


def test_jax_lower_state_without_slice_raises():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    s = State("s", (3,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl._visit_state(s)


def test_jax_lower_control_without_slice_raises():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    c = Control("c", (2,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl._visit_control(c)


def test_jax_lower_state_with_slice():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    x = jnp.arange(10.0)
    s = State("s", (4,))
    s._slice = slice(2, 6)
    jl = JaxLowerer()
    f = jl._visit_state(s)
    out = f(x, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert jnp.allclose(out, x[2:6])


def test_jax_lower_control_with_slice():
    import jax.numpy as jnp

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.jax import JaxLowerer

    u = jnp.arange(8.0)
    c = Control("c", (3,))
    c._slice = slice(5, 8)
    jl = JaxLowerer()
    f = jl._visit_control(c)
    out = f(None, u, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, u[5:8])


# =============================================================================
# CVXPY Lowering Tests
# =============================================================================


def test_cvxpy_state_variable():
    """Test lowering state variables"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    # Create CVXPy variables
    x_cvx = cp.Variable((10, 3), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    # Create symbolic state
    x = State("x", shape=(3,))

    # Lower to CVXPy
    result = lowerer.lower(x)
    assert result is x_cvx  # Should return the mapped variable


def test_cvxpy_state_variable_with_slice():
    """Test state variables with slices"""
    import cvxpy as cp

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    x_cvx = cp.Variable((10, 6), name="x")
    variable_map = {"x": x_cvx}
    lowerer = CvxpyLowerer(variable_map)

    # State with slice
    x = State("x", shape=(3,))
    x._slice = slice(0, 3)

    result = lowerer.lower(x)
    # Should return x_cvx with slice applied
    assert isinstance(result, cp.Expression)


def test_cvxpy_control_variable():
    """Test lowering control variables"""
    import cvxpy as cp

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    u_cvx = cp.Variable((10, 2), name="u")
    variable_map = {"u": u_cvx}
    lowerer = CvxpyLowerer(variable_map)

    u = Control("u", shape=(2,))
    result = lowerer.lower(u)
    assert result is u_cvx


def test_cvxpy_missing_state_variable_error():
    """Test error when state vector not in map"""

    from openscvx.symbolic.expr import State
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    lowerer = CvxpyLowerer({})
    x = State("missing", shape=(3,))

    with pytest.raises(ValueError, match="State vector 'x' not found"):
        lowerer.lower(x)


def test_cvxpy_missing_control_variable_error():
    """Test error when control vector not in map"""

    from openscvx.symbolic.expr import Control
    from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer

    lowerer = CvxpyLowerer({})
    u = Control("thrust", shape=(2,))

    with pytest.raises(ValueError, match="Control vector 'u' not found"):
        lowerer.lower(u)
