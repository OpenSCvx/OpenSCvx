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
