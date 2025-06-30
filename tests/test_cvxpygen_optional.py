"""
Test that cvxpygen is properly handled as an optional dependency.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import os
import shutil
from openscvx.trajoptproblem import TrajOptProblem
from openscvx.dynamics import dynamics
from openscvx.constraints import ctcs
from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter

# Conditionally import cvxpygen to see if it's installed
try:
    import cvxpygen
    CVXPYGEN_INSTALLED = True
except ImportError:
    CVXPYGEN_INSTALLED = False


@dynamics
def simple_dynamics(x_, u_):
    """Simple dynamics for testing."""
    x_dot = u_
    t_dot = 1
    return jnp.hstack([x_dot, t_dot])


def test_cvxpygen_optional_import():
    """Test that cvxpygen import is optional."""
    # This should not raise an ImportError
    try:
        from openscvx.ocp import OptimalControlProblem
        from openscvx.ptr import PTR_init
        assert True
    except ImportError as e:
        if "cvxpygen" in str(e):
            pytest.fail("cvxpygen import should be optional")
        else:
            raise


def test_cvxpygen_disabled_by_default():
    """Test that cvxpygen is disabled by default."""
    # Create a simple problem
    n = 5
    x = State("x", shape=(2,))
    u = Control("u", shape=(1,))
    
    x.min = np.array([-5., -5.])
    x.max = np.array([5., 5.])
    x.initial = np.array([0, -1])
    x.final = np.array([0, 1])
    x.guess = np.linspace([0, -1], [0, 1], n)
    
    u.min = np.array([-2])
    u.max = np.array([2])
    u.guess = np.zeros((n, 1))
    
    constraints = [
        ctcs(lambda x_, u_: x_ - x.true.max),
        ctcs(lambda x_, u_: x.true.min - x_)
    ]
    
    problem = TrajOptProblem(
        dynamics=simple_dynamics,
        x=x,
        u=u,
        params=[],
        idx_time=1,
        constraints=constraints,
        N=n
    )
    
    # Check that cvxpygen is disabled by default
    assert not problem.settings.cvx.cvxpygen


def test_cvxpygen_enabled_raises_error_without_install():
    """Test that enabling cvxpygen without installation raises appropriate error."""

    # This test should only run if cvxpygen is NOT installed
    if CVXPYGEN_INSTALLED:
        pytest.skip("cvxpygen is installed, skipping test for error raising")

    # Create a simple problem
    n = 5
    x = State("x", shape=(2,))
    u = Control("u", shape=(1,))
    
    x.min = np.array([-5., -5.])
    x.max = np.array([5., 5.])
    x.initial = np.array([0, -1])
    x.final = np.array([0, 1])
    x.guess = np.linspace([0, -1], [0, 1], n)
    
    u.min = np.array([-2])
    u.max = np.array([2])
    u.guess = np.zeros((n, 1))
    
    constraints = [
        ctcs(lambda x_, u_: x_ - x.true.max),
        ctcs(lambda x_, u_: x.true.min - x_)
    ]
    
    problem = TrajOptProblem(
        dynamics=simple_dynamics,
        x=x,
        u=u,
        params=[],
        idx_time=1,
        constraints=constraints,
        N=n
    )
    
    # Enable cvxpygen
    problem.settings.cvx.cvxpygen = True
    problem.settings.cvx.cvxpygen_override = True
    
    # This should raise an ImportError with a helpful message
    with pytest.raises(ImportError) as exc_info:
        problem.initialize()
    
    assert "cvxpygen" in str(exc_info.value)
    assert "pip install openscvx[cvxpygen]" in str(exc_info.value)