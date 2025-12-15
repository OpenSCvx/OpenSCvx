"""
Unit tests for expert-mode validation logic.

Tests the validation of bring-your-own-functions (byof) to ensure proper
error handling and clear error messages for invalid user inputs.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_states():
    """Create a simple state setup for validation testing."""
    import openscvx as ox

    position = ox.State("position", shape=(2,))
    velocity = ox.State("velocity", shape=(1,))
    return [position, velocity]


# ===== Byof Key Validation =====


def test_empty_byof_is_valid(simple_states):
    """Empty byof dict should be valid."""
    from openscvx.expert import validate_byof

    byof = {}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_valid_keys_accepted(simple_states):
    """All valid keys should be accepted."""
    from openscvx.expert import validate_byof

    byof = {
        "dynamics": {},
        "nodal_constraints": [],
        "cross_nodal_constraints": [],
        "ctcs_constraints": [],
    }
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_invalid_key_raises(simple_states):
    """Unknown keys should raise ValueError."""
    from openscvx.expert import validate_byof

    byof = {"invalid_key": []}

    with pytest.raises(ValueError, match="Unknown byof keys.*invalid_key"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_multiple_invalid_keys_raises(simple_states):
    """Multiple unknown keys should be reported."""
    from openscvx.expert import validate_byof

    byof = {"bad_key1": [], "bad_key2": {}}

    with pytest.raises(ValueError, match="Unknown byof keys"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== Dynamics Validation =====


def test_valid_dynamics_accepted(simple_states):
    """Valid dynamics functions should pass validation."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def velocity_dynamics(x, u, node, params):
        return jnp.array([1.0])

    byof = {"dynamics": {"velocity": velocity_dynamics}}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_wrong_state_name(simple_states):
    """Dynamics for non-existent state should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_dynamics(x, u, node, params):
        return jnp.array([1.0])

    byof = {"dynamics": {"nonexistent_state": bad_dynamics}}

    with pytest.raises(ValueError, match="does not match any state name"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_not_callable(simple_states):
    """Non-callable dynamics should raise TypeError."""
    from openscvx.expert import validate_byof

    byof = {"dynamics": {"velocity": "not a function"}}

    with pytest.raises(TypeError, match="must be callable"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_wrong_signature_too_few(simple_states):
    """Dynamics with too few parameters should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_dynamics(x, u):  # Missing node, params
        return jnp.array([1.0])

    byof = {"dynamics": {"velocity": bad_dynamics}}

    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_wrong_signature_too_many(simple_states):
    """Dynamics with too many parameters should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_dynamics(x, u, node, params, extra):
        return jnp.array([1.0])

    byof = {"dynamics": {"velocity": bad_dynamics}}

    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_fails_on_call(simple_states):
    """Dynamics that fails when called should raise."""
    from openscvx.expert import validate_byof

    def bad_dynamics(x, u, node, params):
        raise RuntimeError("Intentional failure")

    byof = {"dynamics": {"velocity": bad_dynamics}}

    with pytest.raises(ValueError, match="failed on test call"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_wrong_output_shape(simple_states):
    """Dynamics returning wrong shape should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_dynamics(x, u, node, params):
        return jnp.array([1.0, 2.0])  # Should be (1,) not (2,)

    byof = {"dynamics": {"velocity": bad_dynamics}}

    with pytest.raises(ValueError, match="returned shape.*expected"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_dynamics_not_differentiable(simple_states):
    """Dynamics using numpy instead of jax should raise."""
    from openscvx.expert import validate_byof

    def bad_dynamics(x, u, node, params):
        # Using numpy linalg instead of jax - not differentiable!
        return np.array([float(np.linalg.norm(x))])

    byof = {"dynamics": {"velocity": bad_dynamics}}

    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== Nodal Constraint Validation =====


def test_valid_nodal_constraint_accepted(simple_states):
    """Valid nodal constraint should pass validation."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def constraint(x, u, node, params):
        return x[0] - 10.0

    byof = {"nodal_constraints": [constraint]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_nodal_constraint_not_callable(simple_states):
    """Non-callable nodal constraint should raise."""
    from openscvx.expert import validate_byof

    byof = {"nodal_constraints": ["not a function"]}

    with pytest.raises(TypeError, match="nodal_constraints\\[0\\] must be callable"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_nodal_constraint_wrong_signature(simple_states):
    """Nodal constraint with wrong signature should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_constraint(x, u):  # Missing node, params
        return x[0]

    byof = {"nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_nodal_constraint_fails_on_call(simple_states):
    """Nodal constraint that fails when called should raise."""
    from openscvx.expert import validate_byof

    def bad_constraint(x, u, node, params):
        raise RuntimeError("Intentional failure")

    byof = {"nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="nodal_constraints\\[0\\] failed on test call"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_nodal_constraint_not_differentiable(simple_states):
    """Nodal constraint using numpy should raise."""
    from openscvx.expert import validate_byof

    def bad_constraint(x, u, node, params):
        # Using numpy instead of jax.numpy
        return np.array([x[0] - 10.0])

    byof = {"nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_nodal_constraint_vector_output_accepted(simple_states):
    """Nodal constraint can return vector (not just scalar)."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def vector_constraint(x, u, node, params):
        return jnp.array([x[0] - 10.0, x[1] - 5.0])

    byof = {"nodal_constraints": [vector_constraint]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== Cross-Nodal Constraint Validation =====


def test_valid_cross_nodal_constraint_accepted(simple_states):
    """Valid cross-nodal constraint should pass validation."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def constraint(X, U, params):
        return jnp.sum(X[:, 0])

    byof = {"cross_nodal_constraints": [constraint]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_cross_nodal_constraint_not_callable(simple_states):
    """Non-callable cross-nodal constraint should raise."""
    from openscvx.expert import validate_byof

    byof = {"cross_nodal_constraints": ["not a function"]}

    with pytest.raises(TypeError, match="cross_nodal_constraints\\[0\\] must be callable"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_cross_nodal_constraint_wrong_signature(simple_states):
    """Cross-nodal constraint with wrong signature should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def bad_constraint(X, U):  # Missing params
        return jnp.sum(X[:, 0])

    byof = {"cross_nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="must have signature f\\(X, U, params\\)"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_cross_nodal_constraint_fails_on_call(simple_states):
    """Cross-nodal constraint that fails when called should raise."""
    from openscvx.expert import validate_byof

    def bad_constraint(X, U, params):
        raise RuntimeError("Intentional failure")

    byof = {"cross_nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="cross_nodal_constraints\\[0\\] failed on test call"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_cross_nodal_constraint_not_differentiable(simple_states):
    """Cross-nodal constraint using numpy should raise."""
    from openscvx.expert import validate_byof

    def bad_constraint(X, U, params):
        # Using numpy linalg instead of jax - not differentiable!
        return float(np.linalg.norm(X[:, 0]))

    byof = {"cross_nodal_constraints": [bad_constraint]}

    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== CTCS Constraint Validation =====


def test_valid_ctcs_constraint_accepted(simple_states):
    """Valid CTCS constraint should pass validation."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "penalty": "square",
    }
    byof = {"ctcs_constraints": [ctcs_spec]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_not_dict(simple_states):
    """CTCS constraint not a dict should raise."""
    from openscvx.expert import validate_byof

    byof = {"ctcs_constraints": ["not a dict"]}

    with pytest.raises(TypeError, match="ctcs_constraints\\[0\\] must be a dict"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_missing_constraint_fn(simple_states):
    """CTCS constraint missing constraint_fn should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {"penalty": "square"}  # Missing constraint_fn
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="missing required key 'constraint_fn'"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_fn_not_callable(simple_states):
    """CTCS constraint_fn not callable should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {"constraint_fn": "not a function"}
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(TypeError, match="constraint_fn.*must be callable"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_fn_wrong_signature(simple_states):
    """CTCS constraint_fn with wrong signature should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    ctcs_spec = {"constraint_fn": lambda x, u: x[0] - 10.0}  # Missing node, params
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_fn_fails_on_call(simple_states):
    """CTCS constraint_fn that fails when called should raise."""
    from openscvx.expert import validate_byof

    def bad_fn(x, u, node, params):
        raise RuntimeError("Intentional failure")

    ctcs_spec = {"constraint_fn": bad_fn}
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="constraint_fn.*failed on test call"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_fn_not_scalar(simple_states):
    """CTCS constraint_fn returning non-scalar should raise."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    ctcs_spec = {"constraint_fn": lambda x, u, node, params: jnp.array([x[0] - 10.0, x[1] - 5.0])}
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="must return a scalar"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_constraint_fn_not_differentiable(simple_states):
    """CTCS constraint_fn using numpy should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {"constraint_fn": lambda x, u, node, params: float(np.sum(x))}
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="not differentiable with JAX"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


@pytest.mark.parametrize("penalty", ["square", "l1", "huber"])
def test_ctcs_penalty_builtin_accepted(simple_states, penalty):
    """Built-in penalty functions should be accepted."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "penalty": penalty,
    }
    byof = {"ctcs_constraints": [ctcs_spec]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_penalty_custom_callable_accepted(simple_states):
    """Custom callable penalty should be accepted."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def custom_penalty(r):
        return jnp.maximum(r, 0.0) ** 3

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "penalty": custom_penalty,
    }
    byof = {"ctcs_constraints": [ctcs_spec]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_penalty_invalid_string(simple_states):
    """Invalid penalty string should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "penalty": "invalid_penalty",
    }
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="must be 'square', 'l1', 'huber', or a callable"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_penalty_custom_fails(simple_states):
    """Custom penalty that fails should raise."""
    from openscvx.expert import validate_byof

    def bad_penalty(r):
        raise RuntimeError("Intentional failure")

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "penalty": bad_penalty,
    }
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="penalty.*custom function failed"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_bounds_valid(simple_states):
    """Valid bounds should be accepted."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "bounds": (0.0, 1e-4),
    }
    byof = {"ctcs_constraints": [ctcs_spec]}
    # Should not raise
    validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_bounds_not_tuple_or_list(simple_states):
    """Bounds that is not tuple/list should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "bounds": "not a tuple",
    }
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="bounds.*must be a \\(min, max\\) tuple"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_bounds_wrong_length(simple_states):
    """Bounds with wrong length should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "bounds": (0.0, 1e-4, 1.0),  # Too many elements
    }
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="bounds.*must be a \\(min, max\\) tuple"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_ctcs_bounds_min_greater_than_max(simple_states):
    """Bounds with min > max should raise."""
    from openscvx.expert import validate_byof

    ctcs_spec = {
        "constraint_fn": lambda x, u, node, params: x[0] - 10.0,
        "bounds": (1.0, 0.0),  # min > max
    }
    byof = {"ctcs_constraints": [ctcs_spec]}

    with pytest.raises(ValueError, match="min.*must be <= max"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== Multiple Constraints =====


def test_multiple_nodal_constraints_index_correctly(simple_states):
    """Error messages should have correct indices for multiple constraints."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    def good_constraint(x, u, node, params):
        return x[0] - 10.0

    def bad_constraint(x, u):  # Wrong signature
        return x[0]

    byof = {"nodal_constraints": [good_constraint, bad_constraint]}

    with pytest.raises(ValueError, match="nodal_constraints\\[1\\]"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


def test_multiple_ctcs_constraints_index_correctly(simple_states):
    """Error messages should have correct indices for multiple CTCS constraints."""
    import jax.numpy as jnp

    from openscvx.expert import validate_byof

    good_spec = {"constraint_fn": lambda x, u, node, params: x[0] - 10.0}
    bad_spec = {"constraint_fn": "not a function"}

    byof = {"ctcs_constraints": [good_spec, bad_spec]}

    with pytest.raises(TypeError, match="ctcs_constraints\\[1\\]"):
        validate_byof(byof, simple_states, n_x=3, n_u=1)


# ===== Integration with Problem =====


def test_problem_validates_byof_at_initialization():
    """Problem should validate byof when initialize() is called."""
    import jax.numpy as jnp

    import openscvx as ox

    position = ox.State("position", shape=(2,))
    position.min = np.array([0.0, 0.0])
    position.max = np.array([10.0, 10.0])
    position.initial = np.array([0.0, 10.0])
    position.final = np.array([10.0, 5.0])

    velocity = ox.State("velocity", shape=(1,))
    velocity.min = np.array([0.0])
    velocity.max = np.array([10.0])
    velocity.initial = np.array([0.0])

    theta = ox.Control("theta", shape=(1,))
    theta.min = np.array([0.0])
    theta.max = np.array([np.pi / 2])

    # Only define position symbolically, velocity will be in byof
    dynamics = {"position": ox.Concat(velocity[0], velocity[0])}

    time = ox.Time(initial=0.0, final=1.0, min=0.0, max=2.0)

    # Invalid byof - dynamics with wrong signature (missing node, params)
    byof = {"dynamics": {"velocity": lambda x, u: jnp.array([1.0])}}

    # Validation happens during Problem construction (when lowering)
    with pytest.raises(ValueError, match="must have signature f\\(x, u, node, params\\)"):
        problem = ox.Problem(
            dynamics=dynamics,
            constraints=[],
            states=[position, velocity],
            controls=[theta],
            time=time,
            N=2,
            byof=byof,
        )


def test_problem_accepts_valid_byof():
    """Problem should accept valid byof without errors during validation."""
    import jax.numpy as jnp

    import openscvx as ox
    from openscvx.expert import validate_byof

    # Create simple states
    position = ox.State("position", shape=(2,))
    velocity = ox.State("velocity", shape=(1,))
    states = [position, velocity]

    # Valid byof with proper dynamics signature
    byof = {
        "dynamics": {"velocity": lambda x, u, node, params: jnp.array([1.0])},
        "nodal_constraints": [lambda x, u, node, params: x[0] - 10.0],
        "ctcs_constraints": [
            {"constraint_fn": lambda x, u, node, params: x[1] - 5.0, "penalty": "square"}
        ],
    }

    # Direct validation should pass
    # n_x = 2 (position) + 1 (velocity) = 3, n_u = 1 (theta)
    validate_byof(byof, states, n_x=3, n_u=1)

    # Note: Full end-to-end integration testing is covered by test_brachistochrone.py::test_byof
    # This test just verifies the validation layer accepts valid byof specifications
