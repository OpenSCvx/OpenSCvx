import numpy as np
import pytest

from openscvx.backend.augmentation import augment_dynamics_with_ctcs
from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    PositivePart,
    Square,
    ctcs,
)
from openscvx.backend.state import State


def test_augment_no_constraints():
    """Test augmentation with no constraints."""
    x = State("x", (2,))
    xdot = Add(x, Constant(np.ones(2)))
    states = [x]

    xdot_aug, states_aug, node_constraints = augment_dynamics_with_ctcs(xdot, states, [])

    # No augmentation should occur
    assert xdot_aug is xdot
    assert len(states_aug) == 1
    assert states_aug[0] is x
    assert len(node_constraints) == 0


def test_augment_only_nodal_constraints():
    """Test with only regular (nodal) constraints."""
    x = State("x", (2,))
    xdot = Add(x, Constant(np.ones(2)))
    states = [x]

    # Regular constraint
    constraint = x[0] <= 1.0

    xdot_aug, states_aug, node_constraints = augment_dynamics_with_ctcs(xdot, states, [constraint])

    # No dynamics augmentation, but constraint should be in nodal list
    assert xdot_aug is xdot
    assert len(states_aug) == 1
    assert len(node_constraints) == 1
    assert node_constraints[0] is constraint


def test_augment_single_ctcs_constraint():
    """Test augmentation with a single CTCS constraint."""
    x = State("x", (2,))
    u = Control("u", (1,))
    xdot = Add(x, u)
    states = [x]

    # CTCS constraint
    constraint = ctcs(x[0] <= 1.0, penalty="squared_relu")

    xdot_aug, states_aug, node_constraints = augment_dynamics_with_ctcs(xdot, states, [constraint])

    # Should have augmented dynamics and new state
    assert isinstance(xdot_aug, Concat)
    assert len(states_aug) == 2  # original + 1 augmented
    assert states_aug[0] is x
    assert isinstance(states_aug[1], State)
    assert states_aug[1].name == "_ctcs_aug_0"
    assert states_aug[1].shape == (1,)
    assert len(node_constraints) == 0


def test_augment_multiple_ctcs_constraints():
    """Test augmentation with multiple CTCS constraints."""
    x = State("x", (3,))
    xdot = x * 2.0
    states = [x]

    # Multiple CTCS constraints with different penalties
    c1 = ctcs(x[0] <= 1.0, penalty="squared_relu")
    c2 = ctcs(x[1] >= -1.0, penalty="huber")
    c3 = ctcs(x[2] == 0.0, penalty="smooth_relu")

    xdot_aug, states_aug, node_constraints = augment_dynamics_with_ctcs(xdot, states, [c1, c2, c3])

    # Should have 3 augmented states
    assert isinstance(xdot_aug, Concat)
    assert len(states_aug) == 4  # original + 3 augmented
    assert states_aug[0] is x

    # Check augmented state names
    for i in range(3):
        assert states_aug[i + 1].name == f"_ctcs_aug_{i}"
        assert states_aug[i + 1].shape == (1,)

    assert len(node_constraints) == 0


def test_augment_mixed_constraints():
    """Test with both CTCS and regular constraints."""
    x = State("x", (2,))
    y = State("y", (1,))
    xdot = Concat(x, y)
    states = [x, y]

    # Mix of CTCS and regular constraints
    c1 = ctcs(x[0] <= 1.0, penalty="squared_relu")
    c2 = y == 0.0  # Regular constraint
    c3 = ctcs(x[1] >= -2.0, penalty="huber")
    c4 = x[0] + x[1] <= 3.0  # Regular constraint

    xdot_aug, states_aug, node_constraints = augment_dynamics_with_ctcs(
        xdot, states, [c1, c2, c3, c4]
    )

    # Should have 2 augmented states (for 2 CTCS constraints)
    assert isinstance(xdot_aug, Concat)
    assert len(states_aug) == 4  # 2 original + 2 augmented

    # Check original states are preserved
    assert states_aug[0] is x
    assert states_aug[1] is y

    # Check augmented states
    assert states_aug[2].name == "_ctcs_aug_0"
    assert states_aug[3].name == "_ctcs_aug_1"

    # Check nodal constraints (only regular ones)
    assert len(node_constraints) == 2
    assert c2 in node_constraints
    assert c4 in node_constraints


def test_augment_preserves_state_list_reference():
    """Test that the states list is modified in-place."""
    x = State("x", (2,))
    xdot = x * 0.1
    states = [x]
    original_list = states  # Keep reference to original list

    constraint = ctcs(x[0] <= 1.0)

    xdot_aug, states_aug, _ = augment_dynamics_with_ctcs(xdot, states, [constraint])

    # The returned states should be the same list object
    assert states_aug is original_list
    assert len(original_list) == 2  # Modified in-place


def test_augment_penalty_expression_structure():
    """Test that the penalty expressions are correctly structured."""
    x = State("x", (1,))
    xdot = x
    states = [x]

    # Create CTCS with squared_relu penalty
    constraint = ctcs(x <= 1.0, penalty="squared_relu")

    xdot_aug, _, _ = augment_dynamics_with_ctcs(xdot, states, [constraint])

    # Check structure of augmented dynamics
    assert isinstance(xdot_aug, Concat)
    assert len(xdot_aug.exprs) == 2

    # The second expression should be the penalty
    penalty_expr = xdot_aug.exprs[1]
    assert isinstance(penalty_expr, Square)
    assert isinstance(penalty_expr.x, PositivePart)


def test_augment_invalid_constraint_type_raises():
    """Test that invalid constraint types raise an error."""
    x = State("x", (1,))
    xdot = x
    states = [x]

    # Not a Constraint or CTCS
    invalid = Add(x, Constant(1.0))

    with pytest.raises(ValueError) as exc:
        augment_dynamics_with_ctcs(xdot, states, [invalid])

    assert "Constraints must be `Constraint` or `CTCS`" in str(exc.value)


def test_augment_empty_states_list():
    """Test augmentation with empty states list."""
    xdot = Constant(np.array([1.0, 2.0]))
    states = []

    # CTCS constraint on a constant (unusual but valid)
    constraint = ctcs(Constant(1.0) <= 2.0)

    xdot_aug, states_aug, _ = augment_dynamics_with_ctcs(xdot, states, [constraint])

    # Should create one augmented state
    assert len(states_aug) == 1
    assert states_aug[0].name == "_ctcs_aug_0"


def test_augment_with_different_penalties():
    """Test that different penalty types are correctly applied."""
    x = State("x", (1,))
    xdot = x
    states = [x]

    penalties = ["squared_relu", "huber", "smooth_relu"]
    constraints = [ctcs(x <= float(i), penalty=p) for i, p in enumerate(penalties)]

    xdot_aug, states_aug, _ = augment_dynamics_with_ctcs(xdot, states, constraints)

    # Should have 3 augmented states with appropriate penalties
    assert len(states_aug) == 4
    assert isinstance(xdot_aug, Concat)
    assert len(xdot_aug.exprs) == 4  # original + 3 penalties

    # Each penalty expression should be different based on the penalty type
    # (The actual structure validation would depend on the penalty implementations)
