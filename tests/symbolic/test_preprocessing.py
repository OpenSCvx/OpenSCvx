import jax.numpy as jnp
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import Add, Constant
from openscvx.backend.preprocessing import (
    collect_and_assign_slices,
    validate_constraints_at_root,
    validate_variable_names,
)
from openscvx.backend.state import State


def test_unique_names_passes():
    a = State("a", (2,))
    b = State("b", (2,))
    c = Control("c", (1,))
    validate_variable_names([Add(a, b), c])  # no error


def test_duplicate_names_raises():
    a1 = State("x", (2,))
    a2 = State("x", (3,))
    with pytest.raises(ValueError):
        validate_variable_names([a1, a2])


def test_reserved_prefix_raises():
    bad = State("_hidden", (1,))
    with pytest.raises(ValueError):
        validate_variable_names([bad])


def test_reserved_names_collision():
    s = State("foo", (1,))
    with pytest.raises(ValueError):
        validate_variable_names([s], reserved_names={"foo", "bar"})


def test_collect_single_state():
    x = State("x", (4,))
    expr = Add(x, Constant(1.0))
    collect_and_assign_slices([expr])
    assert x._slice == slice(0, 4)


def test_collect_multiple_states_preserves_order():
    a = State("a", (2,))
    b = State("b", (3,))
    collect_and_assign_slices([Add(a, b)])
    assert slice(0, 2, None) == slice(0, 2)
    assert a._slice == slice(0, 2)
    assert b._slice == slice(2, 5)


def test_collect_states_and_controls_separate_namespaces():
    s1 = State("s1", (2,))
    c1 = Control("c1", (3,))
    collect_and_assign_slices([Add(s1, c1)])
    # states live in x; controls live in u
    assert s1._slice == slice(0, 2)
    assert c1._slice == slice(0, 3)


def test_states_and_controls_independent_offsets():
    # two states but only one control
    s1 = State("s1", (2,))
    s2 = State("s2", (1,))
    c1 = Control("c1", (2,))
    exprs = [Add(s1, s2), Add(c1, Constant(0.0))]
    collect_and_assign_slices(exprs)
    # states: offsets 0→2, 2→3
    assert s1._slice == slice(0, 2)
    assert s2._slice == slice(2, 3)
    # controls: offset resets to zero
    assert c1._slice == slice(0, 2)


def test_manual_slice_shape_mismatch_raises():
    # Create a State of dimension 3, but give it a slice of length 2
    s = State("s", (3,))
    s._slice = slice(0, 2)

    with pytest.raises(ValueError) as excinfo:
        collect_and_assign_slices([s])

    msg = str(excinfo.value)
    assert "Manual slice for 's'" in msg
    assert "length 2" in msg
    assert "(3,)" in msg


def test_idempotent_on_repeat_calls():
    s = State("s", (3,))
    collect_and_assign_slices([Add(s, Constant(0.0))])
    first = s._slice
    collect_and_assign_slices([Add(s, Constant(0.0))])
    assert s._slice is first


def test_manual_slice_assignment():
    s = State("s", (2,))
    s._slice = slice(0, 2)
    t = State("t", (3,))  # left to auto-assign
    collect_and_assign_slices([Add(s, t)])
    assert s._slice == slice(0, 2)
    assert t._slice == slice(2, 5)


def test_invalid_manual_slice_assignment_nonzero_start():
    # starts at nonzero:
    s = State("s", (2,))
    s._slice = slice(1, 3)
    with pytest.raises(ValueError):
        collect_and_assign_slices([s])


def test_invalid_manual_slice_assignment_gaps():
    # gap/overlap:
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(3, 5)
    with pytest.raises(ValueError):
        collect_and_assign_slices([Add(a, b)])


def test_root_constraint_passes():
    # a == 5  is a top‐level constraint → OK
    a = Constant(jnp.array([1.0, 2.0]))
    c1 = a == 5
    c2 = a <= jnp.array([3.0, 4.0])
    # should not raise
    validate_constraints_at_root(c1)
    validate_constraints_at_root(c2)


def test_nested_constraint_raises():
    # Add(a, (b == 3))  nests a constraint under Add → should error
    a = Constant(jnp.array([1.0, 2.0]))
    b = Constant(jnp.array([3.0, 4.0]))
    nested = Add(a, b == 3)

    with pytest.raises(ValueError) as exc:
        validate_constraints_at_root(nested)
    msg = str(exc.value)
    assert "Nested Constraint found at depth 1" in msg
    assert "constraints must only appear as top‐level roots" in msg
