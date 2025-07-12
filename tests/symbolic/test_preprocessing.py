import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import Add, Constant
from openscvx.backend.preprocessing import collect_and_assign_slices
from openscvx.backend.state import State


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
