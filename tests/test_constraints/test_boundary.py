import pytest
import numpy as np

from openscvx.backend.state import State, Free, Maximize, Minimize, Fix


@pytest.mark.parametrize("attr", ["initial", "final"])
def test_valid_boundary_types_parsing(attr):
    state = State("x", shape=(5,))
    data = np.array([1, Free(2), Minimize(3), Maximize(4), Fix(5)], dtype=object)

    setattr(state, attr, data)

    expected_values = [1, 2, 3, 4, 5]
    expected_types = ["Fix", "Free", "Minimize", "Maximize", "Fix"]

    actual_values = getattr(state, f"_{attr}")
    actual_types = getattr(state, f"{attr}_type")

    assert np.allclose(actual_values, expected_values)
    assert (actual_types == expected_types).all()


@pytest.mark.parametrize(
    "attr, bad_input, error_fragment",
    [
        ("initial", np.array([1, 2, "Fixed", 4, 5], dtype=object), "Fixed"),
        ("initial", np.array([1, "Freee", 3, 4, 5], dtype=object), "Freee"),
        ("final",   np.array([1, "Minim", 3, 4, 5], dtype=object), "Minim"),
        ("final",   np.array(["Max", 2, 3, 4, 5], dtype=object), "Max"),
    ]
)
def test_invalid_boundary_type_raises(attr, bad_input, error_fragment):
    state = State("x", shape=(5,))
    with pytest.raises(ValueError, match=error_fragment):
        setattr(state, attr, bad_input)


@pytest.mark.parametrize("attr", ["initial", "final"])
def test_shape_mismatch_raises(attr):
    state = State("x", shape=(4,))  # expects shape (4,)
    bad_input = np.array([1, 2, 3, 4, 5], dtype=object)  # shape mismatch
    with pytest.raises(ValueError):
        setattr(state, attr, bad_input)


@pytest.mark.parametrize("attr, type_wrapper, expected_type", [
    ("initial", Free(1.0), "Free"),
    ("final", Minimize(2.0), "Minimize"),
])
def test_single_type_assignment_correct(attr, type_wrapper, expected_type):
    state = State("x", shape=(3,))
    setattr(state, attr, np.array([type_wrapper] * 3))
    actual_types = getattr(state, f"{attr}_type")
    assert (actual_types == [expected_type] * 3).all()