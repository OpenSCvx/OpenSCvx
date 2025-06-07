import numpy as np
import jax.numpy as jnp
import pytest

from openscvx.backend.state import State, Free, Minimize, Fix
from openscvx.backend.control import Control

@pytest.mark.parametrize("shape", [(3,), (2, 4), (5, 5, 5)])
def test_state_creation(shape):
    # Test that a State object can be created
    state = State("test_state", shape=shape)
    assert state.name == "test_state"
    assert state.shape == shape  # Fix: Match the parameterized shape
    assert isinstance(state, State)

@pytest.mark.parametrize("shapes", [[(3,), (2,)], [(4, ), (5, )]])
def test_append_non_augmented_state(shapes):
    shape_main, shape_new = shapes
    n_main = shape_main[0]
    n_new = shape_new[0]
    m = 10  # number of time steps for guess

    state = State("test_state", shape=shape_main)
    non_aug_state = State("non_augmented", shape=shape_new)

    # Assign attributes for both states
    state.min = np.full((n_main,), -1.0)
    state.max = np.full((n_main,), 1.0)
    state.guess = np.full((m, n_main), 0.5)
    state.initial = np.full((n_main,), 0.0)
    state.final = np.full((n_main,), 1.0)

    non_aug_state.min = np.full((n_new,), -10.0)
    non_aug_state.max = np.full((n_new,), 10.0)
    non_aug_state.guess = np.full((m, n_new), 5.0)
    non_aug_state.initial = np.full((n_new,), -5.0)
    non_aug_state.final = np.full((n_new,), 5.0)

    # Append without marking as augmented
    state.append(non_aug_state)

    # Shape checks
    assert state.shape == (n_main + n_new,)
    assert state.true.shape == (n_main + n_new,)
    assert state.augmented.shape == (0,)

    # Value checks
    np.testing.assert_array_equal(state.min, np.concatenate([
        np.full((n_main,), -1.0),
        np.full((n_new,), -10.0)
    ]))

    np.testing.assert_array_equal(state.max, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_new,), 10.0)
    ]))

    np.testing.assert_array_equal(state.guess, np.concatenate([
        np.full((m, n_main), 0.5),
        np.full((m, n_new), 5.0)
    ], axis=1))

    np.testing.assert_array_equal(state.initial, np.concatenate([
        np.full((n_main,), 0.0),
        np.full((n_new,), -5.0)
    ]))

    np.testing.assert_array_equal(state.final, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_new,), 5.0)
    ]))


@pytest.mark.parametrize("shapes", [[(3,), (2,)], [(40,), (5,)]])
def test_append_augmented_state(shapes):
    shape_main, shape_aug = shapes
    n_main = shape_main[0]
    n_aug = shape_aug[0]
    m = 10  # number of time steps for guess

    state = State("test_state", shape=shape_main)
    augmented_state = State("augmented", shape=shape_aug)

    # Assign main state
    state.min = np.full((n_main,), -1.0)
    state.max = np.full((n_main,), 1.0)
    state.guess = np.full((m, n_main), 0.5)
    state.initial = np.full((n_main,), 0.0)
    state.final = np.full((n_main,), 1.0)

    # Assign augmented state
    augmented_state.min = np.full((n_aug,), -10.0)
    augmented_state.max = np.full((n_aug,), 10.0)
    augmented_state.guess = np.full((m, n_aug), 5.0)
    augmented_state.initial = np.full((n_aug,), -5.0)
    augmented_state.final = np.full((n_aug,), 5.0)

    # Append as augmented
    state.append(augmented_state, augmented=True)

    # Check shapes
    assert state.true.shape == shape_main
    assert state.augmented.shape == shape_aug
    assert state.shape == (n_main + n_aug,)

    # Check values
    np.testing.assert_array_equal(state.min, np.concatenate([
        np.full((n_main,), -1.0),
        np.full((n_aug,), -10.0)
    ]))

    np.testing.assert_array_equal(state.max, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_aug,), 10.0)
    ]))

    np.testing.assert_array_equal(state.guess, np.concatenate([
        np.full((m, n_main), 0.5),
        np.full((m, n_aug), 5.0)
    ], axis=1))

    np.testing.assert_array_equal(state.initial, np.concatenate([
        np.full((n_main,), 0.0),
        np.full((n_aug,), -5.0)
    ]))

    np.testing.assert_array_equal(state.final, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_aug,), 5.0)
    ]))

@pytest.mark.parametrize("field, values, min_val, max_val, should_raise, expected_index", [
    # Initial within bounds
    ("initial", [Fix(0.0), Fix(-1.5)], [-1.0, -2.0], [1.0, 2.0], False, None),

    # Final within bounds
    ("final", [Fix(0.5), Fix(1.0)], [-1.0, -2.0], [1.0, 2.0], False, None),

    # Initial below min (at index 0)
    ("initial", [Fix(-0.5), Fix(0.0)], [0.0, -1.0], [1.0, 1.0], True, 0),

    # Final above max (at index 1)
    ("final", [Fix(0.0), Fix(1.0)], [-1.0, -1.0], [1.0, 0.5], True, 1),
])
def test_state_fix_bounds_check(field, values, min_val, max_val, should_raise, expected_index):
    s = State("x", shape=(2,))
    s.min = np.array(min_val)
    s.max = np.array(max_val)

    if should_raise:
        with pytest.raises(ValueError) as excinfo:
            setattr(s, field, values)
        # Construct expected error message pattern
        val = values[expected_index].value
        i_str = expected_index if isinstance(expected_index, int) else expected_index[0]  # convert tuple index to int

        if val < min_val[i_str]:
            err_msg = f"{field.capitalize()} Fixed value at index {i_str} is lower then the min: {val} < {min_val[i_str]}"
        elif val > max_val[i_str]:
            err_msg = f"{field.capitalize()} Fixed value at index {i_str} is greater then the max: {val} > {max_val[i_str]}"
        assert err_msg in str(excinfo.value)
    else:
        setattr(s, field, values)

@pytest.mark.parametrize("field, values, min_val, should_raise, expected_index", [
    # No error
    ("initial", [Fix(0.0), Fix(-1.5)], [-1.0, -2.0], False, None),

    # Error below min
    ("initial", [Fix(-0.5), Fix(0.0)], [0.0, -1.0], True, 0),

    # Error below min (final)
    ("final", [Fix(-2.0), Fix(0.0)], [-1.0, -1.0], True, 0),
])
def test_state_fix_bounds_check_min_only(field, values, min_val, should_raise, expected_index):
    s = State("x", shape=(2,))
    setattr(s, field, values)

    if should_raise:
        with pytest.raises(ValueError) as excinfo:
            s.min = np.array(min_val)
        val = values[expected_index].value
        i_str = expected_index if isinstance(expected_index, int) else expected_index[0]
        err_msg = f"{field.capitalize()} Fixed value at index {i_str} is lower then the min: {val} < {min_val[i_str]}"
        assert err_msg in str(excinfo.value)
    else:
        s.min = np.array(min_val)
    
@pytest.mark.parametrize("field, values, max_val, should_raise, expected_index", [
    # No error
    ("initial", [Fix(0.0), Fix(-1.5)], [1.0, 2.0], False, None),

    # Error above max
    ("initial", [Fix(1.5), Fix(0.0)], [1.0, 1.0], True, 0),

    # Error above max (final)
    ("final", [Fix(0.0), Fix(1.5)], [1.0, 1.0], True, 1),
])
def test_state_fix_bounds_check_max_only(field, values, max_val, should_raise, expected_index):
    s = State("x", shape=(2,))
    setattr(s, field, values)

    if should_raise:
        with pytest.raises(ValueError) as excinfo:
            s.max = np.array(max_val)
        val = values[expected_index].value
        i_str = expected_index if isinstance(expected_index, int) else expected_index[0]
        err_msg = f"{field.capitalize()} Fixed value at index {i_str} is greater then the max: {val} > {max_val[i_str]}"
        assert err_msg in str(excinfo.value)
    else:
        s.max = np.array(max_val)

@pytest.mark.parametrize("shapes", [[(3,), (2,)], [(4,), (5,)]])
def test_append_augmented_control(shapes):
    shape_main, shape_aug = shapes
    n_main = shape_main[0]
    n_aug = shape_aug[0]
    m = 10  # time steps for guess

    control = Control("control_main", shape=shape_main)
    control_aug = Control("control_aug", shape=shape_aug)

    control.min = np.full((n_main,), -1.0)
    control.max = np.full((n_main,), 1.0)
    control.guess = np.full((m, n_main), 0.5)

    control_aug.min = np.full((n_aug,), -10.0)
    control_aug.max = np.full((n_aug,), 10.0)
    control_aug.guess = np.full((m, n_aug), 5.0)

    control.append(control_aug, augmented=True)

    # Check shape tracking
    assert control.true.shape == shape_main
    assert control.augmented.shape == shape_aug
    assert control.shape == (n_main + n_aug,)

    # Check concatenated values
    np.testing.assert_array_equal(control.min, np.concatenate([
        np.full((n_main,), -1.0),
        np.full((n_aug,), -10.0)
    ]))

    np.testing.assert_array_equal(control.max, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_aug,), 10.0)
    ]))

    np.testing.assert_array_equal(control.guess, np.concatenate([
        np.full((m, n_main), 0.5),
        np.full((m, n_aug), 5.0)
    ], axis=1))

@pytest.mark.parametrize("shapes", [[(3,), (2,)], [(4,), (5,)]])
def test_append_non_augmented_control(shapes):
    shape_main, shape_new = shapes
    n_main = shape_main[0]
    n_new = shape_new[0]
    m = 10  # time steps for guess

    control = Control("control_main", shape=shape_main)
    control_new = Control("control_new", shape=shape_new)

    control.min = np.full((n_main,), -1.0)
    control.max = np.full((n_main,), 1.0)
    control.guess = np.full((m, n_main), 0.5)

    control_new.min = np.full((n_new,), -2.0)
    control_new.max = np.full((n_new,), 2.0)
    control_new.guess = np.full((m, n_new), 1.5)

    control.append(control_new)

    # Check shape tracking
    assert control.shape == (n_main + n_new,)
    assert control.true.shape == (n_main + n_new,)
    assert control.augmented.shape == (0,)

    # Check concatenated values
    np.testing.assert_array_equal(control.min, np.concatenate([
        np.full((n_main,), -1.0),
        np.full((n_new,), -2.0)
    ]))

    np.testing.assert_array_equal(control.max, np.concatenate([
        np.full((n_main,), 1.0),
        np.full((n_new,), 2.0)
    ]))

    np.testing.assert_array_equal(control.guess, np.concatenate([
        np.full((m, n_main), 0.5),
        np.full((m, n_new), 1.5)
    ], axis=1))