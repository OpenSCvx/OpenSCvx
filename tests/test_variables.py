import numpy as np
import jax.numpy as jnp
import pytest

from openscvx.backend.state import State, Free, Minimize, Fix
from openscvx.backend.control import Control
from openscvx.config import SimConfig, get_affine_scaling_matrices

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

class DummyState:
    def __init__(self, min_val, max_val):
        self.min = np.array(min_val)
        self.max = np.array(max_val)

class DummyControl:
    def __init__(self, min_val, max_val):
        self.min = np.array(min_val)
        self.max = np.array(max_val)

def test_scaling_overrides():
    # Dummy bounds
    state_min = [0, -1, -2]
    state_max = [10, 1, 2]
    control_min = [-2, -3]
    control_max = [2, 3]

    x = DummyState(state_min, state_max)
    x_prop = DummyState(state_min, state_max)
    u = DummyControl(control_min, control_max)

    # No overrides: should use min/max for all
    sim = SimConfig(
        x=x,
        x_prop=x_prop,
        u=u,
        total_time=1.0,
        idx_x_true=slice(0,3),
        idx_x_true_prop=slice(0,3),
        idx_u_true=slice(0,2),
        idx_t=slice(0,1),
        idx_y=slice(0,0),
        idx_y_prop=slice(0,0),
        idx_s=slice(0,0),
    )
    S_x_expected, c_x_expected = get_affine_scaling_matrices(3, np.array(state_min), np.array(state_max))
    S_u_expected, c_u_expected = get_affine_scaling_matrices(2, np.array(control_min), np.array(control_max))
    np.testing.assert_allclose(sim.S_x, S_x_expected)
    np.testing.assert_allclose(sim.c_x, c_x_expected)
    np.testing.assert_allclose(sim.S_u, S_u_expected)
    np.testing.assert_allclose(sim.c_u, c_u_expected)

    # With custom scaling overrides for state 0 and control 1
    scaling_x_overrides = [
        (5, -5, 0),                # Custom scale for state 0
        ([2, 3], [-2, -3], [1, 2]) # Custom scale for states 1 and 2
    ]
    scaling_u_overrides = [
        (10, -10, 0),              # Custom scale for control 0
        (30, -30, 1)               # Custom scale for control 1
    ]
    sim2 = SimConfig(
        x=x,
        x_prop=x_prop,
        u=u,
        total_time=1.0,
        idx_x_true=slice(0,3),
        idx_x_true_prop=slice(0,3),
        idx_u_true=slice(0,2),
        idx_t=slice(0,1),
        idx_y=slice(0,0),
        idx_y_prop=slice(0,0),
        idx_s=slice(0,0),
        scaling_x_overrides=scaling_x_overrides,
        scaling_u_overrides=scaling_u_overrides,
    )
    # Expected: all states/controls use custom scaling
    S_x_expected2, c_x_expected2 = get_affine_scaling_matrices(3, np.array([-5, -2, -3]), np.array([5, 2, 3]))
    S_u_expected2, c_u_expected2 = get_affine_scaling_matrices(2, np.array([-10, -30]), np.array([10, 30]))
    np.testing.assert_allclose(sim2.S_x, S_x_expected2)
    np.testing.assert_allclose(sim2.c_x, c_x_expected2)
    np.testing.assert_allclose(sim2.S_u, S_u_expected2)
    np.testing.assert_allclose(sim2.c_u, c_u_expected2)

    # Partial override: only state 1, rest use min/max
    scaling_x_overrides_partial = [
        (100, -100, 1)
    ]
    sim3 = SimConfig(
        x=x,
        x_prop=x_prop,
        u=u,
        total_time=1.0,
        idx_x_true=slice(0,3),
        idx_x_true_prop=slice(0,3),
        idx_u_true=slice(0,2),
        idx_t=slice(0,1),
        idx_y=slice(0,0),
        idx_y_prop=slice(0,0),
        idx_s=slice(0,0),
        scaling_x_overrides=scaling_x_overrides_partial,
    )
    lower_x = np.array([0, -100, -2])
    upper_x = np.array([10, 100, 2])
    S_x_expected3, c_x_expected3 = get_affine_scaling_matrices(3, lower_x, upper_x)
    np.testing.assert_allclose(sim3.S_x, S_x_expected3)
    np.testing.assert_allclose(sim3.c_x, c_x_expected3)