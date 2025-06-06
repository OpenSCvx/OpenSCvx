import numpy as np
import jax.numpy as jnp
import pytest

from openscvx.backend.state import State, Free, Minimize
from openscvx.backend.control import Control

@pytest.mark.parametrize("shape", [(3,), (2, 4), (5, 5, 5)])
def test_state_creation(shape):
    # Test that a State object can be created
    state = State("test_state", shape=shape)
    assert state.name == "test_state"
    assert state.shape == shape  # Fix: Match the parameterized shape
    assert isinstance(state, State)

@pytest.mark.parametrize("shapes", [[(3,), (2,)], [(4, 4), (5, 5)]])
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
    assert state.true_state.shape == (n_main + n_new,)
    assert state.augmented_state.shape == (0,)

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
    assert state.true_state.shape == shape_main
    assert state.augmented_state.shape == shape_aug
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
    assert control.true_control.shape == shape_main
    assert control.augmented_control.shape == shape_aug
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
    assert control.true_control.shape == (n_main + n_new,)
    assert control.augmented_control.shape == (0,)

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