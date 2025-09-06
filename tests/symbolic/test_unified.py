import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.state import State
from openscvx.backend.unified import UnifiedControl, UnifiedState, unify_controls, unify_states


# Test unify_states function
def test_unify_states_sorting():
    """Test that true states come before augmented states."""
    true_state1 = State("pos", (2,))
    true_state1.min = np.array([0.0, 1.0])
    
    aug_state = State("_slack", (1,))
    aug_state.min = np.array([5.0])
    
    true_state2 = State("vel", (2,))
    true_state2.min = np.array([2.0, 3.0])
    
    # Pass in mixed order - augmented state in the middle
    unified = unify_states([true_state1, aug_state, true_state2])
    
    # Should have total shape 5, true dim 4
    assert unified.shape == (5,)
    assert unified._true_dim == 4
    
    # Check that true states come first, then augmented
    expected_min = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    np.testing.assert_array_equal(unified.min, expected_min)


def test_unify_states_none_handling():
    """Test that None values are handled properly with defaults."""
    state1 = State("x", (2,))
    # Don't set min/max - they should default
    
    state2 = State("_aug", (1,))
    state2.min = np.array([1.0])
    
    unified = unify_states([state1, state2])
    
    # Should have -inf for state1's min values, 1.0 for state2
    expected_min = np.array([-np.inf, -np.inf, 1.0])
    np.testing.assert_array_equal(unified.min, expected_min)
    
    # Should have +inf for state1's max values 
    expected_max = np.array([np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(unified.max, expected_max)


def test_unify_states_underscore_naming():
    """Test various underscore naming patterns."""
    normal = State("position", (1,))
    normal.min = np.array([0.0])
    
    underscore_start = State("_slack_var", (1,))
    underscore_start.min = np.array([1.0])
    
    underscore_middle = State("some_var", (1,))  # Should be treated as normal
    underscore_middle.min = np.array([2.0])
    
    double_underscore = State("__private", (1,))  # Should be treated as augmented
    double_underscore.min = np.array([3.0])
    
    unified = unify_states([underscore_start, normal, double_underscore, underscore_middle])
    
    # True states: normal, underscore_middle (2 total)
    assert unified._true_dim == 2
    
    # Order should be: normal, underscore_middle, underscore_start, double_underscore
    expected_min = np.array([0.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(unified.min, expected_min)


# Test unify_controls function
def test_unify_controls_sorting():
    """Test that true controls come before augmented controls."""
    true_control1 = Control("thrust", (1,))
    true_control1.min = np.array([-10.0])
    
    aug_control = Control("_auxiliary", (1,))
    aug_control.min = np.array([0.0])
    
    true_control2 = Control("torque", (1,))  
    true_control2.min = np.array([-5.0])
    
    # Pass in mixed order
    unified = unify_controls([aug_control, true_control1, true_control2])
    
    assert unified.shape == (3,)
    assert unified._true_dim == 2
    
    # Should be: true_control1, true_control2, aug_control
    expected_min = np.array([-10.0, -5.0, 0.0])
    np.testing.assert_array_equal(unified.min, expected_min)


# Test UnifiedState properties and methods
def test_unified_state_properties():
    """Test true and augmented properties of UnifiedState."""
    true_state = State("x", (2,))
    true_state.min = np.array([0.0, 1.0])
    
    aug_state = State("_slack", (1,))
    aug_state.min = np.array([5.0])
    
    unified = unify_states([true_state, aug_state])
    
    # Test true property
    true_part = unified.true
    assert true_part.shape == (2,)
    np.testing.assert_array_equal(true_part.min, np.array([0.0, 1.0]))
    
    # Test augmented property  
    aug_part = unified.augmented
    assert aug_part.shape == (1,)
    np.testing.assert_array_equal(aug_part.min, np.array([5.0]))


def test_unified_state_append():
    """Test appending to UnifiedState."""
    state1 = State("x", (2,))
    state1.min = np.array([0.0, 1.0])
    
    unified = unify_states([state1])
    
    # Append as augmented state
    state2 = State("_aug", (1,))
    state2.min = np.array([5.0])
    unified.append(state2, augmented=True)
    
    assert unified.shape == (3,)
    assert unified._true_dim == 2  # Should not change
    np.testing.assert_array_equal(unified.min, np.array([0.0, 1.0, 5.0]))
    
    # Append scalar variable
    unified.append(min=-1.0, max=1.0, augmented=False)
    assert unified.shape == (4,)
    assert unified._true_dim == 3


def test_unified_state_slicing():
    """Test slicing UnifiedState."""
    state1 = State("x", (2,))
    state1.min = np.array([0.0, 1.0])
    
    state2 = State("_aug", (2,))
    state2.min = np.array([5.0, 6.0])
    
    unified = unify_states([state1, state2])
    
    # Get first 3 elements
    subset = unified[0:3]
    assert subset.shape == (3,)
    assert subset._true_dim == 2
    np.testing.assert_array_equal(subset.min, np.array([0.0, 1.0, 5.0]))


# Test UnifiedControl properties and methods
def test_unified_control_properties():
    """Test true and augmented properties of UnifiedControl."""
    true_control = Control("u", (1,))
    true_control.min = np.array([-1.0])
    
    aug_control = Control("_aux", (1,))
    aug_control.min = np.array([5.0])
    
    unified = unify_controls([true_control, aug_control])
    
    # Test true property
    true_part = unified.true
    assert true_part.shape == (1,)
    np.testing.assert_array_equal(true_part.min, np.array([-1.0]))
    
    # Test augmented property
    aug_part = unified.augmented
    assert aug_part.shape == (1,)
    np.testing.assert_array_equal(aug_part.min, np.array([5.0]))


# Test integration with arrays
def test_state_with_guess_arrays():
    """Test that guess arrays and initial/final conditions are handled properly."""
    state1 = State("x", (2,))
    state1.initial = np.array([1.0, 2.0])
    guess1 = np.random.rand(100, 2)
    state1.guess = guess1
    
    state2 = State("_aug", (1,))
    state2.initial = np.array([5.0])
    guess2 = np.random.rand(100, 1)
    state2.guess = guess2
    
    unified = unify_states([state1, state2])
    
    # Check guess arrays
    assert unified.guess.shape == (100, 3)
    np.testing.assert_array_equal(unified.guess[:, :2], guess1)
    np.testing.assert_array_equal(unified.guess[:, 2:], guess2)
    
    # Check initial conditions ordering: true states first, then augmented
    expected_initial = np.array([1.0, 2.0, 5.0])
    np.testing.assert_array_equal(unified.initial, expected_initial)