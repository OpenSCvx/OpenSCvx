"""Drop-in replacement dataclasses for unifying multiple State and Control objects.

This module provides UnifiedState and UnifiedControl classes that can hold
aggregated information from multiple State and Control objects while maintaining
compatibility with the existing optimization code that expects a single State and Control.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State


@dataclass
class UnifiedState:
    """Unified state that behaves like a single State but represents multiple States.

    This class provides a drop-in replacement for State that holds aggregated
    data from multiple State objects while preserving all the expected properties.

    Attributes:
        name (str): Name of the unified state
        shape (tuple): Combined shape of all states
        min (np.ndarray): Minimum bounds for all state variables
        max (np.ndarray): Maximum bounds for all state variables
        guess (np.ndarray): Initial guess trajectory for all state variables
        initial (np.ndarray): Initial conditions for all state variables
        final (np.ndarray): Final conditions for all state variables
        _initial (np.ndarray): Internal initial values
        _final (np.ndarray): Internal final values
        initial_type (np.ndarray): Initial condition types
        final_type (np.ndarray): Final condition types
        _true_dim (int): True dimensionality (excluding augmented states)
        _true_slice (slice): Slice for accessing true state variables
        _augmented_slice (slice): Slice for accessing augmented state variables
    """

    name: str
    shape: tuple
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    initial: Optional[np.ndarray] = None
    final: Optional[np.ndarray] = None
    _initial: Optional[np.ndarray] = None
    _final: Optional[np.ndarray] = None
    initial_type: Optional[np.ndarray] = None
    final_type: Optional[np.ndarray] = None
    _true_dim: int = 0
    _true_slice: Optional[slice] = None
    _augmented_slice: Optional[slice] = None

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def true(self) -> "UnifiedState":
        """Get the true state variables (excluding augmented states)."""
        return self[self._true_slice]

    @property
    def augmented(self) -> "UnifiedState":
        """Get the augmented state variables."""
        return self[self._augmented_slice]

    def append(
        self,
        other=None,
        *,
        min=-np.inf,
        max=np.inf,
        guess=0.0,
        initial=0.0,
        final=0.0,
        augmented=False,
    ):
        """Append another state or create a new state variable.

        This method mimics the State.append interface for compatibility.
        """
        if isinstance(other, (State, UnifiedState)):
            # Append another state object
            new_shape = (self.shape[0] + other.shape[0],)

            # Update bounds
            if self.min is not None and other.min is not None:
                new_min = np.concatenate([self.min, other.min])
            else:
                new_min = self.min

            if self.max is not None and other.max is not None:
                new_max = np.concatenate([self.max, other.max])
            else:
                new_max = self.max

            # Update guess
            if self.guess is not None and other.guess is not None:
                new_guess = np.concatenate([self.guess, other.guess], axis=1)
            else:
                new_guess = self.guess

            # Update initial/final conditions
            if self.initial is not None and other.initial is not None:
                new_initial = np.concatenate([self.initial, other.initial])
            else:
                new_initial = self.initial

            if self.final is not None and other.final is not None:
                new_final = np.concatenate([self.final, other.final])
            else:
                new_final = self.final

            # Update internal arrays
            if self._initial is not None and other._initial is not None:
                new__initial = np.concatenate([self._initial, other._initial])
            else:
                new__initial = self._initial

            if self._final is not None and other._final is not None:
                new__final = np.concatenate([self._final, other._final])
            else:
                new__final = self._final

            # Update types
            if self.initial_type is not None and other.initial_type is not None:
                new_initial_type = np.concatenate([self.initial_type, other.initial_type])
            else:
                new_initial_type = self.initial_type

            if self.final_type is not None and other.final_type is not None:
                new_final_type = np.concatenate([self.final_type, other.final_type])
            else:
                new_final_type = self.final_type

            # Update true dimension
            if not augmented:
                new_true_dim = self._true_dim + getattr(other, "_true_dim", other.shape[0])
            else:
                new_true_dim = self._true_dim

            # Update all attributes in place
            self.shape = new_shape
            self.min = new_min
            self.max = new_max
            self.guess = new_guess
            self.initial = new_initial
            self.final = new_final
            self._initial = new__initial
            self._final = new__final
            self.initial_type = new_initial_type
            self.final_type = new_final_type
            self._true_dim = new_true_dim
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

        else:
            # Create a single new variable
            new_shape = (self.shape[0] + 1,)

            # Extend arrays
            if self.min is not None:
                self.min = np.concatenate([self.min, np.array([min])])
            if self.max is not None:
                self.max = np.concatenate([self.max, np.array([max])])
            if self.guess is not None:
                guess_arr = np.full((self.guess.shape[0], 1), guess)
                self.guess = np.concatenate([self.guess, guess_arr], axis=1)
            if self.initial is not None:
                self.initial = np.concatenate([self.initial, np.array([initial])])
            if self.final is not None:
                self.final = np.concatenate([self.final, np.array([final])])
            if self._initial is not None:
                self._initial = np.concatenate([self._initial, np.array([initial])])
            if self._final is not None:
                self._final = np.concatenate([self._final, np.array([final])])
            if self.initial_type is not None:
                self.initial_type = np.concatenate(
                    [self.initial_type, np.array(["Fix"], dtype=object)]
                )
            if self.final_type is not None:
                self.final_type = np.concatenate([self.final_type, np.array(["Fix"], dtype=object)])

            # Update dimensions
            self.shape = new_shape
            if not augmented:
                self._true_dim += 1
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    def __getitem__(self, idx):
        """Get a subset of the unified state variables."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("Step slicing not supported")

            new_shape = (stop - start,)
            new_name = f"{self.name}[{start}:{stop}]"

            # Slice all arrays
            new_min = self.min[idx] if self.min is not None else None
            new_max = self.max[idx] if self.max is not None else None
            new_guess = self.guess[:, idx] if self.guess is not None else None
            new_initial = self.initial[idx] if self.initial is not None else None
            new_final = self.final[idx] if self.final is not None else None
            new__initial = self._initial[idx] if self._initial is not None else None
            new__final = self._final[idx] if self._final is not None else None
            new_initial_type = self.initial_type[idx] if self.initial_type is not None else None
            new_final_type = self.final_type[idx] if self.final_type is not None else None

            # Calculate new true dimension
            new_true_dim = max(0, min(stop, self._true_dim) - max(start, 0))

            return UnifiedState(
                name=new_name,
                shape=new_shape,
                min=new_min,
                max=new_max,
                guess=new_guess,
                initial=new_initial,
                final=new_final,
                _initial=new__initial,
                _final=new__final,
                initial_type=new_initial_type,
                final_type=new_final_type,
                _true_dim=new_true_dim,
                _true_slice=slice(0, new_true_dim),
                _augmented_slice=slice(new_true_dim, new_shape[0]),
            )
        else:
            raise NotImplementedError("Only slice indexing is supported")

    def __repr__(self):
        """String representation of the UnifiedState object."""
        return f"UnifiedState('{self.name}', shape={self.shape})"


@dataclass
class UnifiedControl:
    """Unified control that behaves like a single Control but represents multiple Controls.

    This class provides a drop-in replacement for Control that holds aggregated
    data from multiple Control objects while preserving all the expected properties.

    Attributes:
        name (str): Name of the unified control
        shape (tuple): Combined shape of all controls
        min (np.ndarray): Minimum bounds for all control variables
        max (np.ndarray): Maximum bounds for all control variables
        guess (np.ndarray): Initial guess trajectory for all control variables
        _true_dim (int): True dimensionality (excluding augmented controls)
        _true_slice (slice): Slice for accessing true control variables
        _augmented_slice (slice): Slice for accessing augmented control variables
    """

    name: str
    shape: tuple
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    _true_dim: int = 0
    _true_slice: Optional[slice] = None
    _augmented_slice: Optional[slice] = None

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def true(self) -> "UnifiedControl":
        """Get the true control variables (excluding augmented controls)."""
        return self[self._true_slice]

    @property
    def augmented(self) -> "UnifiedControl":
        """Get the augmented control variables."""
        return self[self._augmented_slice]

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, augmented=False):
        """Append another control or create a new control variable.

        This method mimics the Control.append interface for compatibility.
        """
        if isinstance(other, (Control, UnifiedControl)):
            # Append another control object
            new_shape = (self.shape[0] + other.shape[0],)

            # Update bounds
            if self.min is not None and other.min is not None:
                new_min = np.concatenate([self.min, other.min])
            else:
                new_min = self.min

            if self.max is not None and other.max is not None:
                new_max = np.concatenate([self.max, other.max])
            else:
                new_max = self.max

            # Update guess
            if self.guess is not None and other.guess is not None:
                new_guess = np.concatenate([self.guess, other.guess], axis=1)
            else:
                new_guess = self.guess

            # Update true dimension
            if not augmented:
                new_true_dim = self._true_dim + getattr(other, "_true_dim", other.shape[0])
            else:
                new_true_dim = self._true_dim

            # Update all attributes in place
            self.shape = new_shape
            self.min = new_min
            self.max = new_max
            self.guess = new_guess
            self._true_dim = new_true_dim
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

        else:
            # Create a single new variable
            new_shape = (self.shape[0] + 1,)

            # Extend arrays
            if self.min is not None:
                self.min = np.concatenate([self.min, np.array([min])])
            if self.max is not None:
                self.max = np.concatenate([self.max, np.array([max])])
            if self.guess is not None:
                guess_arr = np.full((self.guess.shape[0], 1), guess)
                self.guess = np.concatenate([self.guess, guess_arr], axis=1)

            # Update dimensions
            self.shape = new_shape
            if not augmented:
                self._true_dim += 1
            self._true_slice = slice(0, self._true_dim)
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    def __getitem__(self, idx):
        """Get a subset of the unified control variables."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                raise NotImplementedError("Step slicing not supported")

            new_shape = (stop - start,)
            new_name = f"{self.name}[{start}:{stop}]"

            # Slice all arrays
            new_min = self.min[idx] if self.min is not None else None
            new_max = self.max[idx] if self.max is not None else None
            new_guess = self.guess[:, idx] if self.guess is not None else None

            # Calculate new true dimension
            new_true_dim = max(0, min(stop, self._true_dim) - max(start, 0))

            return UnifiedControl(
                name=new_name,
                shape=new_shape,
                min=new_min,
                max=new_max,
                guess=new_guess,
                _true_dim=new_true_dim,
                _true_slice=slice(0, new_true_dim),
                _augmented_slice=slice(new_true_dim, new_shape[0]),
            )
        else:
            raise NotImplementedError("Only slice indexing is supported")

    def __repr__(self):
        """String representation of the UnifiedControl object."""
        return f"UnifiedControl('{self.name}', shape={self.shape})"


def unify_states(states: List[State], name: str = "unified_state") -> UnifiedState:
    """Create a UnifiedState from a list of State objects.

    Args:
        states: List of State objects to unify
        name: Name for the unified state

    Returns:
        UnifiedState object containing aggregated data from all states
    """
    if not states:
        return UnifiedState(name=name, shape=(0,))

    # Sort states: true states (not starting with '_') first, then augmented states
    # (starting with '_')
    true_states = [state for state in states if not state.name.startswith("_")]
    augmented_states = [state for state in states if state.name.startswith("_")]
    sorted_states = true_states + augmented_states

    # Calculate total shape
    total_shape = sum(state.shape[0] for state in sorted_states)

    # Concatenate all arrays, handling None values properly
    min_arrays = []
    max_arrays = []
    guess_arrays = []
    initial_arrays = []
    final_arrays = []
    _initial_arrays = []
    _final_arrays = []
    initial_type_arrays = []
    final_type_arrays = []

    for state in sorted_states:
        if state.min is not None:
            min_arrays.append(state.min)
        else:
            # If min is None, fill with -inf for this state's dimensions
            min_arrays.append(np.full(state.shape[0], -np.inf))

        if state.max is not None:
            max_arrays.append(state.max)
        else:
            # If max is None, fill with +inf for this state's dimensions
            max_arrays.append(np.full(state.shape[0], np.inf))

        if state.guess is not None:
            guess_arrays.append(state.guess)
        if state.initial is not None:
            initial_arrays.append(state.initial)
        if state.final is not None:
            final_arrays.append(state.final)
        if state._initial is not None:
            _initial_arrays.append(state._initial)
        if state._final is not None:
            _final_arrays.append(state._final)
        if state.initial_type is not None:
            initial_type_arrays.append(state.initial_type)
        else:
            # If initial_type is None, fill with "Free" for this state's dimensions
            initial_type_arrays.append(np.full(state.shape[0], "Free", dtype=object))

        if state.final_type is not None:
            final_type_arrays.append(state.final_type)
        else:
            # If final_type is None, fill with "Free" for this state's dimensions
            final_type_arrays.append(np.full(state.shape[0], "Free", dtype=object))

    # Concatenate arrays if they exist
    unified_min = np.concatenate(min_arrays) if min_arrays else None
    unified_max = np.concatenate(max_arrays) if max_arrays else None
    unified_guess = np.concatenate(guess_arrays, axis=1) if guess_arrays else None
    unified_initial = np.concatenate(initial_arrays) if initial_arrays else None
    unified_final = np.concatenate(final_arrays) if final_arrays else None
    unified__initial = np.concatenate(_initial_arrays) if _initial_arrays else None
    unified__final = np.concatenate(_final_arrays) if _final_arrays else None
    unified_initial_type = np.concatenate(initial_type_arrays) if initial_type_arrays else None
    unified_final_type = np.concatenate(final_type_arrays) if final_type_arrays else None

    # Calculate true dimension (only from user-defined states, not augmented ones)
    # Since we simplified State/Control classes, all user states are "true" dimensions
    true_dim = sum(state.shape[0] for state in true_states)

    return UnifiedState(
        name=name,
        shape=(total_shape,),
        min=unified_min,
        max=unified_max,
        guess=unified_guess,
        initial=unified_initial,
        final=unified_final,
        _initial=unified__initial,
        _final=unified__final,
        initial_type=unified_initial_type,
        final_type=unified_final_type,
        _true_dim=true_dim,
        _true_slice=slice(0, true_dim),
        _augmented_slice=slice(true_dim, total_shape),
    )


def unify_controls(controls: List[Control], name: str = "unified_control") -> UnifiedControl:
    """Create a UnifiedControl from a list of Control objects.

    Args:
        controls: List of Control objects to unify
        name: Name for the unified control

    Returns:
        UnifiedControl object containing aggregated data from all controls
    """
    if not controls:
        return UnifiedControl(name=name, shape=(0,))

    # Sort controls: true controls (not starting with '_') first, then augmented controls
    # (starting with '_')
    true_controls = [control for control in controls if not control.name.startswith("_")]
    augmented_controls = [control for control in controls if control.name.startswith("_")]
    sorted_controls = true_controls + augmented_controls

    # Calculate total shape
    total_shape = sum(control.shape[0] for control in sorted_controls)

    # Concatenate all arrays, handling None values properly
    min_arrays = []
    max_arrays = []
    guess_arrays = []

    for control in sorted_controls:
        if control.min is not None:
            min_arrays.append(control.min)
        else:
            # If min is None, fill with -inf for this control's dimensions
            min_arrays.append(np.full(control.shape[0], -np.inf))

        if control.max is not None:
            max_arrays.append(control.max)
        else:
            # If max is None, fill with +inf for this control's dimensions
            max_arrays.append(np.full(control.shape[0], np.inf))

        if control.guess is not None:
            guess_arrays.append(control.guess)

    # Concatenate arrays if they exist
    unified_min = np.concatenate(min_arrays) if min_arrays else None
    unified_max = np.concatenate(max_arrays) if max_arrays else None
    unified_guess = np.concatenate(guess_arrays, axis=1) if guess_arrays else None

    # Calculate true dimension (only from user-defined controls, not augmented ones)
    # Since we simplified State/Control classes, all user controls are "true" dimensions
    true_dim = sum(control.shape[0] for control in true_controls)

    return UnifiedControl(
        name=name,
        shape=(total_shape,),
        min=unified_min,
        max=unified_max,
        guess=unified_guess,
        _true_dim=true_dim,
        _true_slice=slice(0, true_dim),
        _augmented_slice=slice(true_dim, total_shape),
    )


# Usage:

# # Convert from your symbolic layer (lists) to existing trajoptproblem interface
# unified_x = unify_states([state1, state2, state3], name="x")
# unified_u = unify_controls([control1, control2], name="u")

# # Now use unified_x and unified_u as drop-in replacements in TrajOptProblem
# problem = TrajOptProblem(dynamics, constraints, unified_x, unified_u, N, idx_time)

# The dataclasses preserve all expected properties and methods, so the existing optimization code
# in trajoptproblem.py should work without modification. The translation layer
# keeps the aggregation logic separate and functional as requested.
