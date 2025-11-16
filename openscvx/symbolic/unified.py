"""Unified state and control representation for trajectory optimization.

This module provides the unification layer that aggregates multiple symbolic State
and Control objects into unified representations. The unification process enables:

- Multiple separate State/Control objects in symbolic expressions
- Single unified state/control vectors for numerical optimization
- Automatic slice assignment for extracting subvectors
- Compatibility with existing optimization infrastructure
- Separation of user-defined and augmented (internal) variables

Architecture:
    The unification layer bridges the symbolic expression system and numerical
    optimization by:

    1. **Collection**: Gathers all State and Control objects from expression trees
    2. **Sorting**: Organizes variables (user-defined first, then augmented)
    3. **Aggregation**: Concatenates bounds, guesses, and boundary conditions
    4. **Slice Assignment**: Assigns each State/Control a slice for indexing
    5. **Unified Representation**: Creates UnifiedState/UnifiedControl objects

    This separation allows users to define problems with natural variable names
    while maintaining efficient vectorized operations during optimization.

Unified Variables:
    UnifiedState and UnifiedControl are dataclasses that act as drop-in replacements
    for individual State and Control objects, aggregating:

    - Shapes and bounds (min/max)
    - Initial guesses and boundary conditions
    - Slices for extracting individual variables from unified vectors
    - Distinction between true (user-defined) and augmented (internal) variables

Example:
    Creating and unifying multiple states::

        import openscvx as ox

        # Define separate symbolic states
        position = ox.State("position", shape=(3,), min=-10, max=10)
        velocity = ox.State("velocity", shape=(3,), min=-5, max=5)
        mass = ox.State("mass", shape=(1,), min=0.1, max=10.0)

        # Unify into single state vector
        from openscvx.symbolic.unified import unify_states
        unified_x = unify_states([position, velocity, mass], name="x")

        # Access unified properties
        print(unified_x.shape)    # (7,) - combined shape
        print(unified_x.min)      # Combined bounds: [-10, -10, -10, -5, -5, -5, 0.1]
        print(unified_x.true)     # Access only user-defined states

    Accessing slices after unification::

        # After unification, each State has a slice assigned
        print(position._slice)    # slice(0, 3)
        print(velocity._slice)    # slice(3, 6)
        print(mass._slice)        # slice(6, 7)

        # During lowering, these slices extract values from unified vector
        x_unified = jnp.array([1, 2, 3, 4, 5, 6, 7])
        position_val = x_unified[position._slice]  # [1, 2, 3]

See Also:
    - unify_states(): Main function for creating UnifiedState
    - unify_controls(): Main function for creating UnifiedControl
    - State: Individual symbolic state variable (symbolic/expr/state.py)
    - Control: Individual symbolic control variable (symbolic/expr/control.py)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State


@dataclass
class UnifiedState:
    """Unified state vector aggregating multiple State objects.

    UnifiedState is a drop-in replacement for individual State objects that holds
    aggregated data from multiple State instances. It maintains compatibility with
    optimization infrastructure while providing access to individual state components
    through slicing.

    The unified state separates user-defined "true" states from augmented states
    added internally (e.g., for CTCS constraints or time variables). This separation
    allows clean access to physical states while supporting advanced features.

    Attributes:
        name (str): Name identifier for the unified state vector
        shape (tuple): Combined shape (total_dim,) of all aggregated states
        min (np.ndarray): Lower bounds for all state variables, shape (total_dim,)
        max (np.ndarray): Upper bounds for all state variables, shape (total_dim,)
        guess (np.ndarray): Initial guess trajectory, shape (num_nodes, total_dim)
        initial (np.ndarray): Initial boundary conditions, shape (total_dim,)
        final (np.ndarray): Final boundary conditions, shape (total_dim,)
        _initial (np.ndarray): Internal initial values, shape (total_dim,)
        _final (np.ndarray): Internal final values, shape (total_dim,)
        initial_type (np.ndarray): Boundary condition types at t0 ("Fix" or "Free"),
            shape (total_dim,), dtype=object
        final_type (np.ndarray): Boundary condition types at tf ("Fix" or "Free"),
            shape (total_dim,), dtype=object
        _true_dim (int): Number of user-defined state dimensions (excludes augmented)
        _true_slice (slice): Slice for extracting true states from unified vector
        _augmented_slice (slice): Slice for extracting augmented states
        time_slice (Optional[slice]): Slice for time state variable, if present
        ctcs_slice (Optional[slice]): Slice for CTCS augmented states, if present

    Properties:
        true: Returns UnifiedState view containing only true (user-defined) states
        augmented: Returns UnifiedState view containing only augmented states

    Example:
        Creating a unified state from multiple State objects::

            position = ox.State("pos", shape=(3,), min=-10, max=10)
            velocity = ox.State("vel", shape=(3,), min=-5, max=5)

            unified = unify_states([position, velocity], name="x")
            print(unified.shape)        # (6,)
            print(unified.min)          # [-10, -10, -10, -5, -5, -5]
            print(unified.true.shape)   # (6,) - all are true states
            print(unified.augmented.shape)  # (0,) - no augmented states

        Appending states dynamically::

            unified = UnifiedState(name="x", shape=(0,), _true_dim=0)
            unified.append(min=-1, max=1, guess=0.5)  # Add scalar state
            print(unified.shape)  # (1,)

    See Also:
        - unify_states(): Factory function for creating UnifiedState from State list
        - State: Individual symbolic state variable
        - UnifiedControl: Analogous unified control vector
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
    time_slice: Optional[slice] = None  # Slice for time state
    ctcs_slice: Optional[slice] = None  # Slice for CTCS augmented states

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def true(self) -> "UnifiedState":
        """Get the true (user-defined) state variables.

        Returns a view of the unified state containing only user-defined states,
        excluding internal augmented states added for CTCS, time, etc.

        Returns:
            UnifiedState: Sliced view containing only true state variables

        Example:
            Get true user-defined state:

                unified = unify_states([position, velocity, ctcs_aug], name="x")
                true_states = unified.true  # Only position and velocity
                true_states.shape  # (6,) if position and velocity are 3D each
        """
        return self[self._true_slice]

    @property
    def augmented(self) -> "UnifiedState":
        """Get the augmented (internal) state variables.

        Returns a view of the unified state containing only augmented states
        added internally by the optimization framework (e.g., CTCS penalty states,
        time variables).

        Returns:
            UnifiedState: Sliced view containing only augmented state variables

        Example:
            Get augmented state:

                unified = unify_states([position, ctcs_aug], name="x")
                aug_states = unified.augmented  # Only CTCS states
        """
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

        This method allows dynamic extension of the unified state, either by appending
        another State/UnifiedState object or by creating a new scalar state variable
        with specified properties. Modifies the unified state in-place.

        Args:
            other (Optional[State | UnifiedState]): State object to append. If None,
                creates a new scalar state variable with properties from keyword args.
            min (float): Lower bound for new scalar state (default: -inf)
            max (float): Upper bound for new scalar state (default: inf)
            guess (float): Initial guess value for new scalar state (default: 0.0)
            initial (float): Initial boundary condition for new scalar state (default: 0.0)
            final (float): Final boundary condition for new scalar state (default: 0.0)
            augmented (bool): Whether the appended state is augmented (internal) rather
                than true (user-defined). Affects _true_dim tracking. Default: False

        Returns:
            None: Modifies the unified state in-place

        Example:
            Appending a State object::

                unified = unify_states([position], name="x")
                velocity = ox.State("vel", shape=(3,), min=-5, max=5)
                unified.append(velocity)
                print(unified.shape)  # (6,) - position (3) + velocity (3)

            Creating new scalar state variables::

                unified = UnifiedState(name="x", shape=(0,), _true_dim=0)
                unified.append(min=-1, max=1, guess=0.5)  # Add scalar state
                unified.append(min=-2, max=2, augmented=True)  # Add augmented state
                print(unified.shape)  # (2,)
                print(unified._true_dim)  # 1 (only first is true)

        Note:
            Maintains the invariant that true states appear before augmented states
            in the unified vector. When appending augmented states, they are added
            to the end but don't increment _true_dim.
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
        """Get a subset of the unified state variables.

        Enables slicing of the unified state to extract subsets of state variables.
        Returns a new UnifiedState containing only the sliced dimensions.

        Args:
            idx (slice): Slice object specifying which state dimensions to extract.
                Only simple slices with step=1 are supported.

        Returns:
            UnifiedState: New unified state containing only the sliced dimensions

        Raises:
            NotImplementedError: If idx is not a slice, or if step != 1

        Example:
            Generate unified state object

                unified = unify_states([position, velocity], name="x")

            position has shape (3,), velocity has shape (3,)

                first_three = unified[0:3]  # Extract position only
                print(first_three.shape)  # (3,)
                last_three = unified[3:6]  # Extract velocity only
                print(last_three.shape)  # (3,)

        Note:
            The sliced state maintains all properties (bounds, guesses, etc.) for
            the selected dimensions. The _true_dim is recalculated based on which
            dimensions fall within the original true state range.
        """
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
    """Unified control vector aggregating multiple Control objects.

    UnifiedControl is a drop-in replacement for individual Control objects that holds
    aggregated data from multiple Control instances. It maintains compatibility with
    optimization infrastructure while providing access to individual control components
    through slicing.

    The unified control separates user-defined "true" controls from augmented controls
    added internally (e.g., for time dilation). This separation allows clean access to
    physical control inputs while supporting advanced features.

    Attributes:
        name (str): Name identifier for the unified control vector
        shape (tuple): Combined shape (total_dim,) of all aggregated controls
        min (np.ndarray): Lower bounds for all control variables, shape (total_dim,)
        max (np.ndarray): Upper bounds for all control variables, shape (total_dim,)
        guess (np.ndarray): Initial guess trajectory, shape (num_nodes, total_dim)
        _true_dim (int): Number of user-defined control dimensions (excludes augmented)
        _true_slice (slice): Slice for extracting true controls from unified vector
        _augmented_slice (slice): Slice for extracting augmented controls
        time_dilation_slice (Optional[slice]): Slice for time dilation control, if present

    Properties:
        true: Returns UnifiedControl view containing only true (user-defined) controls
        augmented: Returns UnifiedControl view containing only augmented controls

    Example:
        Creating a unified control from multiple Control objects::

            thrust = ox.Control("thrust", shape=(3,), min=0, max=10)
            torque = ox.Control("torque", shape=(3,), min=-1, max=1)

            unified = unify_controls([thrust, torque], name="u")
            print(unified.shape)        # (6,)
            print(unified.min)          # [0, 0, 0, -1, -1, -1]
            print(unified.true.shape)   # (6,) - all are true controls
            print(unified.augmented.shape)  # (0,) - no augmented controls

        Appending controls dynamically::

            unified = UnifiedControl(name="u", shape=(0,), _true_dim=0)
            unified.append(min=-1, max=1, guess=0.0)  # Add scalar control
            print(unified.shape)  # (1,)

    See Also:
        - unify_controls(): Factory function for creating UnifiedControl from Control list
        - Control: Individual symbolic control variable
        - UnifiedState: Analogous unified state vector
    """

    name: str
    shape: tuple
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    guess: Optional[np.ndarray] = None
    _true_dim: int = 0
    _true_slice: Optional[slice] = None
    _augmented_slice: Optional[slice] = None
    time_dilation_slice: Optional[slice] = None  # Slice for time dilation control

    def __post_init__(self):
        """Initialize slices after dataclass creation."""
        if self._true_slice is None:
            self._true_slice = slice(0, self._true_dim)
        if self._augmented_slice is None:
            self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def true(self) -> "UnifiedControl":
        """Get the true (user-defined) control variables.

        Returns a view of the unified control containing only user-defined controls,
        excluding internal augmented controls added for time dilation, etc.

        Returns:
            UnifiedControl: Sliced view containing only true control variables

        Example:
            Get true user defined controls:

                unified = unify_controls([thrust, torque, time_dilation], name="u")
                true_controls = unified.true  # Only thrust and torque
        """
        return self[self._true_slice]

    @property
    def augmented(self) -> "UnifiedControl":
        """Get the augmented (internal) control variables.

        Returns a view of the unified control containing only augmented controls
        added internally by the optimization framework (e.g., time dilation control).

        Returns:
            UnifiedControl: Sliced view containing only augmented control variables

        Example:
            Get augmented controls:

                unified = unify_controls([thrust, time_dilation], name="u")
                aug_controls = unified.augmented  # Only time dilation
        """
        return self[self._augmented_slice]

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, augmented=False):
        """Append another control or create a new control variable.

        This method allows dynamic extension of the unified control, either by appending
        another Control/UnifiedControl object or by creating a new scalar control variable
        with specified properties. Modifies the unified control in-place.

        Args:
            other (Optional[Control | UnifiedControl]): Control object to append. If None,
                creates a new scalar control variable with properties from keyword args.
            min (float): Lower bound for new scalar control (default: -inf)
            max (float): Upper bound for new scalar control (default: inf)
            guess (float): Initial guess value for new scalar control (default: 0.0)
            augmented (bool): Whether the appended control is augmented (internal) rather
                than true (user-defined). Affects _true_dim tracking. Default: False

        Returns:
            None: Modifies the unified control in-place

        Example:
            Appending a Control object::

                unified = unify_controls([thrust], name="u")
                torque = ox.Control("torque", shape=(3,), min=-1, max=1)
                unified.append(torque)
                print(unified.shape)  # (6,) - thrust (3) + torque (3)

            Creating new scalar control variables::

                unified = UnifiedControl(name="u", shape=(0,), _true_dim=0)
                unified.append(min=-1, max=1, guess=0.0)  # Add scalar control
                print(unified.shape)  # (1,)
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
        """Get a subset of the unified control variables.

        Enables slicing of the unified control to extract subsets of control variables.
        Returns a new UnifiedControl containing only the sliced dimensions.

        Args:
            idx (slice): Slice object specifying which control dimensions to extract.
                Only simple slices with step=1 are supported.

        Returns:
            UnifiedControl: New unified control containing only the sliced dimensions

        Raises:
            NotImplementedError: If idx is not a slice, or if step != 1

        Example:
            Generate unified control object:

                unified = unify_controls([thrust, torque], name="u")

            thrust has shape (3,), torque has shape (3,)

                first_three = unified[0:3]  # Extract thrust only
                print(first_three.shape)  # (3,)

        Note:
            The sliced control maintains all properties (bounds, guesses, etc.) for
            the selected dimensions. The _true_dim is recalculated based on which
            dimensions fall within the original true control range.
        """
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

    This function is the primary way to aggregate multiple symbolic State objects into
    a single unified state vector for numerical optimization. It:

    1. Sorts states (user-defined first, augmented states second)
    2. Concatenates all state properties (bounds, guesses, boundary conditions)
    3. Assigns slices to each State for extracting values from unified vector
    4. Identifies special states (time, CTCS augmented states)
    5. Returns a UnifiedState with all aggregated data

    Args:
        states (List[State]): List of State objects to unify. Can include both
            user-defined states and augmented states (names starting with '_').
        name (str): Name identifier for the unified state vector (default: "unified_state")

    Returns:
        UnifiedState: Unified state object containing:
            - Aggregated bounds, guesses, and boundary conditions
            - Shape equal to sum of all state shapes
            - Slices for extracting individual state components
            - Properties for accessing true vs augmented states

    Example:
        Basic unification::

            import openscvx as ox
            from openscvx.symbolic.unified import unify_states

            position = ox.State("pos", shape=(3,), min=-10, max=10)
            velocity = ox.State("vel", shape=(3,), min=-5, max=5)

            unified = unify_states([position, velocity], name="x")
            print(unified.shape)       # (6,)
            print(unified._true_dim)   # 6 (all are user states)
            print(position._slice)     # slice(0, 3) - assigned during unification
            print(velocity._slice)     # slice(3, 6)

        With augmented states::

            # CTCS or other features may add augmented states
            time_state = ox.State("time", shape=(1,))
            ctcs_aug = ox.State("_ctcs_aug_0", shape=(2,))  # Augmented state

            unified = unify_states([position, velocity, time_state, ctcs_aug])
            print(unified._true_dim)         # 7 (pos + vel + time)
            print(unified.true.shape)        # (7,)
            print(unified.augmented.shape)   # (2,) - only CTCS augmented

    Note:
        After unification, each State object has its `_slice` attribute set,
        which is used during JAX lowering to extract the correct values from
        the unified state vector.

    See Also:
        - UnifiedState: Return type with detailed documentation
        - unify_controls(): Analogous function for Control objects
        - State: Individual symbolic state variable
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

    # Find time state slice
    time_state = next((s for s in sorted_states if s.name == "time"), None)
    time_slice = time_state._slice if time_state else None

    # Find CTCS augmented states slice
    ctcs_states = [s for s in sorted_states if s.name.startswith("_ctcs_aug_")]
    ctcs_slice = (
        slice(ctcs_states[0]._slice.start, ctcs_states[-1]._slice.stop) if ctcs_states else None
    )

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
        time_slice=time_slice,
        ctcs_slice=ctcs_slice,
    )


def unify_controls(controls: List[Control], name: str = "unified_control") -> UnifiedControl:
    """Create a UnifiedControl from a list of Control objects.

    This function is the primary way to aggregate multiple symbolic Control objects into
    a single unified control vector for numerical optimization. It:

    1. Sorts controls (user-defined first, augmented controls second)
    2. Concatenates all control properties (bounds, guesses)
    3. Assigns slices to each Control for extracting values from unified vector
    4. Identifies special controls (time dilation)
    5. Returns a UnifiedControl with all aggregated data

    Args:
        controls (List[Control]): List of Control objects to unify. Can include both
            user-defined controls and augmented controls (names starting with '_').
        name (str): Name identifier for the unified control vector (default: "unified_control")

    Returns:
        UnifiedControl: Unified control object containing:
            - Aggregated bounds and guesses
            - Shape equal to sum of all control shapes
            - Slices for extracting individual control components
            - Properties for accessing true vs augmented controls

    Example:
        Basic unification::

            import openscvx as ox
            from openscvx.symbolic.unified import unify_controls

            thrust = ox.Control("thrust", shape=(3,), min=0, max=10)
            torque = ox.Control("torque", shape=(3,), min=-1, max=1)

            unified = unify_controls([thrust, torque], name="u")
            print(unified.shape)       # (6,)
            print(unified._true_dim)   # 6 (all are user controls)
            print(thrust._slice)       # slice(0, 3) - assigned during unification
            print(torque._slice)       # slice(3, 6)

        With augmented controls::

            # Time-optimal problems may add time dilation control
            time_dilation = ox.Control("_time_dilation", shape=(1,))

            unified = unify_controls([thrust, torque, time_dilation])
            print(unified._true_dim)         # 6 (thrust + torque)
            print(unified.true.shape)        # (6,)
            print(unified.augmented.shape)   # (1,) - time dilation

    Note:
        After unification, each Control object has its `_slice` attribute set,
        which is used during JAX lowering to extract the correct values from
        the unified control vector.

    See Also:
        - UnifiedControl: Return type with detailed documentation
        - unify_states(): Analogous function for State objects
        - Control: Individual symbolic control variable
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

    # Find time dilation control slice
    time_dilation_control = next((c for c in sorted_controls if c.name == "_time_dilation"), None)
    time_dilation_slice = time_dilation_control._slice if time_dilation_control else None

    return UnifiedControl(
        name=name,
        shape=(total_shape,),
        min=unified_min,
        max=unified_max,
        guess=unified_guess,
        _true_dim=true_dim,
        _true_slice=slice(0, true_dim),
        _augmented_slice=slice(true_dim, total_shape),
        time_dilation_slice=time_dilation_slice,
    )
