import numpy as np

from openscvx.backend.variable import Variable


class Fix:
    """Class representing a fixed state variable in the optimization problem.
    
    A fixed state variable is one that is constrained to a specific value
    and cannot be optimized.
    
    Attributes:
        value: The fixed value that the state variable must take.
    """
    def __init__(self, value):
        """Initialize a new fixed state variable.
        
        Args:
            value: The fixed value that the state variable must take.
        """
        self.value = value

    def __repr__(self):
        """Get a string representation of this fixed state variable.
        
        Returns:
            str: A string representation showing the fixed value.
        """
        return f"Fix({self.value})"


class Free:
    """Class representing a free state variable in the optimization problem.
    
    A free state variable is one that is not constrained to any specific value
    but can be optimized within its bounds.
    
    Attributes:
        guess: The initial guess value for optimization.
    """
    def __init__(self, guess):
        """Initialize a new free state variable.
        
        Args:
            guess: The initial guess value for optimization.
        """
        self.guess = guess

    def __repr__(self):
        """Get a string representation of this free state variable.
        
        Returns:
            str: A string representation showing the guess value.
        """
        return f"Free({self.guess})"


class Minimize:
    """Class representing a state variable to be minimized in the optimization problem.
    
    A minimized state variable is one that is optimized to achieve the lowest
    possible value within its bounds.
    
    Attributes:
        guess: The initial guess value for optimization.
    """
    def __init__(self, guess):
        """Initialize a new minimized state variable.
        
        Args:
            guess: The initial guess value for optimization.
        """
        self.guess = guess

    def __repr__(self):
        """Get a string representation of this minimized state variable.
        
        Returns:
            str: A string representation showing the guess value.
        """
        return f"Minimize({self.guess})"


class Maximize:
    """Class representing a state variable to be maximized in the optimization problem.
    
    A maximized state variable is one that is optimized to achieve the highest
    possible value within its bounds.
    
    Attributes:
        guess: The initial guess value for optimization.
    """
    def __init__(self, guess):
        """Initialize a new maximized state variable.
        
        Args:
            guess: The initial guess value for optimization.
        """
        self.guess = guess

    def __repr__(self):
        """Get a string representation of this maximized state variable.
        
        Returns:
            str: A string representation showing the guess value.
        """
        return f"Maximize({self.guess})"

class State(Variable):
    """A class representing the state variables in an optimal control problem.

    The State class extends Variable to handle state-specific properties like initial and final conditions,
    as well as true and augmented state dimensions. It supports various boundary condition types:
    - Fixed values (Fix)
    - Free variables (Free)
    - Minimization objectives (Minimize)
    - Maximization objectives (Maximize)

    Attributes:
        name (str): Name of the state variable.
        shape (tuple): Shape of the state variable array.
        min (np.ndarray): Minimum bounds for the state variables. Shape: (n_states,).
        max (np.ndarray): Maximum bounds for the state variables. Shape: (n_states,).
        guess (np.ndarray): Used to initialize SCP and contains the current SCP solution for the state trajectory. Shape: (n_nodes, n_states).
        initial (np.ndarray): Initial state values or boundary condition objects (Free, Fixed, Minimize, Maximize). Shape: (n_states,).
        final (np.ndarray): Final state values or boundary condition objects (Free, Fixed, Minimize, Maximize). Shape: (n_states,).
        _initial (np.ndarray): Internal storage for initial state values.
        _final (np.ndarray): Internal storage for final state values.
        initial_type (str): Type of initial boundary condition ('fix', 'free', 'minimize', 'maximize').
        final_type (str): Type of final boundary condition ('fix', 'free', 'minimize', 'maximize').
        _true_dim (int): True dimensionality of the state variables.
        _true_slice (slice): Slice for accessing true state variables.
        _augmented_slice (slice): Slice for accessing augmented state variables.

    Notes:
        Attributes prefixed with underscore (_) are for internal use only and should not be accessed directly.

    Example:
    ```python
    state = State("position", (3,))
    state.min = np.array([0, 0, 10])
    state.max = np.array([10, 10, 200])
    state.guess = np.linspace([0, 1, 2], [10, 5, 8], 3)
    state.initial = np.array([Fix(0), Free(1), 2])
    state.final = np.array([Fix(10), Free(5), Maximize(8)])
    ```
    """

    def __init__(self, name, shape):
        """Initialize a State object.

        Args:
            name (str): Name identifier for the state variable
            shape (tuple): Shape of the state vector
        """
        super().__init__(name, shape)
        self._initial = None
        self.initial_type = None
        self._final = None
        self.final_type = None

        self._true_dim = shape[0]
        self._update_slices()

    def _update_slices(self):
        """Update the slice objects for true and augmented states."""
        self._true_slice = slice(0, self._true_dim)
        self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def min(self):
        """Get the minimum bounds for the state variables.

        Returns:
            np.ndarray: Array of minimum values for each state variable
        """
        return self._min

    @min.setter
    def min(self, val):
        """Set the minimum bounds for the state variables.

        Args:
            val (np.ndarray): Array of minimum values for each state variable

        Raises:
            ValueError: If the shape of val doesn't match the state shape
        """
        val = np.asarray(val)
        if val.shape != self.shape:
            raise ValueError(f"Min shape {val.shape} does not match State shape {self.shape}")
        self._min = val
        self._check_bounds_against_initial_final()

    @property
    def max(self):
        """Get the maximum bounds for the state variables.

        Returns:
            np.ndarray: Array of maximum values for each state variable
        """
        return self._max

    @max.setter
    def max(self, val):
        """Set the maximum bounds for the state variables.

        Args:
            val (np.ndarray): Array of maximum values for each state variable

        Raises:
            ValueError: If the shape of val doesn't match the state shape
        """
        val = np.asarray(val)
        if val.shape != self.shape:
            raise ValueError(f"Max shape {val.shape} does not match State shape {self.shape}")
        self._max = val
        self._check_bounds_against_initial_final()

    def _check_bounds_against_initial_final(self):
        """Check if initial and final values respect the bounds.

        Raises:
            ValueError: If any fixed initial or final value violates the bounds
        """
        for field_name, data, types in [('initial', self._initial, self.initial_type),
                                        ('final', self._final, self.final_type)]:
            if data is None or types is None:
                continue
            for i, val in np.ndenumerate(data):
                if types[i] != "Fix":
                    continue
                min_i = self._min[i] if self._min is not None else -np.inf
                max_i = self._max[i] if self._max is not None else np.inf
                if val < min_i:
                    raise ValueError(f"{field_name.capitalize()} Fixed value at index {i[0]} is lower then the min: {val} < {min_i}")
                if val > max_i:
                    raise ValueError(f"{field_name.capitalize()} Fixed value at index {i[0]} is greater then the max: {val} > {max_i}")

    @property
    def initial(self):
        """Get the initial state values.

        Returns:
            np.ndarray: Array of initial state values
        """
        return self._initial

    @initial.setter
    def initial(self, arr):
        """Set the initial state values and their types.

        Args:
            arr (np.ndarray): Array of initial values or boundary condition objects
                (Fix, Free, Minimize, Maximize)

        Raises:
            ValueError: If the shape of arr doesn't match the state shape
        """
        arr = np.asarray(arr, dtype=object)
        if arr.shape != self.shape:
            raise ValueError(f"Initial value shape {arr.shape} does not match State shape {self.shape}")
        self._initial = np.zeros(arr.shape)
        self.initial_type = np.full(arr.shape, "Fix", dtype=object)

        for i, v in np.ndenumerate(arr):
            if isinstance(v, Free):
                self._initial[i] = v.guess
                self.initial_type[i] = "Free"
            elif isinstance(v, Minimize):
                self._initial[i] = v.guess
                self.initial_type[i] = "Minimize"
            elif isinstance(v, Maximize):
                self._initial[i] = v.guess
                self.initial_type[i] = "Maximize"
            elif isinstance(v, Fix):
                val = v.value
                self._initial[i] = val
                self.initial_type[i] = "Fix"
            else:
                val = v
                self._initial[i] = val
                self.initial_type[i] = "Fix"

        self._check_bounds_against_initial_final()

    @property
    def final(self):
        """Get the final state values.

        Returns:
            np.ndarray: Array of final state values
        """
        return self._final

    @final.setter
    def final(self, arr):
        """Set the final state values and their types.

        Args:
            arr (np.ndarray): Array of final values or boundary condition objects
                (Fix, Free, Minimize, Maximize)

        Raises:
            ValueError: If the shape of arr doesn't match the state shape
        """
        arr = np.asarray(arr, dtype=object)
        if arr.shape != self.shape:
            raise ValueError(f"Final value shape {arr.shape} does not match State shape {self.shape}")
        self._final = np.zeros(arr.shape)
        self.final_type = np.full(arr.shape, "Fix", dtype=object)

        for i, v in np.ndenumerate(arr):
            if isinstance(v, Free):
                self._final[i] = v.guess
                self.final_type[i] = "Free"
            elif isinstance(v, Minimize):
                self._final[i] = v.guess
                self.final_type[i] = "Minimize"
            elif isinstance(v, Maximize):
                self._final[i] = v.guess
                self.final_type[i] = "Maximize"
            elif isinstance(v, Fix):
                val = v.value
                self._final[i] = val
                self.final_type[i] = "Fix"
            else:
                val = v
                self._final[i] = val
                self.final_type[i] = "Fix"

        self._check_bounds_against_initial_final()

    @property
    def true(self):
        """Get the true state variables (excluding augmented states).

        Returns:
            State: A new State object containing only the true state variables
        """
        return self[self._true_slice]

    @property
    def augmented(self):
        """Get the augmented state variables.

        Returns:
            State: A new State object containing only the augmented state variables
        """
        return self[self._augmented_slice]

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, initial=0.0, final=0.0, augmented=False):
        """Append another state or create a new state variable.

        Args:
            other (State, optional): Another State object to append
            min (float, optional): Minimum bound for new state. Defaults to -np.inf
            max (float, optional): Maximum bound for new state. Defaults to np.inf
            guess (float, optional): Initial guess for new state. Defaults to 0.0
            initial (float, optional): Initial value for new state. Defaults to 0.0
            final (float, optional): Final value for new state. Defaults to 0.0
            augmented (bool, optional): Whether the new state is augmented. Defaults to False
        """
        if isinstance(other, State):
            super().append(other=other)

            if self._initial is None:
                self._initial = np.array(other._initial) if other._initial is not None else None
            elif other._initial is not None:
                self._initial = np.concatenate([self._initial, other._initial], axis=0)

            if self._final is None:
                self._final = np.array(other._final) if other._final is not None else None
            elif other._final is not None:
                self._final = np.concatenate([self._final, other._final], axis=0)

            if self.initial_type is None:
                self.initial_type = np.array(other.initial_type) if other.initial_type is not None else None
            elif other.initial_type is not None:
                self.initial_type = np.concatenate([self.initial_type, other.initial_type], axis=0)

            if self.final_type is None:
                self.final_type = np.array(other.final_type) if other.final_type is not None else None
            elif other.final_type is not None:
                self.final_type = np.concatenate([self.final_type, other.final_type], axis=0)

            if not augmented:
                self._true_dim += getattr(other, "_true_dim", other.shape[0])
            self._update_slices()
        else:
            temp_state = State(name=f"{self.name}_temp_append", shape=(1,))
            temp_state.min = min
            temp_state.max = max
            temp_state.guess = guess
            temp_state.initial = initial
            temp_state.final = final
            self.append(temp_state, augmented=augmented)

    def __getitem__(self, idx):
        """Get a subset of the state variables.

        Args:
            idx: Index or slice to select state variables

        Returns:
            State: A new State object containing the selected variables
        """
        new_state = super().__getitem__(idx)
        new_state.__class__ = State

        def slice_attr(attr):
            if attr is None:
                return None
            if attr.ndim == 2 and attr.shape[1] == self.shape[0]:
                return attr[:, idx]
            return attr[idx]

        new_state._initial = slice_attr(self._initial)
        new_state.initial_type = slice_attr(self.initial_type)
        new_state._final = slice_attr(self._final)
        new_state.final_type = slice_attr(self.final_type)

        if isinstance(idx, slice):
            selected = np.arange(self.shape[0])[idx]
        elif isinstance(idx, (list, np.ndarray)):
            selected = np.array(idx)
        else:
            selected = np.array([idx])

        new_state._true_dim = np.sum(selected < self._true_dim)
        new_state._update_slices()

        return new_state

    def __repr__(self):
        """String representation of the State object.

        Returns:
            str: A string describing the State object
        """
        return f"State('{self.name}', shape={self.shape})"
