from enum import Enum

import numpy as np

from ..canonicalizer import canon_visitor
from .variable import Variable


class BoundaryType(str, Enum):
    """String enum for boundary condition types.

    This allows users to pass plain strings while we maintain type safety internally.
    """

    FIXED = "fixed"
    FREE = "free"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class State(Variable):
    """Symbolic state variable for trajectory optimization.

    A State represents a named state variable in the symbolic expression tree.
    It integrates with the AST system and supports boundary conditions for
    trajectory optimization.

    The State class is designed to be lightweight and focused on symbolic
    representation, with optimization-specific functionality handled by the
    unified layer.

    Boundary condition types:
    - Fixed values (Fix)
    - Free variables (Free)
    - Minimization objectives (Minimize)
    - Maximization objectives (Maximize)

    Attributes:
        name (str): Unique name identifier for this state variable
        shape (tuple): Shape of the state vector (e.g., (3,) for 3D position)
        min (np.ndarray): Minimum bounds for state variables
        max (np.ndarray): Maximum bounds for state variables
        guess (np.ndarray): Initial trajectory guess
        initial (np.ndarray): Initial boundary conditions
        final (np.ndarray): Final boundary conditions

    Example:
        ```python
        # Simple scalar state
        time = State("time", (1,))
        time.min = np.array([0.0])
        time.max = np.array([10.0])
        time.initial = [0.0]  # Defaults to fixed
        time.final = [("minimize", 5.0)]

        # Vector state
        position = State("position", (3,))
        position.min = np.array([0, 0, 10])
        position.max = np.array([10, 10, 200])
        position.initial = [0, ("free", 1), 50]  # Mix of fixed and free
        position.final = [10, ("free", 5), ("maximize", 150)]
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
        for field_name, data, types in [
            ("initial", self._initial, self.initial_type),
            ("final", self._final, self.final_type),
        ]:
            if data is None or types is None:
                continue
            for i, val in np.ndenumerate(data):
                if types[i] != "Fix":
                    continue
                min_i = self._min[i] if self._min is not None else -np.inf
                max_i = self._max[i] if self._max is not None else np.inf
                if val < min_i:
                    raise ValueError(
                        f"{field_name.capitalize()} Fixed value at index {i[0]} is lower then the "
                        f"min: {val} < {min_i}"
                    )
                if val > max_i:
                    raise ValueError(
                        f"{field_name.capitalize()} Fixed value at index {i[0]} is greater then "
                        f"the max: {val} > {max_i}"
                    )

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
            arr: Array of initial values. Can be:
                - Numbers (default to "fixed")
                - Tuples of (type, value) where type is "fixed", "free", "minimize", "maximize"

        Raises:
            ValueError: If the shape doesn't match the state shape
        """
        # Convert to list first to handle mixed types properly
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.shape != self.shape:
                raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
            arr = arr.tolist()

        # Ensure we have the right number of elements
        if len(arr) != self.shape[0]:
            raise ValueError(f"Length mismatch: got {len(arr)} elements, expected {self.shape[0]}")

        self._initial = np.zeros(self.shape, dtype=float)
        self.initial_type = np.full(self.shape, "Fix", dtype=object)

        for i, v in enumerate(arr):
            if isinstance(v, tuple) and len(v) == 2:
                # Tuple API: (type, value)
                bc_type_str, bc_value = v
                try:
                    bc_type = BoundaryType(bc_type_str)  # Validates the string
                except ValueError:
                    valid_types = [t.value for t in BoundaryType]
                    raise ValueError(
                        f"Invalid boundary condition type: {bc_type_str}. "
                        f"Valid types are: {valid_types}"
                    )
                self._initial[i] = float(bc_value)
                self.initial_type[i] = bc_type.value.capitalize()
            elif isinstance(v, (int, float, np.number)):
                # Simple number defaults to fixed
                self._initial[i] = float(v)
                self.initial_type[i] = "Fix"
            else:
                raise ValueError(
                    f"Invalid boundary condition format: {v}. "
                    f"Use a number (defaults to fixed) or tuple ('type', value) "
                    f"where type is 'fixed', 'free', 'minimize', or 'maximize'."
                )

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
            arr: Array of final values. Can be:
                - Numbers (default to "fixed")
                - Tuples of (type, value) where type is "fixed", "free", "minimize", "maximize"

        Raises:
            ValueError: If the shape doesn't match the state shape
        """
        # Convert to list first to handle mixed types properly
        if not isinstance(arr, (list, tuple)):
            arr = np.asarray(arr)
            if arr.shape != self.shape:
                raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
            arr = arr.tolist()

        # Ensure we have the right number of elements
        if len(arr) != self.shape[0]:
            raise ValueError(f"Length mismatch: got {len(arr)} elements, expected {self.shape[0]}")

        self._final = np.zeros(self.shape, dtype=float)
        self.final_type = np.full(self.shape, "Fix", dtype=object)

        for i, v in enumerate(arr):
            if isinstance(v, tuple) and len(v) == 2:
                # Tuple API: (type, value)
                bc_type_str, bc_value = v
                try:
                    bc_type = BoundaryType(bc_type_str)  # Validates the string
                except ValueError:
                    valid_types = [t.value for t in BoundaryType]
                    raise ValueError(
                        f"Invalid boundary condition type: {bc_type_str}. "
                        f"Valid types are: {valid_types}"
                    )
                self._final[i] = float(bc_value)
                self.final_type[i] = bc_type.value.capitalize()
            elif isinstance(v, (int, float, np.number)):
                # Simple number defaults to fixed
                self._final[i] = float(v)
                self.final_type[i] = "Fix"
            else:
                raise ValueError(
                    f"Invalid boundary condition format: {v}. "
                    f"Use a number (defaults to fixed) or tuple ('type', value) "
                    f"where type is 'fixed', 'free', 'minimize', or 'maximize'."
                )

        self._check_bounds_against_initial_final()

    def __repr__(self):
        """String representation of the State object.

        Returns:
            str: A string describing the State object
        """
        return f"State('{self.name}', shape={self.shape})"


@canon_visitor(State)
def canon_state(node: State):
    # State nodes are already canonical
    return node
