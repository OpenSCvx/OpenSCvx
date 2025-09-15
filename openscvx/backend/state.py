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
        time.initial = np.array([Fix(0.0)])
        time.final = np.array([Minimize(5.0)])

        # Vector state
        position = State("position", (3,))
        position.min = np.array([0, 0, 10])
        position.max = np.array([10, 10, 200])
        position.initial = np.array([Fix(0), Free(1), Fix(50)])
        position.final = np.array([Fix(10), Free(5), Maximize(150)])
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
            arr (np.ndarray): Array of initial values or boundary condition objects
                (Fix, Free, Minimize, Maximize)

        Raises:
            ValueError: If the shape of arr doesn't match the state shape
        """
        if arr.shape != self.shape:
            raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
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
        if arr.shape != self.shape:
            raise ValueError(f"Shape mismatch: {arr.shape} != {self.shape}")
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




    def __repr__(self):
        """String representation of the State object.

        Returns:
            str: A string describing the State object
        """
        return f"State('{self.name}', shape={self.shape})"
