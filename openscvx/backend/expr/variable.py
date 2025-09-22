import numpy as np

from .expr import Leaf


class Variable(Leaf):
    """A base class for variables in an optimal control problem.

    The Variable class provides the fundamental structure for state and control variables,
    handling their shapes, bounds, and initial guesses. It supports operations like
    appending new variables and slicing.

    Attributes:
        name (str): Name identifier for the variable
        _shape (tuple): Shape of the variable vector
        _min (np.ndarray): Minimum bounds for the variable
        _max (np.ndarray): Maximum bounds for the variable
        _guess (np.ndarray): Initial guess for the variable trajectory

    """

    def __init__(self, name, shape):
        """Initialize a Variable object.

        Args:
            name (str): Name identifier for the variable
            shape (tuple): Shape of the variable vector
        """
        super().__init__(name, shape)
        self._slice = None
        self._min = None
        self._max = None
        self._guess = None

    def __repr__(self):
        return f"Var({self.name!r})"

    @property
    def min(self):
        """Get the minimum bounds for the variable.

        Returns:
            np.ndarray: Array of minimum values for each variable
        """
        return self._min

    @min.setter
    def min(self, arr):
        """Set the minimum bounds for the variable.

        Args:
            arr (np.ndarray): Array of minimum values for each variable

        Raises:
            ValueError: If the shape of arr doesn't match the variable shape
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} min must be 1D with shape ({self.shape[0]},), got"
                f" {arr.shape}"
            )
        self._min = arr

    @property
    def max(self):
        """Get the maximum bounds for the variable.

        Returns:
            np.ndarray: Array of maximum values for each variable
        """
        return self._max

    @max.setter
    def max(self, arr):
        """Set the maximum bounds for the variable.

        Args:
            arr (np.ndarray): Array of maximum values for each variable

        Raises:
            ValueError: If the shape of arr doesn't match the variable shape
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} max must be 1D with shape ({self.shape[0]},), got"
                f" {arr.shape}"
            )
        self._max = arr

    @property
    def guess(self):
        """Get the initial guess for the variable trajectory.

        Returns:
            np.ndarray: Array of initial guesses for each variable at each time point
        """
        return self._guess

    @guess.setter
    def guess(self, arr):
        """Set the initial guess for the variable trajectory.

        Args:
            arr (np.ndarray): 2D array of initial guesses with shape (n_guess_points, n_variables)

        Raises:
            ValueError: If the shape of arr doesn't match the expected dimensions
        """
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"Guess must be a 2D array of shape (n_guess_points, {self.shape[0]}), got shape"
                f" {arr.shape}"
            )
        if arr.shape[1] != self.shape[0]:
            raise ValueError(
                f"Guess must have second dimension equal to variable dimension {self.shape[0]}, got"
                f" {arr.shape[1]}"
            )
        self._guess = arr

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0):
        """Append another variable or create a new variable.

        Args:
            other (Variable, optional): Another Variable object to append
            min (float, optional): Minimum bound for new variable. Defaults to -np.inf
            max (float, optional): Maximum bound for new variable. Defaults to np.inf
            guess (float, optional): Initial guess for new variable. Defaults to 0.0
        """

        def process_array(val, is_guess=False):
            """Process input array to ensure correct shape and type.

            Args:
                val: Input value to process
                is_guess (bool): Whether the value is a guess array

            Returns:
                np.ndarray: Processed array with correct shape and type
            """
            arr = np.asarray(val, dtype=float)
            if is_guess:
                return np.atleast_2d(arr)
            return np.atleast_1d(arr)

        if isinstance(other, Variable):
            self._shape = (self.shape[0] + other.shape[0],)

            if self._min is not None and other._min is not None:
                self._min = np.concatenate([self._min, process_array(other._min)], axis=0)

            if self._max is not None and other._max is not None:
                self._max = np.concatenate([self._max, process_array(other._max)], axis=0)

            if self._guess is not None and other._guess is not None:
                self._guess = np.concatenate(
                    [self._guess, process_array(other._guess, is_guess=True)], axis=1
                )

        else:
            self._shape = (self.shape[0] + 1,)

            if self._min is not None:
                self._min = np.concatenate([self._min, process_array(min)], axis=0)

            if self._max is not None:
                self._max = np.concatenate([self._max, process_array(max)], axis=0)

            if self._guess is not None:
                guess_arr = process_array(guess, is_guess=True)
                if guess_arr.shape[1] != 1:
                    guess_arr = guess_arr.T
                self._guess = np.concatenate([self._guess, guess_arr], axis=1)
