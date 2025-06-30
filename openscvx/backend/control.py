import numpy as np
from openscvx.backend.variable import Variable

class Control(Variable):
    """A class representing the control variables in an optimal control problem.

    The Control class extends Variable to handle control-specific properties and supports
    both true and augmented control dimensions. It provides methods for appending new control
    variables and accessing subsets of the control vector.

    Attributes:
        name (str): Name of the control variable.
        shape (tuple): Shape of the control variable array.
        min (np.ndarray): Minimum bounds for the control variables. Shape: (n_controls,).
        max (np.ndarray): Maximum bounds for the control variables. Shape: (n_controls,).
        guess (np.ndarray): Used to initialize SCP and contains the current SCP solution for the control trajectory. Shape: (n_nodes, n_controls).
        _true_dim (int): True dimensionality of the control variables.
        _true_slice (slice): Slice for accessing true control variables.
        _augmented_slice (slice): Slice for accessing augmented control variables.

    Notes:
        Attributes prefixed with underscore (_) are for internal use only and should not be accessed directly.

    Example:
    ```python
    control = Control("thrust", (3,))
    control.min = [-1, -1, 0]
    control.max = [1, 1, 10]
    control.guess = np.repeat([[0, 0, 10]], 5, axis=0)
    ```
    """

    def __init__(self, name, shape):
        """Initialize a Control object.

        Args:
            name (str): Name identifier for the control variable
            shape (tuple): Shape of the control vector
        """
        super().__init__(name, shape)
        self._true_dim = shape[0]
        self._update_slices()

    def _update_slices(self):
        """Update the slice objects for true and augmented controls."""
        self._true_slice = slice(0, self._true_dim)
        self._augmented_slice = slice(self._true_dim, self.shape[0])

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, augmented=False):
        """Append another control or create a new control variable.

        Args:
            other (Control, optional): Another Control object to append
            min (float, optional): Minimum bound for new control. Defaults to -np.inf
            max (float, optional): Maximum bound for new control. Defaults to np.inf
            guess (float, optional): Initial guess for new control. Defaults to 0.0
            augmented (bool, optional): Whether the new control is augmented. Defaults to False
        """
        if isinstance(other, Control):
            super().append(other=other)
            if not augmented:
                self._true_dim += getattr(other, "_true_dim", other.shape[0])
            self._update_slices()
        else:
            temp = Control(name=f"{self.name}_temp_append", shape=(1,))
            temp.min = min
            temp.max = max
            temp.guess = guess
            self.append(temp, augmented=augmented)

    @property
    def true(self):
        """Get the true control variables (excluding augmented controls).

        Returns:
            Control: A new Control object containing only the true control variables
        """
        return self[self._true_slice]

    @property
    def augmented(self):
        """Get the augmented control variables.

        Returns:
            Control: A new Control object containing only the augmented control variables
        """
        return self[self._augmented_slice]

    def __getitem__(self, idx):
        """Get a subset of the control variables.

        Args:
            idx: Index or slice to select control variables

        Returns:
            Control: A new Control object containing the selected variables
        """
        new_ctrl = super().__getitem__(idx)
        new_ctrl.__class__ = Control

        if isinstance(idx, slice):
            selected = np.arange(self.shape[0])[idx]
        elif isinstance(idx, (list, np.ndarray)):
            selected = np.array(idx)
        else:
            selected = np.array([idx])

        new_ctrl._true_dim = np.sum(selected < self._true_dim)
        new_ctrl._update_slices()

        return new_ctrl

    def __repr__(self):
        """String representation of the Control object.

        Returns:
            str: A string describing the Control object
        """
        return f"Control('{self.name}', shape={self.shape})"