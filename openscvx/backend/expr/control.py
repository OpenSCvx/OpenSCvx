from ..canonicalizer import canon_visitor
from ..shape_checker import shape_visitor
from .variable import Variable


class Control(Variable):
    """Symbolic control variable for trajectory optimization.

    A Control represents a named control variable in the symbolic expression tree.
    It integrates with the AST system for trajectory optimization problems.

    The Control class is designed to be lightweight and focused on symbolic
    representation, with optimization-specific functionality handled by the
    unified layer.

    Attributes:
        name (str): Unique name identifier for this control variable
        shape (tuple): Shape of the control vector (e.g., (2,) for 2D thrust)
        min (np.ndarray): Minimum bounds for control variables
        max (np.ndarray): Maximum bounds for control variables
        guess (np.ndarray): Initial trajectory guess

    Example:
        ```python
        # Simple scalar control
        throttle = Control("throttle", (1,))
        throttle.min = np.array([0.0])
        throttle.max = np.array([1.0])
        throttle.guess = np.full((10, 1), 0.5)  # 10 time steps

        # Vector control
        thrust = Control("thrust", (3,))
        thrust.min = np.array([-1, -1, 0])
        thrust.max = np.array([1, 1, 10])
        thrust.guess = np.repeat([[0, 0, 5]], 10, axis=0)
        ```
    """

    def __init__(self, name, shape):
        """Initialize a Control object.

        Args:
            name (str): Name identifier for the control variable
            shape (tuple): Shape of the control vector
        """
        super().__init__(name, shape)

    def __repr__(self):
        """String representation of the Control object.

        Returns:
            str: A string describing the Control object
        """
        return f"Control('{self.name}', shape={self.shape})"


@canon_visitor(Control)
def canon_control(node):
    # Control nodes are already canonical
    return node


@shape_visitor(Control)
def check_shape_control(v):
    return v.shape
