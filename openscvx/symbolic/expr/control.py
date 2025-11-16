from .variable import Variable


class Control(Variable):
    """Control input variable for trajectory optimization problems.

    Control represents control input variables (actuator commands) in a trajectory
    optimization problem. Unlike State variables which evolve according to dynamics,
    Controls are direct decision variables that the optimizer can freely adjust
    (within specified bounds) at each time step to influence the system dynamics.

    Controls are conceptually similar to State variables but simpler - they don't
    have boundary conditions (initial/final specifications) since controls are
    typically not constrained at the endpoints. Like States, Controls support:

    - Min/max bounds to enforce actuator limits
    - Initial trajectory guesses to help the optimizer converge

    Common examples of control inputs include:

    - Thrust magnitude and direction for spacecraft/rockets
    - Throttle settings for engines
    - Steering angles for vehicles
    - Torques for robotic manipulators
    - Force/acceleration commands

    Attributes:
        name (str): Unique name identifier for this control variable
        _shape (tuple[int, ...]): Shape of the control vector (typically 1D like (3,) for 3D thrust)
        _slice (slice | None): Internal slice information for variable indexing
        _min (np.ndarray | None): Minimum bounds for each element of the control
        _max (np.ndarray | None): Maximum bounds for each element of the control
        _guess (np.ndarray | None): Initial guess for the control trajectory (n_points, n_controls)

    Example:
        Scalar throttle control bounded [0, 1]:

            throttle = Control("throttle", shape=(1,))
            throttle.min = [0.0]
            throttle.max = [1.0]
            throttle.guess = np.full((50, 1), 0.5)  # Start at 50% throttle

        3D thrust vector for spacecraft:

            thrust = Control("thrust", shape=(3,))
            thrust.min = [-10, -10, 0]    # No downward thrust
            thrust.max = [10, 10, 50]     # Limited thrust
            thrust.guess = np.zeros((50, 3))  # Initialize with zero thrust

        2D steering control (left/right, forward/backward):

            steer = Control("steer", shape=(2,))
            steer.min = [-1, -1]
            steer.max = [1, 1]
            steer.guess = np.linspace([0, 0], [0, 1], 50)  # Gradual acceleration
    """

    def __init__(self, name, shape):
        """Initialize a Control object.

        Args:
            name: Name identifier for the control variable
            shape: Shape of the control vector (typically 1D tuple like (3,))
        """
        super().__init__(name, shape)

    def __repr__(self):
        """String representation of the Control object.

        Returns:
            Concise string showing the control name and shape.
        """
        return f"Control('{self.name}', shape={self.shape})"
