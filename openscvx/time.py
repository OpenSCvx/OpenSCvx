from typing import Union


class Time:
    """Time configuration for trajectory optimization problems.

    This class encapsulates time-related parameters for trajectory optimization.
    The time derivative is internally assumed to be 1.0.

    Attributes:
        initial (float or tuple): Initial time boundary condition.
            Can be a float (fixed) or tuple like ("free", value), ("minimize", value),
            or ("maximize", value).
        final (float or tuple): Final time boundary condition.
            Can be a float (fixed) or tuple like ("free", value), ("minimize", value),
            or ("maximize", value).
        min (float): Minimum bound for time variable (required).
        max (float): Maximum bound for time variable (required).

    Example:
        ```python
        # Fixed initial and final time
        time = Time(initial=0.0, final=10.0, min=0.0, max=20.0)

        # Free final time
        time = Time(initial=0.0, final=("free", 10.0), min=0.0, max=20.0)

        # Minimize final time
        time = Time(initial=0.0, final=("minimize", 10.0), min=0.0, max=20.0)

        # Maximize initial time
        time = Time(initial=("maximize", 0.0), final=10.0, min=0.0, max=20.0)
        ```
    """

    def __init__(
        self,
        initial: Union[float, tuple],
        final: Union[float, tuple],
        min: float,
        max: float,
    ):
        """Initialize a Time object.

        Args:
            initial: Initial time boundary condition (float or tuple).
                Tuple format: ("free", value), ("minimize", value), or ("maximize", value).
            final: Final time boundary condition (float or tuple).
                Tuple format: ("free", value), ("minimize", value), or ("maximize", value).
            min: Minimum bound for time variable (required).
            max: Maximum bound for time variable (required).

        Raises:
            ValueError: If tuple format is invalid.
        """
        # Validate tuple format if provided
        for name, value in [("initial", initial), ("final", final)]:
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"{name} tuple must have exactly 2 elements: (type, value)")
                bc_type, bc_value = value
                if bc_type not in ["free", "minimize", "maximize"]:
                    raise ValueError(
                        f"{name} boundary condition type must be 'free', "
                        f"'minimize', or 'maximize', got '{bc_type}'"
                    )
                if not isinstance(bc_value, (int, float)):
                    raise ValueError(
                        f"{name} boundary condition value must be a number, "
                        f"got {type(bc_value).__name__}"
                    )

        self.initial = initial
        self.final = final
        self.min = min
        self.max = max
        # Time derivative is always 1.0 internally
        self.derivative = 1.0
