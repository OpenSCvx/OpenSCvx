"""Parameter dictionary that syncs between JAX and CVXPy."""

import numpy as np


class ParameterDict(dict):
    """Dictionary that syncs to both internal _parameters dict and CVXPy parameters.

    This allows users to naturally update parameters like:
        problem.parameters["obs_radius"] = 2.0

    Changes automatically propagate to:
    1. Internal _parameters dict (plain dict for JAX)
    2. CVXPy parameters (for optimization)
    """

    def __init__(self, problem, internal_dict, *args, **kwargs):
        self._problem = problem
        self._internal_dict = internal_dict  # Reference to plain dict for JAX
        super().__init__()
        # Initialize with float enforcement by using __setitem__
        if args:
            other = args[0]
            if hasattr(other, "items"):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        # Enforce float dtype to prevent int/float mismatch bugs
        value = np.asarray(value, dtype=float)
        super().__setitem__(key, value)
        # Sync to internal dict for JAX
        self._internal_dict[key] = value
        # Sync to CVXPy if it exists
        lowered = getattr(self._problem, "_lowered", None)
        if lowered is not None and key in lowered.cvxpy_params:
            lowered.cvxpy_params[key].value = value

    def update(self, other=None, **kwargs):
        """Update multiple parameters and sync to internal dict and CVXPy."""
        if other is not None:
            if hasattr(other, "items"):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value
