import numpy as np
from openscvx.backend.expr import Expr


class Parameter(Expr):
    """A class representing a parameter in the optimization problem.
    
    Parameters are symbolic variables that can be used in expressions and constraints.
    They maintain a registry of all created parameters and can be indexed or sliced.
    
    Attributes:
        _registry (dict): Class-level dictionary storing all created parameters.
        name (str): The name of the parameter.
        _shape (tuple): The shape of the parameter.
        value (any): The current value of the parameter, initially None.
    """

    _registry = {}

    def __init__(self, name, shape=()):
        """Initialize a new Parameter.
        
        Args:
            name (str): The name of the parameter.
            shape (tuple, optional): The shape of the parameter. Defaults to ().
        
        Note:
            The parameter is automatically registered in the class registry if not already present.
        """
        super().__init__()
        self.name = name
        self._shape = shape
        self.value = None

        # Automatically register the parameter if not already present
        if name not in Parameter._registry:
            Parameter._registry[name] = self

    @property
    def shape(self):
        """Get the shape of the parameter.
        
        Returns:
            tuple: The shape of the parameter.
        """
        return self._shape

    def __getitem__(self, idx):
        """Get a subset of the parameter using indexing or slicing.
        
        Args:
            idx (int or slice): The index or slice to use.
                - If int: Returns a scalar parameter
                - If slice: Returns a parameter with shape (length of slice,)
        
        Returns:
            Parameter: A new parameter representing the subset.
        
        Raises:
            TypeError: If idx is neither an int nor a slice.
        """
        if isinstance(idx, int):
            param = Parameter(f"{self.name}[{idx}]", shape=())
        elif isinstance(idx, slice):
            length = len(range(*idx.indices(self.shape[0])))
            param = Parameter(f"{self.name}[{idx.start}:{idx.stop}]", shape=(length,))
        else:
            raise TypeError("Parameter indices must be int or slice")

        return param

    def __repr__(self):
        """Get a string representation of the parameter.
        
        Returns:
            str: A string showing the parameter name and shape.
        """
        return f"Parameter('{self.name}', shape={self.shape})"

    @classmethod
    def get_all(cls):
        """Get all registered parameters.
        
        Returns:
            dict: A dictionary of all registered parameters, with names as keys.
        """
        return dict(cls._registry)

    @classmethod
    def reset(cls):
        """Clear the registry of all parameters.
        
        This method removes all registered parameters from the class registry.
        """
        cls._registry.clear()
