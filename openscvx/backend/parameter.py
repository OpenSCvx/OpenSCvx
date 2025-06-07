import numpy as np
from openscvx.backend.expr import Expr


class Parameter(Expr):
    _registry = {}

    def __init__(self, name, shape=()):
        super().__init__()
        self.name = name
        self._shape = shape
        self.value = None

        # Automatically register the parameter if not already present
        if name not in Parameter._registry:
            Parameter._registry[name] = self

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        if isinstance(idx, int):
            param = Parameter(f"{self.name}[{idx}]", shape=())
        elif isinstance(idx, slice):
            length = len(range(*idx.indices(self.shape[0])))
            param = Parameter(f"{self.name}[{idx.start}:{idx.stop}]", shape=(length,))
        else:
            raise TypeError("Parameter indices must be int or slice")

        return param

    def __repr__(self):
        return f"Parameter('{self.name}', shape={self.shape})"

    @classmethod
    def get_all(cls):
        return dict(cls._registry)

    @classmethod
    def reset(cls):
        cls._registry.clear()
