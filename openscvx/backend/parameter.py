import numpy as np

from openscvx.backend.expr import Expr


class Parameter(Expr):
    def __init__(self, name, shape=()):
        super().__init__()
        self.name = name
        self._shape = shape
        self.value = None

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Parameter(f"{self.name}[{idx}]", shape=())
        elif isinstance(idx, slice):
            length = len(range(*idx.indices(self.shape[0])))
            return Parameter(f"{self.name}[{idx.start}:{idx.stop}]", shape=(length,))
        else:
            raise TypeError("Parameter indices must be int or slice")

    def __repr__(self):
        return f"Parameter('{self.name}', shape={self.shape})"


