import numpy as np
from openscvx.backend.variable import Variable

class Control(Variable):
    def __init__(self, name, shape):
        super().__init__(name, shape)
        self._true_dim = shape[0]
        self._update_slices()

    def _update_slices(self):
        self._true_slice = slice(0, self._true_dim)
        self._augmented_slice = slice(self._true_dim, self.shape[0])

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, augmented=False):
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
        return self[self._true_slice]

    @property
    def augmented(self):
        return self[self._augmented_slice]

    def __getitem__(self, idx):
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
        return f"Control('{self.name}', shape={self.shape})"