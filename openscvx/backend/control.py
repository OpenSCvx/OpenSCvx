import numpy as np
from openscvx.backend.variable import Variable

class Control(Variable):
    def __init__(self, name, shape):
        super().__init__(name, shape)
        self._true_slice = slice(0, shape[0])
        self._augmented_slice = slice(shape[0], shape[0])  # starts empty

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, augmented=False):
        if isinstance(other, Control):
            super().append(other=other)
            new_total = self.shape[0]
            true_len = self._true_slice.stop
            if augmented:
                self._augmented_slice = slice(true_len, new_total)
            else:
                self._true_slice = slice(0, true_len + other.shape[0])
        else:
            temp_control = Control(name=f"{self.name}_temp_append", shape=(1,))
            temp_control.min = min
            temp_control.max = max
            temp_control.guess = guess
            self.append(temp_control, augmented=augmented)

    def true_control(self):
        return self[self._true_slice]

    def augmented_control(self):
        return self[self._augmented_slice]

    def __getitem__(self, idx):
        new_control = super().__getitem__(idx)
        new_control.__class__ = Control
        return new_control

    def __repr__(self):
        return f"Control('{self.name}', shape={self.shape})"
