import numpy as np

from openscvx.backend.variable import Variable


class Fix:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Fix({self.value})"


class Free:
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Free({self.guess})"


class Minimize:
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Minimize({self.guess})"


class Maximize:
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Maximize({self.guess})"


class State(Variable):
    def __init__(self, name, shape):
        super().__init__(name, shape)
        self._initial = None
        self.initial_type = None
        self._final = None
        self.final_type = None

        self._true_dim = shape[0]
        self._update_slices()

    def _update_slices(self):
        self._true_slice = slice(0, self._true_dim)
        self._augmented_slice = slice(self._true_dim, self.shape[0])

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, arr):
        arr = np.asarray(arr, dtype=object)
        self._initial = np.zeros(arr.shape)
        self.initial_type = np.full(arr.shape, "Fix", dtype=object)

        for i, v in np.ndenumerate(arr):
            if isinstance(v, Free):
                self._initial[i] = v.guess
                self.initial_type[i] = "Free"
            elif isinstance(v, Minimize):
                self._initial[i] = v.guess
                self.initial_type[i] = "Minimize"
            elif isinstance(v, Maximize):
                self._initial[i] = v.guess
                self.final_type[i] = "Maximize"
            elif isinstance(v, Fix):
                self._initial[i] = v.value
                self.initial_type[i] = "Fix"
            else:
                self._initial[i] = v
                self.initial_type[i] = "Fix"

    @property
    def final(self):
        return self._final

    @final.setter
    def final(self, arr):
        arr = np.asarray(arr, dtype=object)
        self._final = np.zeros(arr.shape)
        self.final_type = np.full(arr.shape, "Fix", dtype=object)

        for i, v in np.ndenumerate(arr):
            if isinstance(v, Free):
                self._final[i] = v.guess
                self.final_type[i] = "Free"
            elif isinstance(v, Minimize):
                self._final[i] = v.guess
                self.final_type[i] = "Minimize"
            elif isinstance(v, Maximize):
                self._final[i] = v.guess
                self.final_type[i] = "Maximize"
            elif isinstance(v, Fix):
                self._final[i] = v.value
                self.final_type[i] = "Fix"
            else:
                self._final[i] = v
                self.final_type[i] = "Fix"

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, initial=0.0, final=0.0, augmented=False):
        if isinstance(other, State):
            super().append(other=other)

            if self._initial is not None and other._initial is not None:
                self._initial = self._append_array(self._initial, other._initial)
            if self._final is not None and other._final is not None:
                self._final = self._append_array(self._final, other._final)
            if self.initial_type is not None and other.initial_type is not None:
                self.initial_type = self._append_array(self.initial_type, other.initial_type)
            if self.final_type is not None and other.final_type is not None:
                self.final_type = self._append_array(self.final_type, other.final_type)

            if not augmented:
                self._true_dim += other._true_dim if hasattr(other, "_true_dim") else other.shape[0]
            self._update_slices()
        else:
            temp_state = State(name=f"{self.name}_temp_append", shape=(1,))
            temp_state.min = min
            temp_state.max = max
            temp_state.guess = guess
            temp_state.initial = initial
            temp_state.final = final
            self.append(temp_state, augmented=augmented)

    @property
    def true_state(self):
        return self[self._true_slice]

    @property
    def augmented_state(self):
        return self[self._augmented_slice]

    def __getitem__(self, idx):
        new_state = super().__getitem__(idx)
        new_state.__class__ = State

        def slice_attr(attr):
            if attr is None:
                return None
            if attr.ndim == 2 and attr.shape[1] == self.shape[0]:
                return attr[:, idx]
            return attr[idx]

        new_state._initial = slice_attr(self._initial)
        new_state.initial_type = slice_attr(self.initial_type)
        new_state._final = slice_attr(self._final)
        new_state.final_type = slice_attr(self.final_type)

        # Calculate the new _true_dim under the slice
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.shape[0]
            step = idx.step or 1
            selected = np.arange(self.shape[0])[idx]
        elif isinstance(idx, (list, np.ndarray)):
            selected = np.array(idx)
        else:  # assume int
            selected = np.array([idx])

        new_state._true_dim = np.sum(selected < self._true_dim)
        new_state._update_slices()

        return new_state

    def __repr__(self):
        return f"State('{self.name}', shape={self.shape})"