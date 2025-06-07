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
    def min(self):
        return self._min

    @min.setter
    def min(self, val):
        val = np.asarray(val)
        if val.shape != self.shape:
            raise ValueError(f"Min shape {val.shape} does not match State shape {self.shape}")
        self._min = val
        self._check_bounds_against_initial_final()

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, val):
        val = np.asarray(val)
        if val.shape != self.shape:
            raise ValueError(f"Max shape {val.shape} does not match State shape {self.shape}")
        self._max = val
        self._check_bounds_against_initial_final()

    def _check_bounds_against_initial_final(self):
        for field_name, data, types in [('initial', self._initial, self.initial_type),
                                        ('final', self._final, self.final_type)]:
            if data is None or types is None:
                continue
            for i, val in np.ndenumerate(data):
                if types[i] != "Fix":
                    continue
                min_i = self._min[i] if self._min is not None else -np.inf
                max_i = self._max[i] if self._max is not None else np.inf
                if val < min_i:
                    raise ValueError(f"{field_name.capitalize()} Fixed value at index {i[0]} is lower then the min: {val} < {min_i}")
                if val > max_i:
                    raise ValueError(f"{field_name.capitalize()} Fixed value at index {i[0]} is greater then the max: {val} > {max_i}")

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, arr):
        arr = np.asarray(arr, dtype=object)
        if arr.shape != self.shape:
            raise ValueError(f"Initial value shape {arr.shape} does not match State shape {self.shape}")
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
                self.initial_type[i] = "Maximize"
            elif isinstance(v, Fix):
                val = v.value
                self._initial[i] = val
                self.initial_type[i] = "Fix"
            else:
                val = v
                self._initial[i] = val
                self.initial_type[i] = "Fix"

        self._check_bounds_against_initial_final()

    @property
    def final(self):
        return self._final

    @final.setter
    def final(self, arr):
        arr = np.asarray(arr, dtype=object)
        if arr.shape != self.shape:
            raise ValueError(f"Final value shape {arr.shape} does not match State shape {self.shape}")
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
                val = v.value
                self._final[i] = val
                self.final_type[i] = "Fix"
            else:
                val = v
                self._final[i] = val
                self.final_type[i] = "Fix"

        self._check_bounds_against_initial_final()

    @property
    def true(self):
        return self[self._true_slice]

    @property
    def augmented(self):
        return self[self._augmented_slice]

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, initial=0.0, final=0.0, augmented=False):
        if isinstance(other, State):
            super().append(other=other)

            if self._initial is None:
                self._initial = np.array(other._initial) if other._initial is not None else None
            elif other._initial is not None:
                self._initial = np.concatenate([self._initial, other._initial], axis=0)

            if self._final is None:
                self._final = np.array(other._final) if other._final is not None else None
            elif other._final is not None:
                self._final = np.concatenate([self._final, other._final], axis=0)

            if self.initial_type is None:
                self.initial_type = np.array(other.initial_type) if other.initial_type is not None else None
            elif other.initial_type is not None:
                self.initial_type = np.concatenate([self.initial_type, other.initial_type], axis=0)

            if self.final_type is None:
                self.final_type = np.array(other.final_type) if other.final_type is not None else None
            elif other.final_type is not None:
                self.final_type = np.concatenate([self.final_type, other.final_type], axis=0)

            if not augmented:
                self._true_dim += getattr(other, "_true_dim", other.shape[0])
            self._update_slices()
        else:
            temp_state = State(name=f"{self.name}_temp_append", shape=(1,))
            temp_state.min = min
            temp_state.max = max
            temp_state.guess = guess
            temp_state.initial = initial
            temp_state.final = final
            self.append(temp_state, augmented=augmented)

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

        if isinstance(idx, slice):
            selected = np.arange(self.shape[0])[idx]
        elif isinstance(idx, (list, np.ndarray)):
            selected = np.array(idx)
        else:
            selected = np.array([idx])

        new_state._true_dim = np.sum(selected < self._true_dim)
        new_state._update_slices()

        return new_state

    def __repr__(self):
        return f"State('{self.name}', shape={self.shape})"
