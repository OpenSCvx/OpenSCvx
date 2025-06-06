import numpy as np

from openscvx.backend.expr import Expr

class Variable(Expr):
    def __init__(self, name, shape):
        super().__init__()
        self.name = name
        self._shape = shape
        self._min = None
        self._max = None
        self._guess = None

    @property
    def shape(self):
        return self._shape

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, arr):
        self._min = np.asarray(arr)

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, arr):
        self._max = np.asarray(arr)

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, arr):
        self._guess = np.asarray(arr)

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0):
        def process_val(val):
            if isinstance(val, (np.ndarray, list, tuple)):
                vec_func = np.vectorize(lambda v: v)
                return vec_func(val)
            else:
                return val

        if isinstance(other, Variable):
            if self._min is not None and other._min is not None:
                self._min = self._append_array(self._min, process_val(other._min))
            if self._max is not None and other._max is not None:
                self._max = self._append_array(self._max, process_val(other._max))
            if self._guess is not None and other._guess is not None:
                self._guess = self._append_guess(self._guess, process_val(other._guess))
            self._shape = (self._shape[0] + other._shape[0],)
        else:
            if self._min is not None and min is not None:
                self._min = self._append_array(self._min, process_val(min))
            if self._max is not None and max is not None:
                self._max = self._append_array(self._max, process_val(max))
            if self._guess is not None and guess is not None:
                self._guess = self._append_guess(self._guess, process_val(guess))
            self._shape = (self._shape[0] + 1,)

    def _append_array(self, existing, new):
        new = np.atleast_1d(new)
        if existing is None:
            return new
        existing = np.atleast_1d(existing)
        return np.concatenate([existing, new], axis=0)

    def _append_guess(self, existing, new):
        new = np.atleast_2d(new)
        if new.shape[0] == 1 and new.shape[1] != 1:
            # Ensure it's a column vector if it was (1, N)
            new = new.T
        if existing is None:
            return new
        existing = np.atleast_2d(existing)
        if existing.shape[0] == 1 and existing.shape[1] != 1:
            existing = existing.T
        return np.concatenate([existing, new], axis=1)


    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_shape = ()
        elif isinstance(idx, slice):
            new_shape = (len(range(*idx.indices(self.shape[0]))),)
        else:
            raise TypeError("Variable indices must be int or slice")

        sliced = Variable(f"{self.name}[{idx}]", new_shape)

        def slice_attr(attr):
            if attr is None:
                return None
            if attr.ndim == 2 and attr.shape[1] == self.shape[0]:
                return attr[:, idx]
            return attr[idx]

        sliced._min = slice_attr(self._min)
        sliced._max = slice_attr(self._max)
        sliced._guess = slice_attr(self._guess)

        return sliced