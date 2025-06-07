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
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(f"{self.__class__.__name__} min must be 1D with shape ({self.shape[0]},), got {arr.shape}")
        self._min = arr

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.shape[0]:
            raise ValueError(f"{self.__class__.__name__} max must be 1D with shape ({self.shape[0]},), got {arr.shape}")
        self._max = arr

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, arr):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Guess must be a 2D array of shape (n_guess_points, {self.shape[0]}), got shape {arr.shape}")
        if arr.shape[1] != self.shape[0]:
            raise ValueError(f"Guess must have second dimension equal to variable dimension {self.shape[0]}, got {arr.shape[1]}")
        self._guess = arr

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0):
        def process_array(val, is_guess=False):
            arr = np.asarray(val, dtype=float)
            if is_guess:
                return np.atleast_2d(arr)
            return np.atleast_1d(arr)

        if isinstance(other, Variable):
            self._shape = (self.shape[0] + other.shape[0],)

            if self._min is not None and other._min is not None:
                self._min = np.concatenate([self._min, process_array(other._min)], axis=0)

            if self._max is not None and other._max is not None:
                self._max = np.concatenate([self._max, process_array(other._max)], axis=0)

            if self._guess is not None and other._guess is not None:
                self._guess = np.concatenate([self._guess, process_array(other._guess, is_guess=True)], axis=1)

        else:
            self._shape = (self.shape[0] + 1,)

            if self._min is not None:
                self._min = np.concatenate([self._min, process_array(min)], axis=0)

            if self._max is not None:
                self._max = np.concatenate([self._max, process_array(max)], axis=0)

            if self._guess is not None:
                guess_arr = process_array(guess, is_guess=True)
                if guess_arr.shape[1] != 1:
                    guess_arr = guess_arr.T
                self._guess = np.concatenate([self._guess, guess_arr], axis=1)

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