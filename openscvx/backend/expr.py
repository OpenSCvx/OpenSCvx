# expr.py

import numpy as np


class Expr:
    def __add__(self, other):
        return Add(self, to_expr(other))

    def __mul__(self, other):
        return Mul(self, to_expr(other))

    def __matmul__(self, other):
        return MatMul(self, to_expr(other))

    def __neg__(self):
        return Neg(self)

    def children(self):
        return []

    def pretty(self, indent=0):
        pad = '  ' * indent
        lines = [f"{pad}{self.__class__.__name__}"]
        for child in self.children():
            lines.append(child.pretty(indent + 1))
        return '\n'.join(lines)


class Const(Expr):
    def __init__(self, value):
        self.value = value
        self.shape = ()  # scalar by default

    def pretty(self, indent=0):
        return '  ' * indent + f"Const({self.value})"

    def __repr__(self):
        return f"Const({self.value})"


class Variable(Expr):
    def __init__(self, name, shape):
        super().__init__()
        self.name = name
        self._shape = shape  # Use _shape so shape can be a property
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
                # vectorized processing for arrays or sequences
                vec_func = np.vectorize(self._process_bound_value)
                return vec_func(val)
            else:
                return self._process_bound_value(val)

        if isinstance(other, Variable):
            self._min = self._append_array(self._min, process_val(other._min))
            self._max = self._append_array(self._max, process_val(other._max))
            self._guess = self._append_guess(self._guess, process_val(other._guess))
            self._shape = (self._shape[0] + other._shape[0],)
        else:
            self._min = self._append_array(self._min, process_val(min))
            self._max = self._append_array(self._max, process_val(max))
            self._guess = self._append_guess(self._guess, process_val(guess))
            self._shape = (self._shape[0] + 1,)

    def _process_bound_value(self, val):
        if isinstance(val, Free):
            return val.guess
        elif isinstance(val, Minimize):
            return val.guess
        elif isinstance(val, Maximize):
            return val.guess
        elif isinstance(val, Fix):
            return val.value
        else:
            return val

    def _append_array(self, existing, new):
        new = np.atleast_1d(new)
        if existing is None:
            return new
        return np.append(existing, new)

    def _append_guess(self, existing, new):
        new = np.atleast_2d(new)
        if existing is None:
            return new
        return np.concatenate([existing, new], axis=1)

class State(Variable):
    def __init__(self, name, shape):
        super().__init__(name, shape)
        self._initial = None
        self.initial_type = None
        self._final = None
        self.final_type = None

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, arr):
        arr = np.asarray(arr, dtype=object)
        self._initial = np.zeros(arr.shape)
        self.initial_type = np.full(arr.shape, "Fixed", dtype=object)

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
                self._initial[i] = v.value
                self.initial_type[i] = "Fixed"
            else:
                self._initial[i] = v
                self.initial_type[i] = "Fixed"

    @property
    def final(self):
        return self._final

    @final.setter
    def final(self, arr):
        arr = np.asarray(arr, dtype=object)
        self._final = np.zeros(arr.shape)
        self.final_type = np.full(arr.shape, "Fixed", dtype=object)

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
                self.final_type[i] = "Fixed"
            else:
                self._final[i] = v
                self.final_type[i] = "Fixed"

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0, initial=0.0, final=0.0):
        if isinstance(other, State):
            # Direct append as before
            super().append(other=other)
            self._initial = self._append_array(self._initial, other._initial)
            self._final = self._append_array(self._final, other._final)
            self.initial_type = self._append_array(self.initial_type, other.initial_type)
            self.final_type = self._append_array(self.final_type, other.final_type)
        else:
            # Create a new temporary State from the scalar inputs
            temp_state = State(name=f"{self.name}_temp_append", shape=(1,))
            temp_state.min = min
            temp_state.max = max
            temp_state.guess = guess
            temp_state.initial = initial
            temp_state.final = final

            # Append the new State using the existing logic
            self.append(temp_state)

    def _process_val(self, val):
        if isinstance(val, Free):
            return val.guess, "Free"
        elif isinstance(val, Minimize):
            return val.guess, "Minimize"
        elif isinstance(val, Maximize):
            return val.guess, "Maximize"
        elif isinstance(val, Fix):
            return val.value, "Fixed"
        else:
            return val, "Fixed"

    def _process_boundary_array(self, arr, types):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=object)
        result = np.empty(arr.shape, dtype=float)

        for i, v in np.ndenumerate(arr):
            t = types[i] if types is not None else "Fixed"
            try:
                if t == "Free" or t == "Minimize" or t == "Maximize":
                    if hasattr(v, 'guess'):
                        result[i] = float(v.guess)
                    else:
                        result[i] = float(v)
                elif t == "Fixed":
                    if hasattr(v, 'value'):
                        result[i] = float(v.value)
                    else:
                        result[i] = float(v)
                else:
                    result[i] = float(v)
            except Exception:
                result[i] = np.nan
        return result

    def _append_array(self, existing, new):
        new = np.atleast_1d(new)
        if existing is None:
            return new
        return np.append(existing, new)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_shape = ()
        elif isinstance(idx, slice):
            new_shape = (len(range(*idx.indices(self.shape[0]))),)
        else:
            raise TypeError("State indices must be int or slice")

        new_state = State(f"{self.name}[{idx}]", new_shape)

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
        new_state._guess = slice_attr(self._guess)
        new_state._min = slice_attr(self._min)
        new_state._max = slice_attr(self._max)

        return new_state

    def __repr__(self):
        return f"State('{self.name}', shape={self.shape})"

    @property
    def shape(self):
        return self._shape
    
class Control(Variable):
    def __init__(self, name, shape):
        super().__init__(name, shape)

    def append(self, other=None, *, min=-np.inf, max=np.inf, guess=0.0):
        if isinstance(other, Control):
            # Append another Control; keep self.name unchanged
            self.min = np.concatenate((self.min, other.min), axis=0) if self.min is not None else other.min
            self.max = np.concatenate((self.max, other.max), axis=0) if self.max is not None else other.max
            self.guess = np.concatenate((self.guess, other.guess), axis=1) if self.guess is not None else other.guess
            self._shape = (self.shape[0] + other.shape[0],)
        else:
            # Create temporary Control with unique name, append it, keep original name intact
            temp_control = Control(name=f"{self.name}_temp_append", shape=(1,))
            temp_control.min = np.array([min])
            temp_control.max = np.array([max])
            temp_control.guess = np.array([[guess]])
            
            # Append temp_control (which will not modify self.name)
            self.append(temp_control)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Control(f"{self.name}[{idx}]", shape=())
        elif isinstance(idx, slice):
            length = len(range(*idx.indices(self.shape[0])))
            return Control(f"{self.name}[{idx.start}:{idx.stop}]", shape=(length,))
        else:
            raise TypeError("Control indices must be int or slice")

    def __repr__(self):
        return f"Control('{self.name}', shape={self.shape})"




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


class Fix:
    """Represents a fixed value for a boundary condition."""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Fix({self.value})"


class Free:
    """Represents a free variable with an initial guess."""
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Free({self.guess})"


class Minimize:
    """Represents a free variable that should be minimized, with an initial guess."""
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Minimize({self.guess})"


class Maximize:
    """Represents a free variable that should be maximized, with an initial guess."""
    def __init__(self, guess):
        self.guess = guess

    def __repr__(self):
        return f"Maximize({self.guess})"


# Placeholder classes and function for symbolic operations
# (You can implement these based on your symbolic system)

def to_expr(obj):
    # Converts input to an Expr if needed, e.g., wraps numbers as Const
    if isinstance(obj, Expr):
        return obj
    else:
        return Const(obj)


class Add(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class MatMul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]


class Neg(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]
