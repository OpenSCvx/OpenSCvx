import jax.numpy as jnp
from jax import jit
import operator

class Symbol:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape

    def __getitem__(self, key):
        return Expression(lambda env: env[self.name][key], f"{self.name}[{key}]")

    def __sub__(self, other):
        return Expression(lambda env: env[self.name] - _eval(other, env), f"{self.name} - {other}")

    def __rsub__(self, other):
        return Expression(lambda env: _eval(other, env) - env[self.name], f"{other} - {self.name}")

    def __matmul__(self, other):
        return Expression(lambda env: env[self.name] @ _eval(other, env), f"{self.name} @ {other}")

    def T(self):
        return Expression(lambda env: env[self.name].T, f"{self.name}.T")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class Variable(Symbol):
    pass

class Parameter(Symbol):
    pass

def _eval(obj, env):
    if isinstance(obj, Expression):
        return obj.func(env)
    elif isinstance(obj, Symbol):
        return env[obj.name]
    elif callable(obj):
        return obj(env)
    else:
        return obj

class Expression:
    def __init__(self, func, desc=None):
        self.func = func  # A function: env → jax value
        self.desc = desc

    def __call__(self, env):
        return self.func(env)

    def __add__(self, other):
        return Expression(lambda env: self.func(env) + _eval(other, env), f"({self.desc} + {other})")

    def __radd__(self, other):
        return Expression(lambda env: _eval(other, env) + self.func(env), f"({other} + {self.desc})")

    def __sub__(self, other):
        return Expression(lambda env: self.func(env) - _eval(other, env), f"({self.desc} - {other})")

    def __rsub__(self, other):
        return Expression(lambda env: _eval(other, env) - self.func(env), f"({other} - {self.desc})")

    def __matmul__(self, other):
        return Expression(lambda env: self.func(env) @ _eval(other, env), f"({self.desc} @ {other})")

    def T(self):
        return Expression(lambda env: self.func(env).T, f"({self.desc}.T)")

    def to_jax_function(self):
        @jit
        def f(env):
            return self.func(env)
        return f

    def __str__(self):
        return self.desc or "<expression>"

    def __repr__(self):
        return f"Expression({self.desc})"
