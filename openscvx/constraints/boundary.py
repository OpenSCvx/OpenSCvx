from dataclasses import dataclass, field
from typing import Union, Sequence
import jax.numpy as jnp

ALLOWED_TYPES = {"Fix", "Free", "Minimize", "Maximize"}

@dataclass
class BoundaryConstraint:
    value: jnp.ndarray
    types: list[str] = field(init=False)

    def __post_init__(self):
        self.types = ["Fix"] * len(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, val):
        self.value = self.value.at[key].set(val)

    @property
    def type(self):
        constraint = self
        class TypeProxy:
            def __getitem__(self, key):
                return constraint.types[key]

            def __setitem__(self, key, val: Union[str, Sequence[str]]):
                indices = range(*key.indices(len(constraint.types))) if isinstance(key, slice) else [key]
                values = [val] * len(indices) if isinstance(val, str) else val

                if len(values) != len(indices):
                    raise ValueError("Mismatch between indices and values length")

                for idx, v in zip(indices, values):
                    if v not in ALLOWED_TYPES:
                        raise ValueError(f"Invalid type: {v}, must be one of {ALLOWED_TYPES}")
                    constraint.types[idx] = v

            def __len__(self):
                return len(constraint.types)

            def __repr__(self):
                return repr(constraint.types)

        return TypeProxy()

def boundary(arr: jnp.ndarray):
    return BoundaryConstraint(arr)
