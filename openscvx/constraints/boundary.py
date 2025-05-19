from dataclasses import dataclass, field
from typing import Union, Sequence
import jax.numpy as jnp

ALLOWED_TYPES = {"Fix", "Free", "Minimize", "Maximize"}


@dataclass
class BoundaryConstraint:
    """
    Holds boundary constraints for a trajectory optimization problem 
    _i.e._ _initial_ and _terminal_ state constraints.

    Each element of the `value` array corresponds to a state variable at either
    the initial or terminal time.  The parallel `types` list specifies how each
    state should be treated by the optimizer:

      - "Fix": enforce the state exactly.
      - "Free": allow the optimizer to choose freely.
      - "Minimize": include the state in the cost to be minimized.
      - "Maximize": include the state in the cost to be maximized.

    Note: for convenience and consistency with other constraint types,
    the user should use the `boundary` factory to instantiate a `BoundaryConstraint` object.

    Args:
        value (jnp.ndarray): Array of boundary values.  Length N.
    """

    value: jnp.ndarray
    types: list[str] = field(init=False)

    def __post_init__(self):
        """
        Initialize all constraint types to "Fix" by default.
        """
        self.types = ["Fix"] * len(self.value)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, val):
        self.value = self.value.at[key].set(val)

    @property
    def type(self):
        """
        Proxy for getting and setting constraint types.

        Use indexing or slicing to read/write the `types` list in lockstep
        with `value`.  Examples:

            >>> bc = BoundaryConstraint(jnp.array([0.0, 1.0]))
            >>> bc.type[0]
            'Fix'
            >>> bc.type[1] = 'Free'
            >>> bc.type[:] = ['Minimize', 'Maximize']

        Raises:
            ValueError: If you try to assign a type not in ALLOWED_TYPES.

        Returns:
            TypeProxy: An object that implements __getitem__ and __setitem__
            on the underlying `types` list.
        """
        constraint = self

        class TypeProxy:
            def __getitem__(self, key):
                return constraint.types[key]

            def __setitem__(self, key, val: Union[str, Sequence[str]]):
                indices = (
                    range(*key.indices(len(constraint.types)))
                    if isinstance(key, slice)
                    else [key]
                )
                values = [val] * len(indices) if isinstance(val, str) else val

                if len(values) != len(indices):
                    raise ValueError("Mismatch between indices and values length")

                for idx, v in zip(indices, values):
                    if v not in ALLOWED_TYPES:
                        raise ValueError(
                            f"Invalid type: {v}, must be one of {ALLOWED_TYPES}"
                        )
                    constraint.types[idx] = v

            def __len__(self):
                return len(constraint.types)

            def __repr__(self):
                return repr(constraint.types)

        return TypeProxy()


def boundary(arr: jnp.ndarray):
    """
    Convenience factory to build a `BoundaryConstraint`, _i.e._ initial or terminal state constraint,
    from an array of values.
    Each element of the input array corresponds to a state variable at either
    the initial or terminal time.  The parallel `types` list specifies how each
    state should be treated by the optimizer:

      - "Fix": enforce the state exactly.
      - "Free": allow the optimizer to choose freely.
      - "Minimize": include the state in the cost to be minimized.
      - "Maximize": include the state in the cost to be maximized.
    
    Note that the `type` is initialzed as `Fix` for each value and can be updated after instantiating
    the `BoundaryConstraint`

    Args:
        arr (jnp.ndarray): Array of boundary values.

    Returns:
        BoundaryConstraint: With `value == arr` and all `types == "Fix"`.

    Examples:
        >>> initial_state = boundary(jnp.array(10.0, 0.0, 2.0, 6.0))
        ... initial_state.type[0:3] = "Free
        ... initial_state.type[3] = "Minimize"
    """
    return BoundaryConstraint(arr)
