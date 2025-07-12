from typing import Iterable, Set

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import Expr, traverse
from openscvx.backend.state import State


def validate_variable_names(
    exprs: Iterable[Expr], *, reserved_prefix: str = "_", reserved_names: Set[str] = None
) -> None:
    """
    1) Ensure all State/Control names are unique.
    2) Ensure no user-supplied name starts with `reserved_prefix`.
    3) Ensure no name collides with `reserved_names` if given.
    Raises ValueError on any violation.
    """
    seen = set()
    reserved = set(reserved_names or ())

    def visitor(node):
        if isinstance(node, (State, Control)):
            name = node.name
            # 1) uniqueness
            if name in seen:
                raise ValueError(f"Duplicate variable name: {name!r}")
            # 2) no user-underscore
            if name.startswith(reserved_prefix):
                raise ValueError(
                    f"Variable name {name!r} is reserved (cannot start with {reserved_prefix!r})"
                )
            # 3) no collision with explicit reserved set
            if name in reserved:
                raise ValueError(f"Variable name {name!r} collides with reserved name")
            seen.add(name)

    for e in exprs:
        traverse(e, visitor)


def collect_and_assign_slices(exprs: Iterable[Expr], *, start_index: int = 0):
    # 1) collect all State/Control nodes
    states, controls = [], []

    def visitor(node):
        # Cannot simply use `if node not in states` check since we have overloaded `__eq__` operator
        if isinstance(node, State) and not any(node is s for s in states):
            states.append(node)

        if isinstance(node, Control) and not any(node is c for c in controls):
            controls.append(node)

    for e in exprs:
        traverse(e, visitor)

    def assign(vars_list, start_index):
        # split into manual vs auto
        manual = [v for v in vars_list if v._slice is not None]
        auto = [v for v in vars_list if v._slice is None]

        if manual:
            # 1) shape‐match check
            for v in manual:
                dim = int(np.prod(v.shape))
                sl = v._slice
                if (sl.stop - sl.start) != dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} is length {sl.stop - sl.start}, "
                        f"but variable has shape {v.shape} (dim {dim})"
                    )
            # sort by the start of their slices
            manual.sort(key=lambda v: v._slice.start)
            # 2a) must start at 0
            if manual[0]._slice.start != start_index:
                raise ValueError("User-defined slices must start at index 0")
            # 2b) check contiguity & no overlaps
            cursor = start_index
            for v in manual:
                sl = v._slice
                dim = int(np.prod(v.shape))
                if sl.start != cursor or sl.stop != cursor + dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} must be contiguous and non-overlapping"
                    )
                cursor += dim
            offset = cursor
        else:
            offset = start_index

        # 3) auto-assign the rest
        for v in auto:
            dim = int(np.prod(v.shape))
            v._slice = slice(offset, offset + dim)
            offset += dim

    # run separately on states (x) and controls (u)
    assign(states, start_index)
    assign(controls, start_index)
