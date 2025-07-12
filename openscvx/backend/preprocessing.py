from typing import Iterable

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import Expr, traverse
from openscvx.backend.state import State


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
