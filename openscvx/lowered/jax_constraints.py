"""JAX-lowered constraint dataclass."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openscvx.constraints import CrossNodeConstraintLowered, LoweredNodalConstraint

if TYPE_CHECKING:
    from openscvx.symbolic.expr import CTCS


@dataclass
class LoweredJaxConstraints:
    """JAX-lowered non-convex constraints with gradient functions.

    Contains constraints that have been lowered to JAX callable functions
    with automatically computed gradients. These are used for linearization
    in the SCP (Sequential Convex Programming) loop.

    Attributes:
        nodal: List of LoweredNodalConstraint objects. Each has `func`,
            `grad_g_x`, `grad_g_u` callables and `nodes` list.
        cross_node: List of CrossNodeConstraintLowered objects. Each has
            `func`, `grad_g_X`, `grad_g_U` for trajectory-level constraints.
        ctcs: CTCS constraints (unchanged from input, not lowered here).
    """

    nodal: list[LoweredNodalConstraint] = field(default_factory=list)
    cross_node: list[CrossNodeConstraintLowered] = field(default_factory=list)
    ctcs: list["CTCS"] = field(default_factory=list)
