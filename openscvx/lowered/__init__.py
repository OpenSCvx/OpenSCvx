"""Lowered problem dataclasses.

This module contains dataclasses representing the outputs of the lowering phase,
where symbolic expressions are converted to executable JAX and CVXPy code.

Classes:
    LoweredProblem: Container for all lowering outputs
    LoweredJaxConstraints: JAX-lowered non-convex constraints with gradients
    LoweredCvxpyConstraints: CVXPy-lowered convex constraints
"""

from openscvx.lowered.cvxpy_constraints import LoweredCvxpyConstraints
from openscvx.lowered.jax_constraints import LoweredJaxConstraints
from openscvx.lowered.problem import LoweredProblem

__all__ = [
    "LoweredProblem",
    "LoweredJaxConstraints",
    "LoweredCvxpyConstraints",
]
