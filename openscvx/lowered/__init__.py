"""Lowered problem dataclasses.

This module contains dataclasses representing the outputs of the lowering phase,
where symbolic expressions are converted to executable JAX and CVXPy code.

Classes:
    LoweredProblem: Container for all lowering outputs
    LoweredJaxConstraints: JAX-lowered non-convex constraints with gradients
    LoweredCvxpyConstraints: CVXPy-lowered convex constraints
    CVXPyVariables: CVXPy variables and parameters for the OCP
    ParameterDict: Dictionary that syncs parameters between JAX and CVXPy
    Dynamics: JAX-lowered dynamics functions (f, A, B)
"""

from openscvx.lowered.cvxpy_constraints import LoweredCvxpyConstraints
from openscvx.lowered.cvxpy_variables import CVXPyVariables
from openscvx.lowered.jax_constraints import (
    CrossNodeConstraintLowered,
    LoweredJaxConstraints,
    LoweredNodalConstraint,
)
from openscvx.lowered.parameters import ParameterDict
from openscvx.lowered.dynamics import Dynamics
from openscvx.lowered.problem import LoweredProblem

__all__ = [
    "LoweredProblem",
    "LoweredJaxConstraints",
    "LoweredCvxpyConstraints",
    "LoweredNodalConstraint",
    "CrossNodeConstraintLowered",
    "CVXPyVariables",
    "ParameterDict",
    "Dynamics",
]
