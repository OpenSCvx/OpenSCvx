import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Core symbolic expressions - flat namespace for most common functions
from openscvx.backend.expr import (
    CTCS,
    # Basic arithmetic operations
    Add,
    Concat,
    Constant,
    # Constraints
    Constraint,
    Control,
    Cos,
    Div,
    Equality,
    Exp,
    # Core base classes
    Expr,
    # Array operations
    Index,
    Inequality,
    Leaf,
    Log,
    MatMul,
    Mul,
    Neg,
    NodalConstraint,
    Norm,
    Parameter,
    Power,
    # Mathematical functions
    Sin,
    Sqrt,
    State,
    Sub,
    Sum,
    # Common linear algebra (also available via openscvx.linalg)
    Transpose,
    Variable,
    ctcs,
)
from openscvx import stl
