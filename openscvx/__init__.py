import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Core symbolic expressions - flat namespace for most common functions
import openscvx.backend.expr.linalg as linalg
import openscvx.backend.expr.spatial as spatial
import openscvx.backend.expr.stl as stl
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

__all__ = [
    # Core base classes
    "Expr",
    "Leaf",
    "Parameter",
    "Variable",
    "State",
    "Control",
    # Basic arithmetic operations
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Neg",
    "Power",
    "Sum",
    # Array operations
    "Index",
    "Concat",
    "Constant",
    # Mathematical functions
    "Sin",
    "Cos",
    "Sqrt",
    "Exp",
    "Log",
    # Constraints
    "Constraint",
    "Equality",
    "Inequality",
    "NodalConstraint",
    "CTCS",
    "ctcs",
    # Common linear algebra
    "Transpose",
    "Norm",
    # Submodules
    "stl",
    "spatial",
    "linalg",
]
