import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Core symbolic expressions - flat namespace for most common functions
import openscvx.backend.expr.linalg as linalg
import openscvx.backend.expr.spatial as spatial
import openscvx.backend.expr.stl as stl
from openscvx.backend.expr import (
    CTCS,
    Add,
    Concat,
    Constant,
    Constraint,
    Control,
    Cos,
    Div,
    Equality,
    Exp,
    Expr,
    Index,
    Inequality,
    Leaf,
    Log,
    MatMul,
    Mul,
    Neg,
    NodalConstraint,
    Parameter,
    Power,
    Sin,
    Sqrt,
    State,
    Sub,
    Sum,
    Variable,
    ctcs,
)
from openscvx.trajoptproblem import TrajOptProblem

__all__ = [
    # Main Trajectory Optimization Entrypoint
    "TrajOptProblem",
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
    # Submodules
    "stl",
    "spatial",
    "linalg",
]
