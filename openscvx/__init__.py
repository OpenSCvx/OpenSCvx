import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Core symbolic expressions - flat namespace for most common functions
import openscvx.symbolic.expr.linalg as linalg
import openscvx.symbolic.expr.spatial as spatial
import openscvx.symbolic.expr.stl as stl
from openscvx.symbolic.expr import (
    CTCS,
    Abs,
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
    LogSumExp,
    MatMul,
    Max,
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
from openscvx.time import Time
from openscvx.trajoptproblem import TrajOptProblem

__all__ = [
    # Main Trajectory Optimization Entrypoint
    "TrajOptProblem",
    # Time configuration
    "Time",
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
    "Abs",
    "Exp",
    "Log",
    "LogSumExp",
    "Max",
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
