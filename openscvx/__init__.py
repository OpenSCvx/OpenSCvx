import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Cache management
# Core symbolic expressions - flat namespace for most common functions
import openscvx.symbolic.expr.linalg as linalg
import openscvx.symbolic.expr.spatial as spatial
import openscvx.symbolic.expr.stl as stl
from openscvx.expert import ByofSpec
from openscvx.problem import Problem
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
    Fixed,
    Free,
    Index,
    Inequality,
    Leaf,
    Log,
    LogSumExp,
    MatMul,
    Max,
    Maximize,
    Minimize,
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
    Tan,
    Variable,
    ctcs,
)
from openscvx.symbolic.time import Time
from openscvx.utils.cache import clear_cache, get_cache_dir, get_cache_size

__all__ = [
    # Main Trajectory Optimization Entrypoint
    "Problem",
    # Cache management
    "get_cache_dir",
    "clear_cache",
    "get_cache_size",
    # Time configuration
    "Time",
    # Core base classes
    "Expr",
    "Leaf",
    "Parameter",
    "Variable",
    "State",
    "Control",
    # Boundary condition helpers
    "Free",
    "Fixed",
    "Minimize",
    "Maximize",
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
    "Tan",
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
    # Expert mode types
    "ByofSpec",
]
