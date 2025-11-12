import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"

# Core symbolic expressions - flat namespace for most common functions
import openscvx.symbolic.expr.linalg as linalg
import openscvx.symbolic.expr.spatial as spatial
import openscvx.symbolic.expr.stl as stl
from openscvx.symbolic.expr import (
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

# Global time state reference for use in constraints
# This will be automatically linked to the auto-created time state
# when TrajOptProblem is initialized
_time_state_ref = State("time", shape=(1,))


def get_time_state():
    """Get the global time state reference for use in constraints.
    
    This State("time") object can be used in constraint expressions.
    It will automatically reference the time state that is created
    from the Time object passed to TrajOptProblem.
    
    Example:
        ```python
        import openscvx as ox
        time = ox.Time(initial=0.0, final=10.0, min=0.0, max=20.0)
        # Use ox.time in constraints
        constraint = ox.ctcs(ox.time[0] <= 5.0)
        ```
    """
    return _time_state_ref


# Make time accessible as ox.time
time = _time_state_ref

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
    "Exp",
    "Log",
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
    # Time state reference
    "time",
    "get_time_state",
]
