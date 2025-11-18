"""Symbolic expression package for trajectory optimization.

This package provides a comprehensive symbolic expression system for building
optimization problems in openscvx. It implements an Abstract Syntax Tree (AST)
framework that allows you to write optimization problems using natural mathematical
notation.

Example:
    Import the package through the main openscvx module::

        import openscvx as ox

        # Create symbolic variables
        x = ox.Variable("x", shape=(3,))
        u = ox.Control("u", shape=(2,))

        # Build expressions
        cost = ox.Norm(x - [1, 2, 3])**2 + 0.1 * ox.Norm(u)**2
        constraint = x[0] <= 5.0

Module Organization:
    The package is organized into the following modules:

    Core Expressions (expr.py):
        Base classes and utilities including `Expr`, `Leaf`, `Parameter`, `Constant`,
        and helper functions `to_expr` and `traverse`.

    Arithmetic Operations (arithmetic.py):
        Fundamental arithmetic operations including `Add`, `Sub`, `Mul`, `Div`,
        `MatMul`, `Neg`, and `Power`.

    Array Operations (array.py):
        Array manipulation operations including `Index`, `Concat`, `Stack`, `Hstack`,
        and `Vstack` for indexing, slicing, and combining arrays.

    Constraints (constraint.py):
        Constraint types including `Constraint`, `Equality`, `Inequality`,
        `NodalConstraint`, and `CTCS` (Continuous-Time Constraint Satisfaction).

    Optimization Variables (variable.py, state.py, control.py):
        `Variable` for general optimization variables, `State` for time-varying state
        in trajectory problems, and `Control` for control inputs.

    Mathematical Functions (math.py):
        Trigonometric functions (`Sin`, `Cos`), exponential functions (`Exp`, `Log`,
        `Sqrt`, `Square`), and nonlinear functions (`PositivePart`, `Huber`,
        `SmoothReLU`, `Max`).

    Linear Algebra (linalg.py):
        Matrix operations (`Transpose`, `Diag`) and reductions (`Sum`, `Norm`).

    Spatial Operations (spatial.py):
        6-DOF operations for aerospace and robotics including `QDCM` (Quaternion to
        Direction Cosine Matrix), `SSMP` (4×4 skew-symmetric matrix for quaternion
        dynamics), and `SSM` (3×3 skew-symmetric matrix for cross products).

    Constraint Specifications (constraint.py):
        `NodalConstraint` for enforcing constraints at discrete nodes and `CTCS` for
        continuous-time constraint satisfaction.

    Signal Temporal Logic (stl.py):
        `Or` for logical disjunction in task specifications.
"""

# Arithmetic operations
from .arithmetic import Add, Div, MatMul, Mul, Neg, Power, Sub

# Array operations
from .array import Concat, Hstack, Index, Stack, Vstack

# Specialized constraints
from .constraint import CTCS, Constraint, Equality, Inequality, NodalConstraint, ctcs

# Control
from .control import Control

# Core base classes and fundamental operations
from .expr import (
    Constant,
    Expr,
    Leaf,
    Parameter,
    to_expr,
    traverse,
)

# Linear algebra operations
from .linalg import Diag, Norm, Sum, Transpose

# Mathematical functions
from .math import (
    Abs,
    Cos,
    Exp,
    Huber,
    Log,
    LogSumExp,
    Max,
    PositivePart,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
)

# Spatial/3D operations
from .spatial import QDCM, SSM, SSMP

# State
from .state import BoundaryType, State

# STL operations
from .stl import Or

# Variable
from .variable import Variable

__all__ = [
    # Core base classes and fundamental operations
    "Expr",
    "Leaf",
    "Parameter",
    "to_expr",
    "traverse",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Neg",
    "Power",
    "Sum",
    "Index",
    "Concat",
    "Constant",
    "Constraint",
    "Equality",
    "Inequality",
    # Variable
    "Variable",
    # State
    "State",
    "BoundaryType",
    # Control
    "Control",
    # Mathematical functions
    "Sin",
    "Cos",
    "Sqrt",
    "Abs",
    "PositivePart",
    "Square",
    "Huber",
    "SmoothReLU",
    "Exp",
    "Log",
    "LogSumExp",
    "Max",
    # Linear algebra operations
    "Transpose",
    "Stack",
    "Hstack",
    "Vstack",
    "Norm",
    "Diag",
    # Spatial/3D operations
    "QDCM",
    "SSMP",
    "SSM",
    # Specialized constraints
    "NodalConstraint",
    "CTCS",
    "ctcs",
    # STL operations
    "Or",
]
