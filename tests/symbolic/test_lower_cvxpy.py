"""Tests for CVXPY lowering infrastructure.

This module tests the CVXPY lowering infrastructure, including:
- Core lowering infrastructure (CvxpyLowerer class)
- Integration tests with multiple node types
- Variable mapping and registration
- Convenience functions

Node-specific lowering tests have been moved to their respective test files:
- test_parameters.py - Parameter lowering
- test_arithmetic.py - Arithmetic operation lowering
- test_array.py - Array operation lowering
- test_linalg.py - Linear algebra operation lowering
- test_math.py - Math function lowering
- test_constraint.py - Constraint lowering
- test_variable.py - Variable lowering
"""

import cvxpy as cp
import numpy as np
import pytest

from openscvx.symbolic.expr import (
    Add,
    Constant,
    Control,
    Index,
    Inequality,
    MatMul,
    Mul,
    Norm,
    Square,
    State,
    Sub,
)
from openscvx.symbolic.lowerers.cvxpy import CvxpyLowerer, lower_to_cvxpy


class TestCvxpyLowerer:
    def test_constant(self):
        """Test lowering constant values"""
        lowerer = CvxpyLowerer()

        # Scalar constant
        const_expr = Constant(np.array(5.0))
        result = lowerer.lower(const_expr)
        assert isinstance(result, cp.Constant)
        assert result.value == 5.0

        # Array constant
        const_expr = Constant(np.array([1, 2, 3]))
        result = lowerer.lower(const_expr)
        assert isinstance(result, cp.Constant)
        np.testing.assert_array_equal(result.value, [1, 2, 3])

    def test_complex_expression(self):
        """Test lowering a complex expression"""
        x_cvx = cp.Variable(3, name="x")
        u_cvx = cp.Variable(2, name="u")
        variable_map = {"x": x_cvx, "u": u_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        u = Control("u", shape=(2,))

        # Complex expression: (x + 2*u[0])^2 <= 5
        # Need to broadcast u[0] to match x shape
        u_broadcasted = Mul(Constant(np.array([2.0, 2.0, 2.0])), Index(u, 0))
        expr = Inequality(Square(Add(x, u_broadcasted)), Constant(np.array([5.0, 5.0, 5.0])))

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Constraint)

    def test_convenience_function(self):
        """Test the convenience function lower_to_cvxpy"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}

        x = State("x", shape=(3,))
        expr = Add(x, Constant(np.array(1.0)))

        result = lower_to_cvxpy(expr, variable_map)
        assert isinstance(result, cp.Expression)

    def test_register_variable(self):
        """Test registering variables after initialization"""
        lowerer = CvxpyLowerer()

        x_cvx = cp.Variable((10, 3), name="x")
        lowerer.register_variable("x", x_cvx)

        x = State("x", shape=(3,))
        result = lowerer.lower(x)
        assert result is x_cvx

    def test_empty_variable_map(self):
        """Test behavior with empty variable map"""
        lowerer = CvxpyLowerer()

        # Should work with constants
        const = Constant(np.array(5.0))
        result = lowerer.lower(const)
        assert isinstance(result, cp.Constant)

        # Should fail with variables
        x = State("x", shape=(3,))
        with pytest.raises(ValueError):
            lowerer.lower(x)

    def test_multiple_operations_chained(self):
        """Test chaining multiple operations"""
        x_cvx = cp.Variable(3, name="x")
        u_cvx = cp.Variable(2, name="u")
        variable_map = {"x": x_cvx, "u": u_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        u = Control("u", shape=(2,))

        # x + u[0] - 2 * x, with broadcasting for u[0]
        u_elem = Index(u, 0)  # This will be a scalar
        # Create a vector of u[0] repeated to match x shape
        u_broadcast = Mul(Constant(np.array([1.0, 1.0, 1.0])), u_elem)
        expr = Sub(Add(x, u_broadcast), Mul(Constant(np.array(2.0)), x))

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_standardized_variable_mapping(self):
        """Test the new standardized variable mapping approach using 'x' and 'u' keys"""
        # Single time step variables (like used in lower_convex_constraints)
        x_node = cp.Variable(6, name="x")  # State vector at a specific node
        u_node = cp.Variable(3, name="u")  # Control vector at a specific node
        variable_map = {"x": x_node, "u": u_node}
        lowerer = CvxpyLowerer(variable_map)

        # Create symbolic variables with slices (simulating preprocessing)
        position = State("position", shape=(3,))
        velocity = State("velocity", shape=(3,))
        thrust = Control("thrust", shape=(3,))

        # Assign slices as preprocessing would do
        position._slice = slice(0, 3)
        velocity._slice = slice(3, 6)
        thrust._slice = slice(0, 3)

        # Test that variables correctly map to their sliced portions
        pos_result = lowerer.lower(position)
        vel_result = lowerer.lower(velocity)
        thrust_result = lowerer.lower(thrust)

        # All should be CVXPy expressions
        assert isinstance(pos_result, cp.Expression)
        assert isinstance(vel_result, cp.Expression)
        assert isinstance(thrust_result, cp.Expression)

    def test_gate_constraint_example(self):
        """Test a gate constraint similar to the drone example"""
        # CVXPy variables for a single node (like in lower_convex_constraints)
        x_node = cp.Variable(3, name="x")  # 3D position at node k
        variable_map = {"x": x_node}
        lowerer = CvxpyLowerer(variable_map)

        # Create symbolic position variable
        position = State("position", shape=(3,))
        position._slice = slice(0, 3)

        # Gate constraint: ||A @ position - center||_inf <= 1
        A = Constant(np.eye(3))
        center = Constant(np.array([1.0, 2.0, 3.0]))
        gate_expr = Norm(MatMul(A, position) - center, ord="inf")
        constraint = Inequality(gate_expr, Constant(1.0))

        result = lowerer.lower(constraint)
        assert isinstance(result, cp.Constraint)
