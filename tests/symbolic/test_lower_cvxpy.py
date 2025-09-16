import cvxpy as cp
import numpy as np
import pytest

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Cos,
    Div,
    Equality,
    Huber,
    Index,
    Inequality,
    MatMul,
    Mul,
    Neg,
    Norm,
    PositivePart,
    Sin,
    SmoothReLU,
    Square,
    Sub,
    Sum,
    ctcs,
)
from openscvx.backend.lowerers.cvxpy import CvxpyLowerer, lower_to_cvxpy
from openscvx.backend.state import State


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

    def test_state_variable(self):
        """Test lowering state variables"""
        # Create CVXPy variables
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        # Create symbolic state
        x = State("x", shape=(3,))

        # Lower to CVXPy
        result = lowerer.lower(x)
        assert result is x_cvx  # Should return the mapped variable

    def test_state_variable_with_slice(self):
        """Test state variables with slices"""
        x_cvx = cp.Variable((10, 6), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        # State with slice
        x = State("x", shape=(3,))
        x._slice = slice(0, 3)

        result = lowerer.lower(x)
        # Should return x_cvx with slice applied
        assert isinstance(result, cp.Expression)

    def test_control_variable(self):
        """Test lowering control variables"""
        u_cvx = cp.Variable((10, 2), name="u")
        variable_map = {"u": u_cvx}
        lowerer = CvxpyLowerer(variable_map)

        u = Control("u", shape=(2,))
        result = lowerer.lower(u)
        assert result is u_cvx

    def test_missing_state_variable_error(self):
        """Test error when state vector not in map"""
        lowerer = CvxpyLowerer({})
        x = State("missing", shape=(3,))

        with pytest.raises(ValueError, match="State vector 'x' not found"):
            lowerer.lower(x)

    def test_missing_control_variable_error(self):
        """Test error when control vector not in map"""
        lowerer = CvxpyLowerer({})
        u = Control("thrust", shape=(2,))

        with pytest.raises(ValueError, match="Control vector 'u' not found"):
            lowerer.lower(u)

    def test_add(self):
        """Test addition expressions"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(2.0))
        expr = Add(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_sub(self):
        """Test subtraction expressions"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(1.0))
        expr = Sub(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_mul(self):
        """Test multiplication expressions"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(2.0))
        expr = Mul(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_div(self):
        """Test division expressions"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(2.0))
        expr = Div(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_matmul(self):
        """Test matrix multiplication"""
        x_cvx = cp.Variable(3, name="x")  # Single vector, not time series
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        A = Constant(np.eye(3))
        expr = MatMul(A, x)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_neg(self):
        """Test negation"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Neg(x)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_sum(self):
        """Test sum operation"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Sum(x)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_norm_l2(self):
        """Test L2 norm operation"""
        x_cvx = cp.Variable(3, name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Norm(x, ord=2)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_norm_l1(self):
        """Test L1 norm operation"""
        x_cvx = cp.Variable(3, name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Norm(x, ord=1)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_norm_inf(self):
        """Test infinity norm operation"""
        x_cvx = cp.Variable(3, name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Norm(x, ord="inf")

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_norm_fro(self):
        """Test Frobenius norm operation"""
        x_cvx = cp.Variable((2, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(6,))  # Flattened 2x3 matrix
        expr = Norm(x, ord="fro")

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_index(self):
        """Test indexing"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Index(x, 0)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_concat(self):
        """Test concatenation"""
        x_cvx = cp.Variable(3, name="x")
        u_cvx = cp.Variable(2, name="u")
        variable_map = {"x": x_cvx, "u": u_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        u = Control("u", shape=(2,))
        expr = Concat(x, u)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_equality_constraint(self):
        """Test equality constraints"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(0.0))
        expr = Equality(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Constraint)

    def test_inequality_constraint(self):
        """Test inequality constraints"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        const = Constant(np.array(1.0))
        expr = Inequality(x, const)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Constraint)

    def test_positive_part(self):
        """Test positive part function"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = PositivePart(x)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_square(self):
        """Test square function"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Square(x)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_huber(self):
        """Test Huber loss function"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Huber(x, delta=0.5)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_smooth_relu(self):
        """Test smooth ReLU function"""
        x_cvx = cp.Variable(3, name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = SmoothReLU(x, c=1e-6)

        result = lowerer.lower(expr)
        assert isinstance(result, cp.Expression)

    def test_sin_not_implemented(self):
        """Test that Sin raises NotImplementedError"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Sin(x)

        with pytest.raises(NotImplementedError, match="Trigonometric functions like Sin"):
            lowerer.lower(expr)

    def test_cos_not_implemented(self):
        """Test that Cos raises NotImplementedError"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        expr = Cos(x)

        with pytest.raises(NotImplementedError, match="Trigonometric functions like Cos"):
            lowerer.lower(expr)

    def test_ctcs_not_implemented(self):
        """Test that CTCS raises NotImplementedError"""
        x_cvx = cp.Variable((10, 3), name="x")
        variable_map = {"x": x_cvx}
        lowerer = CvxpyLowerer(variable_map)

        x = State("x", shape=(3,))
        constraint = Inequality(x, Constant(np.array(1.0)))
        expr = ctcs(constraint)

        with pytest.raises(NotImplementedError, match="CTCS constraints are for continuous-time"):
            lowerer.lower(expr)

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
