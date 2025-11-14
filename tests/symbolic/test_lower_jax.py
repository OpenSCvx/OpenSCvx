import jax.numpy as jnp
import numpy as np
import pytest

from openscvx.symbolic.expr import (
    CTCS,
    QDCM,
    SSM,
    SSMP,
    Add,
    Concat,
    Constant,
    Control,
    Diag,
    Div,
    Equality,
    Exp,
    Expr,
    Hstack,
    Huber,
    Inequality,
    Log,
    MatMul,
    Mul,
    Neg,
    Norm,
    Parameter,
    PositivePart,
    SmoothReLU,
    Square,
    Stack,
    State,
    Sub,
    Vstack,
)
from openscvx.symbolic.lower import lower, lower_to_jax
from openscvx.symbolic.lowerers.jax import JaxLowerer


class UnregisteredExpr(Expr):
    pass


def test_jaxlowerer_raises_when_no_visitor_registered():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    with pytest.raises(NotImplementedError) as excinfo:
        # this should internally call dispatch() and fail
        jl.lower(node)

    msg = str(excinfo.value)
    assert "JaxLowerer" in msg, "should mention the lowerer class name"
    assert "UnregisteredExpr" in msg, "should mention the Expr subclass name"


def test_top_level_lower_raises_for_unregistered_expr():
    jl = JaxLowerer()
    node = UnregisteredExpr()
    # our top-level lower() simply forwards to jl.lower(...)
    with pytest.raises(NotImplementedError):
        lower(node, jl)


def test_jax_lower_constant():
    const = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    jl = JaxLowerer()
    f = jl._visit_constant(const)
    out = f(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1, 2], [3, 4]]))


def test_jax_lower_state_without_slice_raises():
    s = State("s", (3,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl._visit_state(s)


def test_jax_lower_control_without_slice_raises():
    c = Control("c", (2,))
    jl = JaxLowerer()
    with pytest.raises(ValueError):
        jl._visit_control(c)


def test_jax_lower_state_with_slice():
    x = jnp.arange(10.0)
    s = State("s", (4,))
    s._slice = slice(2, 6)
    jl = JaxLowerer()
    f = jl._visit_state(s)
    out = f(x, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert jnp.allclose(out, x[2:6])


def test_jax_lower_control_with_slice():
    u = jnp.arange(8.0)
    c = Control("c", (3,))
    c._slice = slice(5, 8)
    jl = JaxLowerer()
    f = jl._visit_control(c)
    out = f(None, u, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, u[5:8])


def test_jax_lower_parameter_scalar():
    """Test Parameter node with scalar value."""
    param = Parameter("alpha", (), value=5.0)
    jl = JaxLowerer()
    f = jl._visit_parameter(param)
    parameters = dict(alpha=5.0)

    # Test with scalar parameter
    out = f(None, None, None, parameters)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == ()
    assert jnp.allclose(out, 5.0)

    parameters["alpha"] = -2.5

    # Test with different scalar value
    out = f(None, None, None, parameters)
    assert jnp.allclose(out, -2.5)


def test_jax_lower_parameter_vector():
    """Test Parameter node with vector value."""
    param = Parameter("weights", (3,), value=np.array([1.0, 2.0, 3.0]))
    jl = JaxLowerer()
    f = jl._visit_parameter(param)

    # Test with vector parameter
    weights_val = np.array([1.0, 2.0, 3.0])
    out = f(None, None, None, dict(weights=weights_val))
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (3,)
    assert jnp.allclose(out, weights_val)

    # Test with different vector value
    weights_val2 = np.array([-1.0, 0.5, 2.5])
    out = f(None, None, None, dict(weights=weights_val2))
    assert jnp.allclose(out, weights_val2)


def test_jax_lower_parameter_matrix():
    """Test Parameter node with matrix value."""
    param = Parameter("transform", (2, 3), value=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    jl = JaxLowerer()
    f = jl._visit_parameter(param)

    # Test with matrix parameter
    matrix_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = f(None, None, None, dict(transform=matrix_val))
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 3)
    assert jnp.allclose(out, matrix_val)


def test_parameter_in_arithmetic_expression():
    """Test Parameter nodes in arithmetic expressions with states."""
    x = jnp.array([1.0, 2.0, 3.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    gain = Parameter("gain", (), value=2.5)

    # Expression: gain * x
    expr = Mul(gain, state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, dict(gain=2.5))

    expected = 2.5 * x
    assert jnp.allclose(result, expected)


def test_parameter_with_lower_to_jax():
    """Test Parameter nodes with the top-level lower_to_jax function."""
    param = Parameter("threshold", (2,), value=np.array([1.5, 2.5]))

    fn = lower_to_jax(param)
    param_val = np.array([1.5, 2.5])
    result = fn(None, None, None, dict(threshold=param_val))

    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2,)
    assert jnp.allclose(result, param_val)


def test_parameter_in_double_integrator_dynamics():
    """Test Parameter nodes in a realistic double integrator dynamics expression."""
    x = jnp.array([1.0, 2.0, 0.5, -0.2])  # [pos_x, pos_y, vel_x, vel_y]
    u = jnp.array([0.8, 1.2])  # [acc_x, acc_y]

    # State and control
    state = State("x", (4,))
    state._slice = slice(0, 4)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Parameters for physical system
    mass = Parameter("m", (), value=2.0)
    gravity = Parameter("g", (), value=9.81)

    # Extract state components
    # pos = state[0:2]  # [pos_x, pos_y]
    vel = state[2:4]  # [vel_x, vel_y]

    # Dynamics: pos_dot = vel, vel_dot = u/m + [0, -g]
    pos_dot = vel
    gravity_vec = Concat(Constant(0.0), -gravity)
    vel_dot = control / mass + gravity_vec

    dynamics = Concat(pos_dot, vel_dot)

    fn = lower_to_jax(dynamics)

    # Test with realistic parameter values
    m_val = 2.0  # kg
    g_val = 9.81  # m/s^2
    parameter = dict(m=m_val, g=g_val)
    result = fn(x, u, None, parameter)

    # Expected: [vel_x, vel_y, acc_x/m, acc_y/m - g]
    expected = jnp.array(
        [
            0.5,  # vel_x
            -0.2,  # vel_y
            0.8 / 2.0,  # acc_x / m = 0.4
            1.2 / 2.0 - 9.81,  # acc_y / m - g = 0.6 - 9.81 = -9.21
        ]
    )

    assert jnp.allclose(result, expected)
    assert result.shape == (4,)


def test_parameter_dynamics_with_jit_and_vmap():
    """Test Parameter nodes in dynamics function with JAX JIT compilation and vmap."""
    import jax

    # Create double integrator dynamics with parameters
    state = State("x", (4,))
    state._slice = slice(0, 4)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    mass = Parameter("m", (), value=2.0)
    gravity = Parameter("g", (), value=9.81)

    # Dynamics: pos_dot = vel, vel_dot = u/m + [0, -g]
    pos_dot = state[2:4]  # velocity
    gravity_vec = Concat(Constant(0.0), -gravity)
    vel_dot = control / mass + gravity_vec
    dynamics = Concat(pos_dot, vel_dot)

    # Lower to JAX function
    dynamics_fn = lower_to_jax(dynamics)

    # Create function compatible with trajoptproblem.py calling convention
    def dynamics_with_node(x, u, node, m, g):
        """Dynamics function with node parameter (similar to trajoptproblem.py structure)."""
        parameter = dict(m=m, g=g)
        return dynamics_fn(x, u, node, parameter)

    # JIT compile the function
    dynamics_jit = jax.jit(dynamics_with_node)

    # Test single evaluation
    x = jnp.array([1.0, 2.0, 0.5, -0.2])
    u = jnp.array([0.8, 1.2])
    node = 0
    m_val = 2.0
    g_val = 9.81

    result_single = dynamics_jit(x, u, node, m_val, g_val)
    expected = jnp.array([0.5, -0.2, 0.4, -9.21])
    assert jnp.allclose(result_single, expected)

    # Test with vmap for multiple time steps (similar to trajoptproblem.py)
    N = 5
    x_batch = jnp.tile(x[None, :], (N, 1))  # (N, 4)
    u_batch = jnp.tile(u[None, :], (N, 1))  # (N, 2)
    node_batch = jnp.arange(N)  # (N,)

    # Create vmapped function with parameters as None (not vectorized)
    dynamics_vmap = jax.vmap(
        dynamics_with_node,
        in_axes=(0, 0, 0, None, None),  # vmap over x, u, node; keep m, g scalar
    )

    # JIT compile the vmapped function
    dynamics_vmap_jit = jax.jit(dynamics_vmap)

    # Test batch evaluation
    result_batch = dynamics_vmap_jit(x_batch, u_batch, node_batch, m_val, g_val)
    expected_batch = jnp.tile(expected[None, :], (N, 1))  # (N, 4)

    assert result_batch.shape == (N, 4)
    assert jnp.allclose(result_batch, expected_batch)

    # Test parameter updates work correctly after compilation
    m_val_new = 3.0
    result_new_mass = dynamics_vmap_jit(x_batch, u_batch, node_batch, m_val_new, g_val)

    # Expected with new mass: [0.5, -0.2, 0.8/3.0, 1.2/3.0 - 9.81]
    expected_new = jnp.array([0.5, -0.2, 0.8 / 3.0, 1.2 / 3.0 - 9.81])
    expected_new_batch = jnp.tile(expected_new[None, :], (N, 1))

    assert jnp.allclose(result_new_mass, expected_new_batch)


def test_jax_lower_add_and_mul_of_slices():
    x = jnp.arange(8.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr_add = Add(a, b)
    expr_mul = Mul(a, b)

    jl = JaxLowerer()
    f_res_add = jl._visit_add(expr_add)
    res_add = f_res_add(x, None, None, None)
    f_res_mul = jl._visit_mul(expr_mul)
    res_mul = f_res_mul(x, None, None, None)

    assert jnp.allclose(res_add, x[0:3] + x[3:6])
    assert jnp.allclose(res_mul, x[0:3] * x[3:6])


def test_jax_lower_sub_and_div_of_slices():
    x = jnp.arange(8.0)
    u = jnp.arange(8.0) * 3.0
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = Control("b", (3,))
    b._slice = slice(0, 3)
    c = Constant(2.0)
    expr_sub = Sub(a, b)
    expr_div = Div(a, c)

    jl = JaxLowerer()
    f_res_sub = jl._visit_sub(expr_sub)
    res_sub = f_res_sub(x, u, None, None)
    f_res_div = jl._visit_div(expr_div)
    res_div = f_res_div(x, u, None, None)

    assert jnp.allclose(res_sub, x[0:3] - u[0:3])
    assert jnp.allclose(res_div, x[0:3] / c.value)


def test_jax_lower_matmul_vector_matrix():
    # (2×2 matrix) @ (2-vector)
    M = Constant(np.array([[1.0, 0.0], [0.0, 2.0]]))
    v = Constant(np.array([3.0, 4.0]))
    expr = MatMul(M, v)

    jl = JaxLowerer()
    f = jl._visit_matmul(expr)
    out = f(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2,)
    assert jnp.allclose(out, jnp.array([3.0, 8.0]))


def test_jax_lower_neg_and_composite():
    x = jnp.arange(6.0)
    u = jnp.arange(6.0) * 3
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = Control("b", (2,))
    b._slice = slice(0, 2)
    c = Constant(np.array([1.0, 1.0]))

    # expr = -((a + b) * c)
    expr = Neg(Mul(Add(a, b), c))
    jl = JaxLowerer()
    f = jl._visit_neg(expr)
    out = f(x, u, None, None)

    expected = -((x[0:2] + u[0:2]) * jnp.array([1.0, 1.0]))
    assert jnp.allclose(out, expected)


def test_lower_to_jax_constant_produces_callable():
    c = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    fns = lower_to_jax([c])
    assert isinstance(fns, list) and len(fns) == 1
    fn = fns[0]
    out = fn(None, None, None, None)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_lower_to_jax_add_with_slices():
    x = jnp.arange(6.0)
    a = State("a", (3,))
    a._slice = slice(0, 3)
    b = State("b", (3,))
    b._slice = slice(3, 6)
    expr = Add(a, b)

    fn = lower_to_jax(expr)
    out = fn(x, None, None, None)
    expected = x[0:3] + x[3:6]
    assert jnp.allclose(out, expected)


def test_lower_to_jax_multiple_exprs_returns_in_order():
    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([1.0, 2.0, 3.0])
    # expr1: constant, expr2: identity of x
    c = Constant(np.array([1.0, 2.0, 3.0]))
    v = State("v", (3,))
    v._slice = slice(0, 3)
    exprs = [c, v]

    fns = lower_to_jax(exprs)
    assert len(fns) == 2

    f_const, f_x = fns
    assert jnp.allclose(f_const(x, None, None, None), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(f_x(x, u, None, None), x)


def test_equality_constraint_lowering():
    """Test that equality constraints are lowered to residual form (lhs - rhs)."""
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0, 1.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Constraint: x == 2*u (should become x - 2*u == 0)
    lhs = state
    rhs = Mul(Constant(2.0), control)
    constraint = Equality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl._visit_constraint(constraint)
    residual = fn(x, u, None, None)

    # Residual should be lhs - rhs = x - 2*u
    expected = x - 2.0 * u
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)


def test_inequality_constraint_lowering():
    """Test that inequality constraints are lowered to residual form (lhs - rhs)."""
    x = jnp.array([0.5, 1.5, 2.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 2.0 (should become x - 2.0 <= 0)
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs, rhs)

    jl = JaxLowerer()
    fn = jl._visit_constraint(constraint)  # Both use the same visitor
    residual = fn(x, None, None, None)

    # Residual should be lhs - rhs = x - 2.0
    expected = x - 2.0
    assert jnp.allclose(residual, expected)
    assert residual.shape == (3,)

    # Check that residual is negative when constraint is satisfied
    # and positive when violated
    assert residual[0] < 0  # 0.5 - 2.0 = -1.5 (satisfied)
    assert residual[1] < 0  # 1.5 - 2.0 = -0.5 (satisfied)
    assert residual[2] > 0  # 2.5 - 2.0 = 0.5 (violated)


def test_constraint_lowering_with_lower_to_jax():
    """Test constraint lowering through the top-level lower_to_jax function."""
    x = jnp.array([1.0, 3.0])
    u = jnp.array([0.5])

    pos = State("pos", (2,))
    pos._slice = slice(0, 2)
    vel = Control("vel", (1,))
    vel._slice = slice(0, 1)

    # Mixed constraint: pos[0] + 2*vel <= pos[1]
    # Rearranged: pos[0] + 2*vel - pos[1] <= 0
    lhs = Add(pos[0], Mul(Constant(2.0), vel))
    rhs = pos[1]
    constraint = Inequality(lhs, rhs)

    # Lower using the top-level function
    fn = lower_to_jax(constraint)
    residual = fn(x, u, None, None)

    # Expected: pos[0] + 2*vel - pos[1] = 1.0 + 2*0.5 - 3.0 = -1.0
    expected = 1.0 + 2.0 * 0.5 - 3.0
    assert jnp.allclose(residual, expected)
    assert residual < 0  # Constraint is satisfied


def test_concat_simple():
    x = jnp.arange(5.0)
    a = State("a", (2,))
    a._slice = slice(0, 2)
    b = State("b", (2,))
    b._slice = slice(2, 4)
    c = Constant(9.0)
    expr = Concat(a, b, c)

    fn = lower_to_jax(expr)
    out = fn(x, None, None, None)
    expected = jnp.concatenate([x[0:2], x[2:4], jnp.array([9.0])], axis=0)
    assert jnp.allclose(out, expected)
    assert out.shape == (5,)


def test_lower_to_jax_double_integrator():
    x = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0
    pos = State("pos", (3,))
    pos._slice = slice(0, 3)
    vel = State("vel", (3,))
    vel._slice = slice(3, 6)

    acc = Control("acc", (3,))
    acc._slice = slice(0, 3)

    pos_dot = vel
    vel_dot = acc / m + Constant(np.array([0.0, 0.0, g]))

    dynamics_expr = Concat(pos_dot, vel_dot)
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x, u, None, None)

    expected = jnp.concatenate([x[3:6], u / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


def test_hstack_constants():
    """Test Hstack with constant arrays."""
    arr1 = Constant(np.array([1.0, 2.0]))
    arr2 = Constant(np.array([3.0, 4.0, 5.0]))
    expr = Hstack([arr1, arr2])

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert jnp.allclose(result, expected)
    assert result.shape == (5,)


def test_hstack_states_and_controls():
    """Test Hstack with state and control variables."""
    x = jnp.array([10.0, 20.0, 30.0])
    u = jnp.array([40.0, 50.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Stack: [state[0:2], control, constant]
    const = Constant(np.array([60.0]))
    expr = Hstack([state[0:2], control, const])

    fn = lower_to_jax(expr)
    result = fn(x, u, None, None)

    expected = jnp.array([10.0, 20.0, 40.0, 50.0, 60.0])
    assert jnp.allclose(result, expected)
    assert result.shape == (5,)


def test_vstack_constants():
    """Test Vstack with constant arrays."""
    arr1 = Constant(np.array([[1.0, 2.0]]))  # (1, 2)
    arr2 = Constant(np.array([[3.0, 4.0], [5.0, 6.0]]))  # (2, 2)
    expr = Vstack([arr1, arr2])

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert jnp.allclose(result, expected)
    assert result.shape == (3, 2)


def test_vstack_vectors():
    """Test Vstack with vector arrays (promotes to 2D)."""
    x = jnp.array([10.0, 20.0, 30.0, 40.0])

    state = State("x", (4,))
    state._slice = slice(0, 4)

    # Split state into two parts and stack vertically
    part1 = state[0:2]  # [10, 20]
    part2 = state[2:4]  # [30, 40]
    expr = Vstack([part1, part2])

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # vstack promotes 1D arrays to 2D: [[10, 20], [30, 40]]
    expected = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    assert jnp.allclose(result, expected)
    assert result.shape == (2, 2)


def test_index_and_slice():
    # make a 4-vector state
    x = jnp.array([10.0, 20.0, 30.0, 40.0])
    s = State("s", (4,))
    s._slice = slice(0, 4)

    # index it and slice it
    expr_elem = s[2]
    expr_slice = s[1:3]

    # lower → callables
    fn_elem = lower_to_jax(expr_elem)
    fn_slice = lower_to_jax(expr_slice)

    # check results
    out_elem = fn_elem(x, None, None, None)
    out_slice = fn_slice(x, None, None, None)

    assert out_elem.shape == () or out_elem.shape == ()  # scalar or 0-D
    assert out_elem == x[2]

    assert out_slice.shape == (2,)
    assert jnp.allclose(out_slice, x[1:3])


def test_lower_to_jax_double_integrator_indexed():
    # numeric inputs
    x_jax = jnp.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    u_jax = jnp.array([1.0, 1.0, 1.0])
    g = 9.81
    m = 1.0

    # one 6-vector state
    x = State("x", (6,))
    x._slice = slice(0, 6)

    # 3-vector control
    u = Control("u", (3,))
    u._slice = slice(0, 3)

    pos_dot = x[3:6]
    vel_dot = u / m + Constant(np.array([0.0, 0.0, g]))
    dynamics_expr = Concat(pos_dot, vel_dot)

    # lower and execute
    fn = lower_to_jax(dynamics_expr)
    xdot = fn(x_jax, u_jax, None, None)

    # expected by hand
    expected = jnp.concatenate([x_jax[3:6], u_jax / m + jnp.array([0.0, 0.0, g])], axis=0)

    assert jnp.allclose(xdot, expected)
    assert xdot.shape == (6,)


# TODO: (norrisg) move to separate file
# Tests for lowering penalty function expressions to JAX
def test_positive_part_constant():
    """Test PositivePart with constant values."""
    # Test with mixed positive and negative values
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = PositivePart(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert jnp.allclose(result, expected)


def test_positive_part_state():
    """Test PositivePart with state variables."""
    x = jnp.array([-3.0, -1.0, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = PositivePart(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected)


def test_positive_part_expression():
    """Test PositivePart with a composite expression."""
    x = jnp.array([1.0, 2.0, 3.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)
    threshold = Constant(np.array([2.0, 2.0, 2.0]))

    # pos(x - 2)
    expr = PositivePart(Sub(state, threshold))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.array([0.0, 0.0, 1.0])  # max(x - 2, 0)
    assert jnp.allclose(result, expected)


def test_square_constant():
    """Test Square with constant values."""
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expr = Square(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = values**2
    assert jnp.allclose(result, expected)


def test_square_state():
    """Test Square with state variables."""
    x = jnp.array([-3.0, -1.5, 0.0, 1.5, 3.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = Square(state)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = x * x
    assert jnp.allclose(result, expected)


def test_squared_relu_pattern():
    """Test the squared ReLU pattern: (max(x, 0))^2."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Build squared ReLU: Square(PositivePart(x))
    expr = Square(PositivePart(state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Expected: [0, 0, 0, 1, 4]
    expected = jnp.maximum(x, 0.0) ** 2
    assert jnp.allclose(result, expected)


def test_huber_constant():
    """Test Huber penalty with constant values."""
    values = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    delta = 0.25

    expr = Huber(Constant(values), delta=delta)

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # Huber formula:
    # if |x| <= delta: 0.5 * x^2
    # else: delta * (|x| - 0.5 * delta)
    expected = jnp.where(
        jnp.abs(values) <= delta, 0.5 * values**2, delta * (jnp.abs(values) - 0.5 * delta)
    )
    assert jnp.allclose(result, expected)


def test_huber_state_various_deltas():
    """Test Huber penalty with different delta values."""
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Test with small delta
    expr_small = Huber(state, delta=0.1)
    fn_small = lower_to_jax(expr_small)
    result_small = fn_small(x, None, None, None)

    # Most values should be in the linear region
    expected_small = jnp.where(jnp.abs(x) <= 0.1, 0.5 * x**2, 0.1 * (jnp.abs(x) - 0.5 * 0.1))
    assert jnp.allclose(result_small, expected_small)

    # Test with large delta
    expr_large = Huber(state, delta=3.0)
    fn_large = lower_to_jax(expr_large)
    result_large = fn_large(x, None, None, None)

    # All values should be in the quadratic region
    expected_large = 0.5 * x**2  # Since all |x| <= 3.0
    assert jnp.allclose(result_large, expected_large)


def test_huber_with_positive_part():
    """Test Huber applied to positive part (common CTCS pattern)."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Huber(PositivePart(x))
    expr = Huber(PositivePart(state), delta=0.5)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # First apply positive part
    pos_x = jnp.maximum(x, 0.0)
    # Then apply Huber
    expected = jnp.where(jnp.abs(pos_x) <= 0.5, 0.5 * pos_x**2, 0.5 * (jnp.abs(pos_x) - 0.5 * 0.5))
    assert jnp.allclose(result, expected)


def test_smooth_relu_constant():
    """Test SmoothReLU with constant values."""
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = 1e-8

    expr = SmoothReLU(Constant(values), c=c)

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    # SmoothReLU: sqrt(max(x, 0)^2 + c^2) - c
    expected = jnp.sqrt(jnp.maximum(values, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_state():
    """Test SmoothReLU with state variables."""
    x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_smooth_relu_approaches_relu():
    """Test that SmoothReLU approaches ReLU as c → 0."""
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Very small c
    expr = SmoothReLU(state, c=1e-12)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Should be very close to max(x, 0)
    expected = jnp.maximum(x, 0.0)
    assert jnp.allclose(result, expected, atol=1e-10)


def test_smooth_relu_differentiability_at_zero():
    """Test that SmoothReLU is smooth at x=0."""
    # Test values around zero
    x = jnp.array([-0.01, -0.001, 0.0, 0.001, 0.01])
    c = 0.01

    state = State("s", (5,))
    state._slice = slice(0, 5)
    expr = SmoothReLU(state, c=c)

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # At x=0, should equal 0 (since sqrt(c^2) - c = 0)
    assert jnp.abs(result[2]) < 1e-10

    # Should be continuous and smooth
    expected = jnp.sqrt(jnp.maximum(x, 0.0) ** 2 + c**2) - c
    assert jnp.allclose(result, expected)


def test_penalty_in_constraint_expression():
    """Test penalty functions used in a constraint-like expression."""
    x = jnp.array([0.5, 1.5, 2.5])
    u = jnp.array([0.0, 0.0, 0.0])

    state = State("s", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 2.0, violation when x > 2.0
    # Penalty: Square(PositivePart(x - 2.0))
    limit = Constant(np.array([2.0, 2.0, 2.0]))
    violation = Sub(state, limit)
    penalty = Square(PositivePart(violation))

    fn = lower_to_jax(penalty)
    result = fn(x, u, None, None)

    # Expected: [0, 0, 0.25] since only x[2]=2.5 violates
    expected = jnp.array([0.0, 0.0, 0.25])
    assert jnp.allclose(result, expected)


def test_combined_penalties():
    """Test combining different penalty functions."""
    x = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])

    state = State("s", (5,))
    state._slice = slice(0, 5)

    # Different penalties for testing
    squared_relu = Square(PositivePart(state))
    huber_penalty = Huber(PositivePart(state), delta=0.5)
    smooth_relu = SmoothReLU(state, c=0.1)

    fn_sq = lower_to_jax(squared_relu)
    fn_hub = lower_to_jax(huber_penalty)
    fn_smooth = lower_to_jax(smooth_relu)

    result_sq = fn_sq(x, None, None, None)
    result_hub = fn_hub(x, None, None, None)
    result_smooth = fn_smooth(x, None, None, None)

    # Squared ReLU should be most aggressive for large violations
    assert result_sq[4] > result_hub[4]  # At x=2.0

    # Huber should be linear for large values
    assert jnp.allclose(result_hub[4], 0.5 * (2.0 - 0.5 * 0.5))

    # All should be zero for negative values
    assert jnp.allclose(result_sq[0], 0.0)
    assert jnp.allclose(result_hub[0], 0.0)
    assert jnp.allclose(result_smooth[0], 0.0, atol=1e-8)


def test_penalty_with_control():
    """Test penalty functions with control variables."""
    u = jnp.array([-1.0, 0.5, 2.0])

    control = Control("u", (3,))
    control._slice = slice(0, 3)

    # Control constraint: |u| <= 1.0
    # Penalty for upper bound: Square(PositivePart(u - 1.0))
    upper_limit = Constant(np.array([1.0, 1.0, 1.0]))
    upper_violation = Sub(control, upper_limit)
    penalty = Square(PositivePart(upper_violation))

    fn = lower_to_jax(penalty)
    result = fn(None, u, None, None)

    # Expected: [0, 0, 1] since only u[2]=2.0 violates
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(result, expected)


def test_ctcs_constraint_can_be_lowered_directly():
    """Test that CTCS constraints can now be lowered directly with node context."""
    x = jnp.array([1.0, 2.0, 3.0])

    # Create state variable
    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Create constraint: x <= 2.0
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs - rhs, Constant(np.array([0.0, 0.0, 0.0])))

    # Wrap in CTCS
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Should work without node context (always active)
    result = fn(x, None, None, None)

    # Expected: sum(max(x - 2, 0)^2) = sum([0, 0, 1]) = 1.0
    assert jnp.allclose(result, 1.0)
    assert result.shape == ()  # Should be scalar


def test_ctcs_penalty_expression_can_be_lowered():
    """Test that the penalty expression from CTCS can be lowered successfully."""

    x = jnp.array([0.5, 1.5, 2.5])

    # Create state variable
    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Create constraint: x <= 2.0
    lhs = state
    rhs = Constant(np.array([2.0, 2.0, 2.0]))
    constraint = Inequality(lhs - rhs, 0)

    # Wrap in CTCS
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    # Extract the penalty expression (this is what would happen during augmentation)
    penalty_expr = ctcs_constraint.penalty_expr()

    # The penalty expression should be lowerable
    jl = JaxLowerer()
    fn = jl.lower(penalty_expr)

    # Execute the penalty function
    result = fn(x, None, None, None)

    # Expected: Sum(Square(PositivePart(x - 2.0))) = sum([0, 0, 0.25]) = 0.25
    # Only x[2] = 2.5 violates the constraint x <= 2.0
    expected = 0.25  # Scalar result from Sum
    assert jnp.allclose(result, expected)
    assert result.shape == ()  # Should be scalar


def test_ctcs_with_different_penalties():
    """Test that CTCS penalty expressions work with different penalty types."""

    x = jnp.array([1.0, 2.0, 3.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)

    # Constraint: x <= 1.5 (violations at x[1] and x[2])
    constraint = Inequality(state - Constant(np.array([1.5, 1.5, 1.5])), np.array([0, 0, 0]))

    # Test different penalty types
    penalties = ["squared_relu", "huber", "smooth_relu"]

    jl = JaxLowerer()

    for penalty_type in penalties:
        ctcs_constraint = CTCS(constraint, penalty=penalty_type)
        penalty_expr = ctcs_constraint.penalty_expr()

        # Should be able to lower without error
        fn = jl.lower(penalty_expr)
        result = fn(x, None, None, None)
        # Result should be a scalar (sum of all penalties)
        assert result.shape == ()  # Should be scalar
        # Total penalty should be positive since there are violations
        assert result > 0
        if penalty_type == "squared_relu":
            # Expected: 0^2 + 0.5^2 + 1.5^2 = 0 + 0.25 + 2.25 = 2.5
            expected = 0.25 + 2.25
            assert jnp.allclose(result, expected, rtol=1e-5)


def test_ctcs_with_node_range():
    """Test that CTCS constraints respect node ranges."""
    x = jnp.array([3.0])  # Violates constraint x <= 2.0

    state = State("x", (1,))
    state._slice = slice(0, 1)

    # Constraint: x <= 2.0
    constraint = Inequality(state - Constant(np.array([2.0])), Constant(np.array([0.0])))

    # CTCS active only between nodes 5-10
    ctcs_constraint = CTCS(constraint, penalty="squared_relu", nodes=(5, 10))

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Test at different nodes
    result_node_3 = fn(x, None, 3, None)  # Before active range
    result_node_7 = fn(x, None, 7, None)  # Within active range
    result_node_12 = fn(x, None, 12, None)  # After active range

    # Should be zero outside active range
    assert jnp.allclose(result_node_3, 0.0)
    assert jnp.allclose(result_node_12, 0.0)

    # Should have penalty within active range
    # Expected: sum(max(3 - 2, 0)^2) = 1.0
    assert jnp.allclose(result_node_7, 1.0)


def test_ctcs_without_node_range_always_active():
    """Test that CTCS constraints without node range are always active."""
    x = jnp.array([2.5])  # Violates constraint x <= 2.0

    state = State("x", (1,))
    state._slice = slice(0, 1)

    # Constraint: x <= 2.0
    constraint = Inequality(state - Constant(np.array([2.0])), Constant(np.array([0.0])))

    # CTCS without node range (always active)
    ctcs_constraint = CTCS(constraint, penalty="squared_relu")

    jl = JaxLowerer()
    fn = jl.lower(ctcs_constraint)

    # Test at different nodes - should always be active
    result_node_0 = fn(x, None, 0, None)
    result_node_50 = fn(x, None, 50, None)
    result_node_100 = fn(x, None, 100, None)

    # Should have same penalty at all nodes
    # Expected: sum(max(2.5 - 2, 0)^2) = 0.25
    expected = 0.25
    assert jnp.allclose(result_node_0, expected)
    assert jnp.allclose(result_node_50, expected)
    assert jnp.allclose(result_node_100, expected)


def test_ctcs_with_extra_kwargs():
    """Test that kwargs flow through all expression types."""
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    # Create a complex expression involving multiple nodes
    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Complex expression: (x[0] + x[1]) * u[0] + x[2] / u[1] - 5.0
    expr = Sub(
        Add(Mul(Add(state[0], state[1]), control[0]), Div(state[2], control[1])), Constant(5.0)
    )

    jl = JaxLowerer()
    fn = jl.lower(expr)

    result = fn(x, u, node=10, params=None)

    # Expected: (1 + 2) * 0.5 + 3 / 1.0 - 5.0 = 1.5 + 3.0 - 5.0 = -0.5
    expected = -0.5
    assert jnp.allclose(result, expected)


def test_normalized_constants_lower_correctly():
    """Test that normalized constants work correctly with JAX lowering"""

    jl = JaxLowerer()

    # Test scalar constant that was squeezed from higher dimensions
    scalar_squeezed = Constant(np.array([[5.0]]))  # (1,1) -> () after squeeze
    assert scalar_squeezed.value.shape == ()  # Verify normalization happened

    fn_scalar = jl._visit_constant(scalar_squeezed)
    result_scalar = fn_scalar(None, None, None, None)

    assert isinstance(result_scalar, jnp.ndarray)
    assert result_scalar.shape == ()
    assert jnp.allclose(result_scalar, 5.0)

    # Test vector constant that was squeezed
    vector_squeezed = Constant(np.array([[1.0, 2.0, 3.0]]))  # (1,3) -> (3,) after squeeze
    assert vector_squeezed.value.shape == (3,)  # Verify normalization happened

    fn_vector = jl._visit_constant(vector_squeezed)
    result_vector = fn_vector(None, None, None, None)

    assert isinstance(result_vector, jnp.ndarray)
    assert result_vector.shape == (3,)
    assert jnp.allclose(result_vector, jnp.array([1.0, 2.0, 3.0]))

    # Test matrix that had singleton dimensions removed
    matrix_squeezed = Constant(
        np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    )  # (1,2,1,2) -> (2,2) after squeeze
    assert matrix_squeezed.value.shape == (2, 2)  # Verify normalization happened

    fn_matrix = jl._visit_constant(matrix_squeezed)
    result_matrix = fn_matrix(None, None, None, None)

    assert isinstance(result_matrix, jnp.ndarray)
    assert result_matrix.shape == (2, 2)
    assert jnp.allclose(result_matrix, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_normalized_constants_in_complex_expressions():
    """Test that normalized constants work correctly in complex expressions that get lowered"""

    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Use constants that were created with extra dimensions and got squeezed
    scalar_const = Constant(np.array([[2.0]]))  # (1,1) -> () after squeeze
    vector_const = Constant(np.array([[1.0, 1.0, 1.0]]))  # (1,3) -> (3,) after squeeze

    # Verify normalization happened
    assert scalar_const.value.shape == ()
    assert vector_const.value.shape == (3,)

    # Create expression: (x + vector_const) * scalar_const
    expr = Mul(Add(state, vector_const), scalar_const)

    jl = JaxLowerer()
    fn = jl.lower(expr)
    result = fn(x, u, None, None)

    # Expected: ([1,2,3] + [1,1,1]) * 2 = [2,3,4] * 2 = [4,6,8]
    expected = jnp.array([4.0, 6.0, 8.0])
    assert jnp.allclose(result, expected)


def test_normalized_constants_preserve_dtype_in_lowering():
    """Test that JAX lowering preserves dtypes from normalized constants"""

    jl = JaxLowerer()

    # Test different dtypes with normalization
    int32_const = Constant(np.array([[42]], dtype=np.int32))  # (1,1) -> (), dtype preserved
    float32_const = Constant(np.array([[3.14]], dtype=np.float32))  # (1,1) -> (), dtype preserved

    # Verify normalization and dtype preservation
    assert int32_const.value.shape == ()
    assert float32_const.value.shape == ()
    assert int32_const.value.dtype == np.int32
    assert float32_const.value.dtype == np.float32

    # Test lowering
    fn_int = jl._visit_constant(int32_const)
    fn_float = jl._visit_constant(float32_const)

    result_int = fn_int(None, None, None, None)
    result_float = fn_float(None, None, None, None)

    # JAX should preserve the dtypes
    assert result_int.dtype == jnp.int32
    assert result_float.dtype == jnp.float32
    assert result_int == 42
    assert jnp.allclose(result_float, 3.14)


# TODO: (norrisg) move these to separate file
# Tests for 6DOF rigid body dynamics from drone racing example
def reference_6dof_dynamics_jax(x_val, u_val):
    """Reference implementation of 6DOF rigid body dynamics in pure JAX."""
    from openscvx.utils import SSM, SSMP, qdcm

    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Extract components
    # r = x_val[0:3]  # position
    v = x_val[3:6]  # velocity
    q = x_val[6:10]  # quaternion
    w = x_val[10:13]  # angular velocity
    # t = x_val[13]  # time

    f = u_val[:3]  # forces
    tau = u_val[3:]  # torques

    # Normalize quaternion
    q_norm = jnp.linalg.norm(q)
    q_normalized = q / q_norm

    # Compute dynamics
    r_dot = v
    v_dot = (1.0 / m) * qdcm(q_normalized) @ f + jnp.array([0, 0, g_const])
    q_dot = 0.5 * SSMP(w) @ q_normalized
    w_dot = jnp.diag(1.0 / J_b) @ (tau - SSM(w) @ jnp.diag(J_b) @ w)
    t_dot = 1.0

    return jnp.concatenate([r_dot, v_dot, q_dot, w_dot, jnp.array([t_dot])])


def test_6dof_rigid_body_dynamics_symbolic():
    """Test the fully symbolic 6DOF rigid body dynamics against reference JAX implementation."""

    # Define symbolic utility functions (from drone racing example)
    def symbolic_qdcm(q_normalized):
        """Quaternion to Direction Cosine Matrix conversion using symbolic expressions"""
        # Assume q is already normalized
        w, x, y, z = q_normalized[0], q_normalized[1], q_normalized[2], q_normalized[3]

        # Create DCM elements
        r11 = Constant(1.0) - Constant(2.0) * (y * y + z * z)
        r12 = Constant(2.0) * (x * y - z * w)
        r13 = Constant(2.0) * (x * z + y * w)

        r21 = Constant(2.0) * (x * y + z * w)
        r22 = Constant(1.0) - Constant(2.0) * (x * x + z * z)
        r23 = Constant(2.0) * (y * z - x * w)

        r31 = Constant(2.0) * (x * z - y * w)
        r32 = Constant(2.0) * (y * z + x * w)
        r33 = Constant(1.0) - Constant(2.0) * (x * x + y * y)

        # Stack into 3x3 matrix
        row1 = Concat(r11, r12, r13)
        row2 = Concat(r21, r22, r23)
        row3 = Concat(r31, r32, r33)

        return Stack([row1, row2, row3])

    def symbolic_ssmp(w):
        """Angular rate to 4x4 skew symmetric matrix for quaternion dynamics"""
        x, y, z = w[0], w[1], w[2]
        zero = Constant(0.0)

        # Create SSMP matrix
        row1 = Concat(zero, -x, -y, -z)
        row2 = Concat(x, zero, z, -y)
        row3 = Concat(y, -z, zero, x)
        row4 = Concat(z, y, -x, zero)

        return Stack([row1, row2, row3, row4])

    def symbolic_ssm(w):
        """Angular rate to 3x3 skew symmetric matrix"""
        x, y, z = w[0], w[1], w[2]
        zero = Constant(0.0)

        # Create SSM matrix
        row1 = Concat(zero, -z, y)
        row2 = Concat(z, zero, -x)
        row3 = Concat(-y, x, zero)

        return Stack([row1, row2, row3])

    def symbolic_diag(v):
        """Create diagonal matrix from vector"""
        if len(v) == 3:
            zero = Constant(0.0)
            row1 = Concat(v[0], zero, zero)
            row2 = Concat(zero, v[1], zero)
            row3 = Concat(zero, zero, v[2])
            return Stack([row1, row2, row3])
        else:
            raise NotImplementedError("Only 3x3 diagonal matrices supported")

    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Test multiple state/control combinations
    test_cases = [
        # Test case 1: Basic case
        (
            jnp.array(
                [10.0, 0.0, 20.0, 0.5, 0.2, -0.1, 1.0, 0.1, 0.05, 0.02, 0.1, 0.05, -0.02, 15.0]
            ),
            jnp.array([0.0, 0.0, 10.0, 0.1, -0.05, 0.02]),
        ),
        # Test case 2: Different orientation
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        ),
        # Test case 3: With angular velocity
        (
            jnp.array([5.0, -2.0, 10.0, 1.0, 0.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.2, -0.1, 0.3, 10.0]),
            jnp.array([1.0, 1.0, 8.0, 0.05, 0.1, -0.05]),
        ),
    ]

    for x_val, u_val in test_cases:
        # Normalize quaternion in test case
        q_norm = jnp.linalg.norm(x_val[6:10])
        x_val = x_val.at[6:10].set(x_val[6:10] / q_norm)

        # Create symbolic state and control
        x = State("x", (14,))
        x._slice = slice(0, 14)
        u = Control("u", (6,))
        u._slice = slice(0, 6)

        # Extract components
        v = x[3:6]
        q = x[6:10]
        q_norm = Norm(q)
        q_normalized = q / q_norm
        w = x[10:13]

        f = u[:3]
        tau = u[3:]

        # Create symbolic dynamics
        r_dot = v
        v_dot = (1.0 / m) * symbolic_qdcm(q_normalized) @ f + Constant(
            np.array([0, 0, g_const], dtype=np.float64)
        )
        q_dot = 0.5 * symbolic_ssmp(w) @ q_normalized
        J_b_inv = 1.0 / J_b
        J_b_diag = symbolic_diag([J_b[0], J_b[1], J_b[2]])
        w_dot = symbolic_diag([J_b_inv[0], J_b_inv[1], J_b_inv[2]]) @ (
            tau - symbolic_ssm(w) @ J_b_diag @ w
        )
        t_dot = 1.0

        dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)

        # Lower to JAX and test
        fn = lower_to_jax(dyn_expr)
        symbolic_result = fn(x_val, u_val, None, None)

        # Compare against reference implementation
        reference_result = reference_6dof_dynamics_jax(x_val, u_val)

        # Should be identical since both lower to the same JAX operations
        # assert jnp.allclose(symbolic_result, reference_result, rtol=1e-12, atol=1e-14)
        # TODO: (norrisg) figure out why it is not closer
        assert jnp.allclose(symbolic_result, reference_result, rtol=1e-6, atol=1e-12)


def test_6dof_rigid_body_dynamics_compact():
    """Test the compact node 6DOF rigid body dynamics against reference JAX implementation."""
    # Test parameters (from drone racing example)
    m = 1.0
    g_const = -9.18
    J_b = jnp.array([1.0, 1.0, 1.0])

    # Test multiple state/control combinations (same as symbolic test)
    test_cases = [
        # Test case 1: Basic case
        (
            jnp.array(
                [10.0, 0.0, 20.0, 0.5, 0.2, -0.1, 1.0, 0.1, 0.05, 0.02, 0.1, 0.05, -0.02, 15.0]
            ),
            jnp.array([0.0, 0.0, 10.0, 0.1, -0.05, 0.02]),
        ),
        # Test case 2: Different orientation
        (
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        ),
        # Test case 3: With angular velocity
        (
            jnp.array([5.0, -2.0, 10.0, 1.0, 0.5, -0.2, 0.5, 0.5, 0.5, 0.5, 0.2, -0.1, 0.3, 10.0]),
            jnp.array([1.0, 1.0, 8.0, 0.05, 0.1, -0.05]),
        ),
    ]

    for x_val, u_val in test_cases:
        # Normalize quaternion in test case
        q_norm = jnp.linalg.norm(x_val[6:10])
        x_val = x_val.at[6:10].set(x_val[6:10] / q_norm)

        # Create symbolic state and control
        x = State("x", (14,))
        x._slice = slice(0, 14)
        u = Control("u", (6,))
        u._slice = slice(0, 6)

        # Extract components
        v = x[3:6]
        q = x[6:10]
        q_norm = Norm(q)
        q_normalized = q / q_norm
        w = x[10:13]

        f = u[:3]
        tau = u[3:]

        # Create compact node dynamics (from drone racing example)
        r_dot = v
        v_dot = (1.0 / m) * QDCM(q_normalized) @ f + Constant(
            np.array([0, 0, g_const], dtype=np.float64)
        )
        q_dot = 0.5 * SSMP(w) @ q_normalized
        J_b_inv = 1.0 / J_b
        J_b_diag = Diag(J_b)
        w_dot = Diag(J_b_inv) @ (tau - SSM(w) @ J_b_diag @ w)
        t_dot = 1.0

        dyn_expr = Concat(r_dot, v_dot, q_dot, w_dot, t_dot)

        # Lower to JAX and test
        fn = lower_to_jax(dyn_expr)
        compact_result = fn(x_val, u_val, None, None)

        # Compare against reference implementation
        reference_result = reference_6dof_dynamics_jax(x_val, u_val)

        # Should be identical since both lower to the same JAX operations
        assert jnp.allclose(compact_result, reference_result, rtol=1e-12, atol=1e-14)


def test_qdcm():
    """Test the QDCM compact node individually."""
    # Test with a few different quaternions
    test_quaternions = [
        jnp.array([1.0, 0.0, 0.0, 0.0]),  # Identity rotation
        jnp.array([0.707, 0.707, 0.0, 0.0]),  # 90° rotation around x-axis
        jnp.array([0.5, 0.5, 0.5, 0.5]),  # 120° rotation around (1,1,1) axis
    ]

    for q_val in test_quaternions:
        # Normalize quaternion
        q_val = q_val / jnp.linalg.norm(q_val)

        # Create quaternion state
        q = State("q", (4,))
        q._slice = slice(0, 4)

        # Test QDCM node
        qdcm_expr = QDCM(q)
        fn = lower_to_jax(qdcm_expr)
        result = fn(q_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be orthogonal (R.T @ R = I)
        identity_check = result.T @ result
        assert jnp.allclose(identity_check, jnp.eye(3), atol=1e-10)

        # Should have determinant 1 (proper rotation)
        det = jnp.linalg.det(result)
        assert jnp.allclose(det, 1.0, atol=1e-10)


def test_ssmp():
    """Test the SSMP compact node individually."""
    # Test with different angular velocities
    test_angular_velocities = [
        jnp.array([0.0, 0.0, 0.0]),  # Zero rotation
        jnp.array([1.0, 0.0, 0.0]),  # Rotation around x-axis
        jnp.array([0.1, 0.2, 0.3]),  # General rotation
    ]

    for w_val in test_angular_velocities:
        # Create angular velocity state
        w = State("w", (3,))
        w._slice = slice(0, 3)

        # Test SSMP node
        ssmp_expr = SSMP(w)
        fn = lower_to_jax(ssmp_expr)
        result = fn(w_val, None, None, None)

        # Should be 4x4 matrix
        assert result.shape == (4, 4)

        # Should be skew-symmetric in the 3x3 submatrix part
        # The structure should be:
        # [0, -wx, -wy, -wz]
        # [wx, 0, wz, -wy]
        # [wy, -wz, 0, wx]
        # [wz, wy, -wx, 0]
        expected = jnp.array(
            [
                [0.0, -w_val[0], -w_val[1], -w_val[2]],
                [w_val[0], 0.0, w_val[2], -w_val[1]],
                [w_val[1], -w_val[2], 0.0, w_val[0]],
                [w_val[2], w_val[1], -w_val[0], 0.0],
            ]
        )
        assert jnp.allclose(result, expected, atol=1e-12)


def test_ssm():
    """Test the SSM compact node individually."""
    # Test with different angular velocities
    test_angular_velocities = [
        jnp.array([0.0, 0.0, 0.0]),  # Zero rotation
        jnp.array([1.0, 0.0, 0.0]),  # Rotation around x-axis
        jnp.array([0.1, 0.2, 0.3]),  # General rotation
    ]

    for w_val in test_angular_velocities:
        # Create angular velocity state
        w = State("w", (3,))
        w._slice = slice(0, 3)

        # Test SSM node
        ssm_expr = SSM(w)
        fn = lower_to_jax(ssm_expr)
        result = fn(w_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be skew-symmetric
        assert jnp.allclose(result, -result.T, atol=1e-12)

        # Check specific structure
        # [0, -wz, wy]
        # [wz, 0, -wx]
        # [-wy, wx, 0]
        expected = jnp.array(
            [[0.0, -w_val[2], w_val[1]], [w_val[2], 0.0, -w_val[0]], [-w_val[1], w_val[0], 0.0]]
        )
        assert jnp.allclose(result, expected, atol=1e-12)


def test_diag():
    """Test the Diag compact node individually."""
    # Test with different vectors
    test_vectors = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([0.5, -1.0, 2.5]),
        jnp.array([0.0, 0.0, 0.0]),
    ]

    for v_val in test_vectors:
        # Create vector state
        v = State("v", (3,))
        v._slice = slice(0, 3)

        # Test Diag node
        diag_expr = Diag(v)
        fn = lower_to_jax(diag_expr)
        result = fn(v_val, None, None, None)

        # Should be 3x3 matrix
        assert result.shape == (3, 3)

        # Should be diagonal
        expected = jnp.diag(v_val)
        assert jnp.allclose(result, expected, atol=1e-12)

        # Off-diagonal elements should be zero
        off_diag_mask = ~jnp.eye(3, dtype=bool)
        assert jnp.allclose(result[off_diag_mask], 0.0, atol=1e-12)


def test_exp_constant():
    """Test Exp with constant values."""
    values = np.array([0.0, 1.0, -1.0, 2.0])
    expr = Exp(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.exp(values)
    assert jnp.allclose(result, expected)


def test_exp_state_and_control():
    """Test Exp with state and control variables in expression."""
    x = jnp.array([1.0, 0.0, -0.5])
    u = jnp.array([0.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (1,))
    control._slice = slice(0, 1)

    # Expression: exp(x[0] + u[0])
    expr = Exp(state[0] + control[0])

    fn = lower_to_jax(expr)
    result = fn(x, u, None, None)

    # Expected: exp(1.0 + 0.5) = exp(1.5)
    expected = jnp.exp(1.5)
    assert jnp.allclose(result, expected)


def test_log_constant():
    """Test Log with constant values."""
    values = np.array([1.0, np.e, 2.0, 0.5])
    expr = Log(Constant(values))

    fn = lower_to_jax(expr)
    result = fn(None, None, None, None)

    expected = jnp.log(values)
    assert jnp.allclose(result, expected)


def test_log_with_exp_identity():
    """Test that log(exp(x)) = x for reasonable values."""
    x = jnp.array([0.0, 1.0, -1.0, 2.0])

    state = State("x", (4,))
    state._slice = slice(0, 4)

    # Expression: log(exp(x))
    expr = Log(Exp(state))

    fn = lower_to_jax(expr)
    result = fn(x, None, None, None)

    # Should recover original values
    assert jnp.allclose(result, x, atol=1e-12)


def test_constant_vs_implicit_conversion_equivalence():
    """Test that expressions with explicit Constant() and implicit conversion via to_expr produce
    identical results.
    """
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5, 1.0])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (2,))
    control._slice = slice(0, 2)

    # Test various expression types with both explicit and implicit constants

    # 1. Arithmetic operations
    scalar_value = 2.5
    vector_value = np.array([1.0, 1.0, 1.0])

    # Explicit vs implicit multiplication
    expr_explicit_mul = Mul(state, Constant(scalar_value))
    expr_implicit_mul = state * scalar_value

    fn_explicit_mul = lower_to_jax(expr_explicit_mul)
    fn_implicit_mul = lower_to_jax(expr_implicit_mul)

    result_explicit_mul = fn_explicit_mul(x, u, None, None)
    result_implicit_mul = fn_implicit_mul(x, u, None, None)

    assert jnp.allclose(result_explicit_mul, result_implicit_mul)
    assert jnp.allclose(result_explicit_mul, x * scalar_value)

    # Explicit vs implicit addition with vector
    expr_explicit_add = Add(state, Constant(vector_value))
    expr_implicit_add = state + vector_value

    fn_explicit_add = lower_to_jax(expr_explicit_add)
    fn_implicit_add = lower_to_jax(expr_implicit_add)

    result_explicit_add = fn_explicit_add(x, u, None, None)
    result_implicit_add = fn_implicit_add(x, u, None, None)

    assert jnp.allclose(result_explicit_add, result_implicit_add)
    assert jnp.allclose(result_explicit_add, x + vector_value)

    # 2. Matrix operations
    matrix_value = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

    # Explicit vs implicit matrix multiplication
    expr_explicit_matmul = MatMul(Constant(matrix_value), state)
    expr_implicit_matmul = matrix_value @ state

    fn_explicit_matmul = lower_to_jax(expr_explicit_matmul)
    fn_implicit_matmul = lower_to_jax(expr_implicit_matmul)

    result_explicit_matmul = fn_explicit_matmul(x, u, None, None)
    result_implicit_matmul = fn_implicit_matmul(x, u, None, None)

    assert jnp.allclose(result_explicit_matmul, result_implicit_matmul)
    assert jnp.allclose(result_explicit_matmul, matrix_value @ x)

    # 3. Comparison operations (constraints)
    bounds_value = np.array([0.5, 1.5, 2.5])

    # Explicit vs implicit inequality
    constraint_explicit = Inequality(Constant(bounds_value), state)
    constraint_implicit = bounds_value <= state

    fn_constraint_explicit = lower_to_jax(constraint_explicit)
    fn_constraint_implicit = lower_to_jax(constraint_implicit)

    result_constraint_explicit = fn_constraint_explicit(x, u, None, None)
    result_constraint_implicit = fn_constraint_implicit(x, u, None, None)

    assert jnp.allclose(result_constraint_explicit, result_constraint_implicit)
    assert jnp.allclose(result_constraint_explicit, bounds_value - x)


def test_complex_expression_constant_equivalence():
    """Test equivalence in complex expressions that mix explicit and implicit constants."""
    x = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([0.5])

    state = State("x", (3,))
    state._slice = slice(0, 3)
    control = Control("u", (1,))
    control._slice = slice(0, 1)

    # Complex dynamics-like expression with both explicit and implicit constants
    m = 2.0
    g = np.array([0.0, 0.0, -9.81])
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Version 1: Fully explicit constants
    expr_explicit = Add(Div(MatMul(Constant(A), state), Constant(m)), Constant(g))

    # Version 2: Mixed explicit/implicit (what users might write)
    expr_mixed = (A @ state) / m + g

    # Version 3: All implicit (most natural)
    expr_implicit = A @ state / m + g

    # All should produce identical results
    fn_explicit = lower_to_jax(expr_explicit)
    fn_mixed = lower_to_jax(expr_mixed)
    fn_implicit = lower_to_jax(expr_implicit)

    result_explicit = fn_explicit(x, u, None, None)
    result_mixed = fn_mixed(x, u, None, None)
    result_implicit = fn_implicit(x, u, None, None)

    # All versions should be identical
    assert jnp.allclose(result_explicit, result_mixed)
    assert jnp.allclose(result_mixed, result_implicit)
    assert jnp.allclose(result_explicit, result_implicit)

    # And match the expected mathematical result
    expected = A @ x / m + g
    assert jnp.allclose(result_explicit, expected)
    assert jnp.allclose(result_mixed, expected)
    assert jnp.allclose(result_implicit, expected)
