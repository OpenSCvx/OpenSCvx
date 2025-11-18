"""JAX backend for lowering symbolic expressions to executable functions.

This module implements the JAX lowering backend that converts symbolic expression
AST nodes into JAX functions with automatic differentiation support. The lowering
uses a visitor pattern where each expression type has a corresponding visitor method.

Architecture:
    The JAX lowerer follows a visitor pattern with centralized registration:

    1. **Visitor Registration**: The @visitor decorator registers handler functions
       for each expression type in the _JAX_VISITORS dictionary
    2. **Dispatch**: The dispatch() function looks up and calls the appropriate
       visitor based on the expression's type
    3. **Recursive Lowering**: Each visitor recursively lowers child expressions
       and composes JAX operations
    4. **Standardized Signature**: All lowered functions have signature
       (x, u, node, params) -> result for uniformity

Key Features:
    - **Automatic Differentiation**: Lowered functions can be differentiated using
      JAX's jacfwd/jacrev for computing Jacobians
    - **JIT Compilation**: All functions are JAX-traceable and JIT-compatible
    - **Functional Closures**: Each visitor returns a closure that captures
      necessary constants and child functions
    - **Broadcasting**: Supports NumPy-style broadcasting through jnp operations

Lowered Function Signature:
    All lowered functions have a uniform signature::

        f(x, u, node, params) -> result

    Where:

    - x: State vector (jnp.ndarray)
    - u: Control vector (jnp.ndarray)
    - node: Node index for time-varying behavior (scalar or array)
    - params: Dictionary of parameter values (dict[str, Any])
    - result: JAX array (scalar, vector, or matrix)

Example:
    Basic usage::

        from openscvx.symbolic.lowerers.jax import JaxLowerer
        import openscvx as ox

        # Create symbolic expression
        x = ox.State("x", shape=(3,))
        u = ox.Control("u", shape=(2,))
        expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2

        # Lower to JAX
        lowerer = JaxLowerer()
        f = lowerer.lower(expr)

        # Evaluate
        import jax.numpy as jnp
        x_val = jnp.array([1.0, 2.0, 3.0])
        u_val = jnp.array([0.5, 0.5])
        result = f(x_val, u_val, node=0, params={})

        # Differentiate
        from jax import jacfwd
        df_dx = jacfwd(f, argnums=0)
        gradient = df_dx(x_val, u_val, node=0, params={})

For Contributors:
    **Adding Support for New Expression Types**

    To add support for a new symbolic expression type to JAX lowering:

    1. **Define the visitor method** in JaxLowerer with the @visitor decorator::

        @visitor(MyNewExpr)
        def _visit_my_new_expr(self, node: MyNewExpr):
            # Lower child expressions recursively
            operand_fn = self.lower(node.operand)

            # Return a closure with signature (x, u, node, params) -> result
            return lambda x, u, node, params: jnp.my_operation(
                operand_fn(x, u, node, params)
            )

    2. **Key requirements**:
        - Use the @visitor(ExprType) decorator to register the handler
        - Method name should be _visit_<expr_name> (private, lowercase, snake_case)
        - Recursively lower all child expressions using self.lower()
        - Return a closure with signature (x, u, node, params) -> jax_array
        - Use jnp.* operations (not np.*) for JAX traceability
        - Ensure the result is JAX-differentiable (avoid Python control flow)

    3. **Example patterns**:
        - Unary operation: Lower operand, apply jnp function
        - Binary operation: Lower both operands, combine with jnp operation
        - N-ary operation: Lower all operands, reduce or combine them
        - Conditional logic: Use jax.lax.cond for branching (see _visit_ctcs)

    4. **Testing**: Ensure your visitor works with:
        - JAX JIT compilation: jax.jit(lowered_fn)
        - Automatic differentiation: jax.jacfwd(lowered_fn, argnums=0)
        - Vectorization: jax.vmap(lowered_fn)

See Also:
    - lower_to_jax(): Convenience wrapper in symbolic/lower.py
    - CVXPyLowerer: Alternative backend for convex constraints
    - dispatch(): Core dispatch function for visitor pattern
"""

from typing import Any, Callable, Dict, Type

import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.special import logsumexp

from openscvx.symbolic.expr import (
    CTCS,
    QDCM,
    SSM,
    SSMP,
    Abs,
    Add,
    Concat,
    Constant,
    Constraint,
    Cos,
    Diag,
    Div,
    Equality,
    Exp,
    Expr,
    Hstack,
    Huber,
    Index,
    Inequality,
    Log,
    LogSumExp,
    MatMul,
    Max,
    Mul,
    Neg,
    NodalConstraint,
    Norm,
    Or,
    Parameter,
    PositivePart,
    Power,
    Sin,
    SmoothReLU,
    Sqrt,
    Square,
    Stack,
    Sub,
    Sum,
    Transpose,
    Vstack,
)
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State

_JAX_VISITORS: Dict[Type[Expr], Callable] = {}
"""Registry mapping expression types to their visitor functions."""


def visitor(expr_cls: Type[Expr]):
    """Decorator to register a visitor function for an expression type.

    This decorator registers a visitor method to handle a specific expression
    type during JAX lowering. The decorated function is stored in _JAX_VISITORS
    and will be called by dispatch() when lowering that expression type.

    Args:
        expr_cls: The Expr subclass this visitor handles (e.g., Add, Mul, Norm)

    Returns:
        Decorator function that registers the visitor and returns it unchanged

    Example:
        Register a visitor function for the Add expression:

            @visitor(Add)
            def _visit_add(self, node: Add):
                # Lower addition to JAX
                ...

    Note:
        Multiple expression types can share a visitor by stacking decorators::

            @visitor(Equality)
            @visitor(Inequality)
            def _visit_constraint(self, node: Constraint):
                # Handle both equality and inequality
                ...
    """

    def register(fn: Callable[[Any, Expr], Callable]):
        _JAX_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(lowerer: Any, expr: Expr):
    """Dispatch an expression to its registered visitor function.

    Looks up the visitor function for the expression's type and calls it.
    This is the core of the visitor pattern implementation.

    Args:
        lowerer: The JaxLowerer instance (provides context for visitor methods)
        expr: The expression node to lower

    Returns:
        The result of calling the visitor function (typically a JAX callable)

    Raises:
        NotImplementedError: If no visitor is registered for the expression type

    Example:
        Dispatch an expression to lower it to a JAX function:

            lowerer = JaxLowerer()
            expr = Add(x, y)
            fn = dispatch(lowerer, expr)  # Calls visit_add
    """
    fn = _JAX_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(
            f"{lowerer.__class__.__name__!r} has no visitor for {type(expr).__name__}"
        )
    return fn(lowerer, expr)


class JaxLowerer:
    """JAX backend for lowering symbolic expressions to executable functions.

    This class implements the visitor pattern for converting symbolic expression
    AST nodes to JAX functions. Each expression type has a corresponding visitor
    method decorated with @visitor that handles the lowering logic.

    The lowering process is recursive: each visitor lowers its child expressions
    first, then composes them into a JAX operation. All lowered functions have
    a standardized signature (x, u, node, params) -> result.

    Attributes:
        None (stateless lowerer - all state is in the expression tree)

    Example:
        Set up the JaxLowerer and lower an expression to a JAX function:

            lowerer = JaxLowerer()
            expr = ox.Norm(x)**2 + 0.1 * ox.Norm(u)**2
            f = lowerer.lower(expr)
            result = f(x_val, u_val, node=0, params={})

    Note:
        The lowerer is stateless and can be reused for multiple expressions.
        All visitor methods are instance methods to maintain a clean interface,
        but they don't modify instance state.
    """

    def lower(self, expr: Expr):
        """Lower a symbolic expression to a JAX function.

        Main entry point for lowering. Delegates to dispatch() which looks up
        the appropriate visitor method based on the expression type.

        Args:
            expr: Symbolic expression to lower (any Expr subclass)

        Returns:
            JAX function with signature (x, u, node, params) -> result

        Raises:
            NotImplementedError: If no visitor exists for the expression type
            ValueError: If the expression is malformed (e.g., State without slice)

        Example:
            Lower an expression to a JAX function:

                lowerer = JaxLowerer()
                x = ox.State("x", shape=(3,))
                expr = ox.Norm(x)
                f = lowerer.lower(expr)
                # f is now callable
        """
        return dispatch(self, expr)

    @visitor(Constant)
    def _visit_constant(self, node: Constant):
        """Lower a constant value to a JAX function.

        Captures the constant value and returns a function that always returns it.
        Scalar constants are squeezed to ensure they're true scalars, not (1,) arrays.

        Args:
            node: Constant expression node

        Returns:
            Function (x, u, node, params) -> constant_value
        """
        # capture the constant value once
        value = jnp.array(node.value)
        # For scalar constants (single element arrays), squeeze to scalar
        # This prevents (1,) shapes in constraint residuals
        if value.size == 1:
            value = value.squeeze()
        return lambda x, u, node, params: value

    @visitor(State)
    def _visit_state(self, node: State):
        """Lower a state variable to a JAX function.

        Extracts the appropriate slice from the unified state vector x using
        the slice assigned during unification.

        Args:
            node: State expression node

        Returns:
            Function (x, u, node, params) -> x[slice]

        Raises:
            ValueError: If the state has no slice assigned (unification not run)
        """
        sl = node._slice
        if sl is None:
            raise ValueError(f"State {node.name!r} has no slice assigned")
        return lambda x, u, node, params: x[sl]

    @visitor(Control)
    def _visit_control(self, node: Control):
        """Lower a control variable to a JAX function.

        Extracts the appropriate slice from the unified control vector u using
        the slice assigned during unification.

        Args:
            node: Control expression node

        Returns:
            Function (x, u, node, params) -> u[slice]

        Raises:
            ValueError: If the control has no slice assigned (unification not run)
        """
        sl = node._slice
        if sl is None:
            raise ValueError(f"Control {node.name!r} has no slice assigned")
        return lambda x, u, node, params: u[sl]

    @visitor(Parameter)
    def _visit_parameter(self, node: Parameter):
        """Lower a parameter to a JAX function.

        Parameters are looked up by name in the params dictionary at evaluation time,
        allowing runtime parameter updates without recompilation.

        Args:
            node: Parameter expression node

        Returns:
            Function (x, u, node, params) -> params[name]
        """
        param_name = node.name
        return lambda x, u, node, params: jnp.array(params[param_name])

    @visitor(Add)
    def _visit_add(self, node: Add):
        """Lower addition to JAX function.

        Recursively lowers all terms and composes them with element-wise addition.
        Supports broadcasting following NumPy/JAX rules.

        Args:
            node: Add expression node with multiple terms

        Returns:
            Function (x, u, node, params) -> sum of all terms
        """
        fs = [self.lower(term) for term in node.terms]

        def fn(x, u, node, params):
            acc = fs[0](x, u, node, params)
            for f in fs[1:]:
                acc = acc + f(x, u, node, params)
            return acc

        return fn

    @visitor(Sub)
    def _visit_sub(self, node: Sub):
        """Lower subtraction to JAX function (element-wise left - right)."""
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, node, params: fL(x, u, node, params) - fR(x, u, node, params)

    @visitor(Mul)
    def _visit_mul(self, node: Mul):
        """Lower element-wise multiplication to JAX function (Hadamard product)."""
        fs = [self.lower(factor) for factor in node.factors]

        def fn(x, u, node, params):
            acc = fs[0](x, u, node, params)
            for f in fs[1:]:
                acc = acc * f(x, u, node, params)
            return acc

        return fn

    @visitor(Div)
    def _visit_div(self, node: Div):
        """Lower element-wise division to JAX function."""
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, node, params: fL(x, u, node, params) / fR(x, u, node, params)

    @visitor(MatMul)
    def _visit_matmul(self, node: MatMul):
        """Lower matrix multiplication to JAX function using jnp.matmul."""
        fL = self.lower(node.left)
        fR = self.lower(node.right)
        return lambda x, u, node, params: jnp.matmul(fL(x, u, node, params), fR(x, u, node, params))

    @visitor(Neg)
    def _visit_neg(self, node: Neg):
        """Lower negation (unary minus) to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: -fO(x, u, node, params)

    @visitor(Sum)
    def _visit_sum(self, node: Sum):
        """Lower sum reduction to JAX function (sums all elements)."""
        f = self.lower(node.operand)
        return lambda x, u, node, params: jnp.sum(f(x, u, node, params))

    @visitor(Norm)
    def _visit_norm(self, node: Norm):
        """Lower norm operation to JAX function.

        Converts symbolic norm to jnp.linalg.norm with appropriate ord parameter.
        Handles string ord values like "inf", "-inf", "fro".

        Args:
            node: Norm expression node with ord attribute

        Returns:
            Function (x, u, node, params) -> norm of operand
        """
        f = self.lower(node.operand)
        ord_val = node.ord

        # Convert string ord values to appropriate JAX values
        if ord_val == "inf":
            ord_val = jnp.inf
        elif ord_val == "-inf":
            ord_val = -jnp.inf
        elif ord_val == "fro":
            # For vectors, Frobenius norm is the same as 2-norm
            ord_val = None  # Default is 2-norm

        return lambda x, u, node, params: jnp.linalg.norm(f(x, u, node, params), ord=ord_val)

    @visitor(Index)
    def _visit_index(self, node: Index):
        """Lower indexing/slicing operation to JAX function."""
        # lower the "base" expr into a fn(x,u,node), then index it
        f_base = self.lower(node.base)
        idx = node.index
        return lambda x, u, node, params: jnp.atleast_1d(f_base(x, u, node, params))[idx]

    @visitor(Concat)
    def _visit_concat(self, node: Concat):
        """Lower concatenation to JAX function (concatenates along axis 0)."""
        # lower each child
        fn_list = [self.lower(child) for child in node.exprs]

        # wrapper that promotes scalars to 1-D and concatenates
        def concat_fn(x, u, node, params):
            parts = [jnp.atleast_1d(fn(x, u, node, params)) for fn in fn_list]
            return jnp.concatenate(parts, axis=0)

        return concat_fn

    @visitor(Sin)
    def _visit_sin(self, node: Sin):
        """Lower sine function to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: jnp.sin(fO(x, u, node, params))

    @visitor(Cos)
    def _visit_cos(self, node: Cos):
        """Lower cosine function to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: jnp.cos(fO(x, u, node, params))

    @visitor(Exp)
    def _visit_exp(self, node: Exp):
        """Lower exponential function to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: jnp.exp(fO(x, u, node, params))

    @visitor(Log)
    def _visit_log(self, node: Log):
        """Lower natural logarithm to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: jnp.log(fO(x, u, node, params))

    @visitor(Abs)
    def _visit_abs(self, node: Abs):
        """Lower absolute value to JAX function."""
        fO = self.lower(node.operand)
        return lambda x, u, node, params: jnp.abs(fO(x, u, node, params))

    @visitor(Equality)
    @visitor(Inequality)
    def _visit_constraint(self, node: Constraint):
        """Lower constraint to residual function.

        Both equality (lhs == rhs) and inequality (lhs <= rhs) constraints are
        lowered to their residual form: lhs - rhs. The constraint is satisfied
        when the residual equals zero (equality) or is non-positive (inequality).

        Args:
            node: Equality or Inequality constraint node

        Returns:
            Function (x, u, node, params) -> lhs - rhs (constraint residual)

        Note:
            The returned residual is used in penalty methods and Lagrangian terms.
            For equality: residual should be 0
            For inequality: residual should be <= 0
        """
        fL = self.lower(node.lhs)
        fR = self.lower(node.rhs)
        return lambda x, u, node, params: fL(x, u, node, params) - fR(x, u, node, params)

    # TODO: (norrisg) CTCS is playing 2 roles here: both as a constraint wrapper and as the penalty
    # expression w/ conditional logic. Consider adding conditional logic as separate AST nodes.
    # Then, CTCS remains a wrapper and we just wrap the penalty expression with the conditional
    # logic when we lower it.
    @visitor(CTCS)
    def _visit_ctcs(self, node: CTCS):
        """Lower CTCS (Continuous-Time Constraint Satisfaction) to JAX function.

        CTCS constraints use penalty methods to enforce constraints over continuous
        time intervals. The lowered function includes conditional logic to activate
        the penalty only within the specified node interval.

        Args:
            node: CTCS constraint node with penalty expression and optional node range

        Returns:
            Function (x, u, current_node, params) -> penalty value or 0

        Note:
            Uses jax.lax.cond for JAX-traceable conditional evaluation. The penalty
            is active only when current_node is in [start_node, end_node).
            If no node range is specified, the penalty is always active.

        See Also:
            - CTCS: The symbolic CTCS constraint class
            - penalty functions: PositivePart, Huber, SmoothReLU
        """
        # Lower the penalty expression (which includes the constraint residual)
        penalty_expr_fn = self.lower(node.penalty_expr())

        def ctcs_fn(x, u, current_node, params):
            # Check if constraint is active at this node
            if node.nodes is not None:
                start_node, end_node = node.nodes
                # Extract scalar value from current_node (which may be array or scalar)
                # Keep as JAX array for tracing compatibility
                node_scalar = jnp.atleast_1d(current_node)[0]
                is_active = (start_node <= node_scalar) & (node_scalar < end_node)

                # Use jax.lax.cond for conditional evaluation
                return cond(
                    is_active,
                    lambda _: penalty_expr_fn(x, u, current_node, params),
                    lambda _: 0.0,
                    operand=None,
                )
            else:
                # Always active if no node range specified
                return penalty_expr_fn(x, u, current_node, params)

        return ctcs_fn

    @visitor(PositivePart)
    def _visit_pos(self, node):
        """Lower positive part function to JAX.

        Computes max(x, 0), used in penalty methods for inequality constraints.

        Args:
            node: PositivePart expression node

        Returns:
            Function (x, u, node, params) -> max(operand, 0)
        """
        f = self.lower(node.x)
        return lambda x, u, node, params: jnp.maximum(f(x, u, node, params), 0.0)

    @visitor(Square)
    def _visit_square(self, node):
        """Lower square function to JAX.

        Computes x^2 element-wise. Used in quadratic penalty methods.

        Args:
            node: Square expression node

        Returns:
            Function (x, u, node, params) -> operand^2
        """
        f = self.lower(node.x)
        return lambda x, u, node, params: f(x, u, node, params) * f(x, u, node, params)

    @visitor(Huber)
    def _visit_huber(self, node):
        """Lower Huber penalty function to JAX.

        Huber penalty is quadratic for small values and linear for large values:
        - |x| <= delta: 0.5 * x^2
        - |x| > delta: delta * (|x| - 0.5 * delta)

        Args:
            node: Huber expression node with delta parameter

        Returns:
            Function (x, u, node, params) -> Huber penalty
        """
        f = self.lower(node.x)
        delta = node.delta
        return lambda x, u, node, params: jnp.where(
            jnp.abs(f(x, u, node, params)) <= delta,
            0.5 * f(x, u, node, params) ** 2,
            delta * (jnp.abs(f(x, u, node, params)) - 0.5 * delta),
        )

    @visitor(SmoothReLU)
    def _visit_srelu(self, node):
        """Lower smooth ReLU penalty function to JAX.

        Smooth approximation to ReLU: sqrt(max(x, 0)^2 + c^2) - c
        Differentiable everywhere, approaches ReLU as c -> 0.

        Args:
            node: SmoothReLU expression node with smoothing parameter c

        Returns:
            Function (x, u, node, params) -> smooth ReLU penalty
        """
        f = self.lower(node.x)
        c = node.c
        # smooth_relu(pos(x)) = sqrt(pos(x)^2 + c^2) - c ; here f already includes pos inside node
        return (
            lambda x, u, node, params: jnp.sqrt(jnp.maximum(f(x, u, node, params), 0.0) ** 2 + c**2)
            - c
        )

    @visitor(NodalConstraint)
    def _visit_nodal_constraint(self, node: NodalConstraint):
        """Lower a NodalConstraint by lowering its underlying constraint.

        NodalConstraint is a wrapper that specifies which nodes a constraint
        applies to. The lowering just unwraps and lowers the inner constraint.

        Args:
            node: NodalConstraint wrapper

        Returns:
            Function from lowering the wrapped constraint expression
        """
        return self.lower(node.constraint)

    @visitor(Sqrt)
    def _visit_sqrt(self, node: Sqrt):
        """Lower square root to JAX function."""
        f = self.lower(node.operand)
        return lambda x, u, node, params: jnp.sqrt(f(x, u, node, params))

    @visitor(Max)
    def _visit_max(self, node: Max):
        """Lower element-wise maximum to JAX function."""
        fs = [self.lower(op) for op in node.operands]

        def fn(x, u, node, params):
            values = [f(x, u, node, params) for f in fs]
            # jnp.maximum can take multiple arguments
            result = values[0]
            for val in values[1:]:
                result = jnp.maximum(result, val)
            return result

        return fn

    @visitor(LogSumExp)
    def _visit_logsumexp(self, node: LogSumExp):
        """Lower log-sum-exp to JAX function.

        Computes log(sum(exp(x_i))) for multiple operands, which is a smooth
        approximation to the maximum function. Uses JAX's numerically stable
        logsumexp implementation. Performs element-wise log-sum-exp with
        broadcasting support.
        """
        fs = [self.lower(op) for op in node.operands]

        def fn(x, u, node, params):
            values = [f(x, u, node, params) for f in fs]
            # Broadcast all values to the same shape, then stack along new axis
            # and compute logsumexp along that axis for element-wise operation
            broadcasted = jnp.broadcast_arrays(*values)
            stacked = jnp.stack(list(broadcasted), axis=0)
            return logsumexp(stacked, axis=0)

        return fn

    @visitor(Transpose)
    def _visit_transpose(self, node: Transpose):
        """Lower matrix transpose to JAX function."""
        f = self.lower(node.operand)
        return lambda x, u, node, params: jnp.transpose(f(x, u, node, params))

    @visitor(Power)
    def _visit_power(self, node: Power):
        """Lower element-wise power (base**exponent) to JAX function."""
        fB = self.lower(node.base)
        fE = self.lower(node.exponent)
        return lambda x, u, node, params: jnp.power(fB(x, u, node, params), fE(x, u, node, params))

    @visitor(Stack)
    def _visit_stack(self, node: Stack):
        """Lower vertical stacking to JAX function (stack along axis 0)."""
        row_fns = [self.lower(row) for row in node.rows]

        def stack_fn(x, u, node, params):
            rows = [jnp.atleast_1d(fn(x, u, node, params)) for fn in row_fns]
            return jnp.stack(rows, axis=0)

        return stack_fn

    @visitor(Hstack)
    def _visit_hstack(self, node: Hstack):
        """Lower horizontal stacking to JAX function."""
        array_fns = [self.lower(arr) for arr in node.arrays]

        def hstack_fn(x, u, node, params):
            arrays = [jnp.atleast_1d(fn(x, u, node, params)) for fn in array_fns]
            return jnp.hstack(arrays)

        return hstack_fn

    @visitor(Vstack)
    def _visit_vstack(self, node: Vstack):
        """Lower vertical stacking to JAX function."""
        array_fns = [self.lower(arr) for arr in node.arrays]

        def vstack_fn(x, u, node, params):
            arrays = [jnp.atleast_1d(fn(x, u, node, params)) for fn in array_fns]
            return jnp.vstack(arrays)

        return vstack_fn

    @visitor(QDCM)
    def _visit_qdcm(self, node: QDCM):
        """Lower quaternion to direction cosine matrix (DCM) conversion.

        Converts a unit quaternion [q0, q1, q2, q3] to a 3x3 rotation matrix.
        Used in 6-DOF spacecraft and robotics applications.

        The quaternion is normalized before conversion to ensure a valid rotation
        matrix. The DCM is computed using the standard quaternion-to-DCM formula.

        Args:
            node: QDCM expression node

        Returns:
            Function (x, u, node, params) -> 3x3 rotation matrix

        Note:
            Quaternion convention: [w, x, y, z] where w is the scalar part
        """
        f = self.lower(node.q)

        def qdcm_fn(x, u, node, params):
            q = f(x, u, node, params)
            # Normalize the quaternion
            q_norm = jnp.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
            w, qx, qy, qz = q / q_norm
            # Convert to direction cosine matrix
            return jnp.array(
                [
                    [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
                    [2 * (qx * qy + qz * w), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * w)],
                    [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx**2 + qy**2)],
                ]
            )

        return qdcm_fn

    @visitor(SSMP)
    def _visit_ssmp(self, node: SSMP):
        """Lower skew-symmetric matrix for quaternion dynamics (4x4).

        Creates a 4x4 skew-symmetric matrix from angular velocity vector for
        quaternion kinematic propagation: q_dot = 0.5 * SSMP(omega) @ q

        The SSMP matrix is used in quaternion kinematics to compute quaternion
        derivatives from angular velocity vectors.

        Args:
            node: SSMP expression node

        Returns:
            Function (x, u, node, params) -> 4x4 skew-symmetric matrix

        Note:
            For angular velocity w = [x, y, z], returns:
            [[0, -x, -y, -z],
             [x,  0,  z, -y],
             [y, -z,  0,  x],
             [z,  y, -x,  0]]
        """
        f = self.lower(node.w)

        def ssmp_fn(x, u, node, params):
            w = f(x, u, node, params)
            wx, wy, wz = w[0], w[1], w[2]
            return jnp.array(
                [
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0],
                ]
            )

        return ssmp_fn

    @visitor(SSM)
    def _visit_ssm(self, node: SSM):
        """Lower skew-symmetric matrix for cross product (3x3).

        Creates a 3x3 skew-symmetric matrix from a vector such that
        SSM(a) @ b = a x b (cross product).

        The SSM is the matrix representation of the cross product operator,
        allowing cross products to be computed as matrix-vector multiplication.

        Args:
            node: SSM expression node

        Returns:
            Function (x, u, node, params) -> 3x3 skew-symmetric matrix

        Note:
            For vector w = [x, y, z], returns:
            [[ 0, -z,  y],
             [ z,  0, -x],
             [-y,  x,  0]]
        """
        f = self.lower(node.w)

        def ssm_fn(x, u, node, params):
            w = f(x, u, node, params)
            wx, wy, wz = w[0], w[1], w[2]
            return jnp.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

        return ssm_fn

    @visitor(Diag)
    def _visit_diag(self, node: Diag):
        """Lower diagonal matrix construction to JAX function."""
        f = self.lower(node.operand)
        return lambda x, u, node, params: jnp.diag(f(x, u, node, params))

    @visitor(Or)
    def _visit_or(self, node: Or):
        """Lower STL disjunction (Or) to JAX using STLJax library.

        Converts a symbolic Or constraint to an STLJax Or formula for handling
        disjunctive task specifications. Each operand becomes an STLJax predicate.

        Args:
            node: Or expression node with multiple operands

        Returns:
            Function (x, u, node, params) -> STL robustness value

        Note:
            Uses STLJax library for signal temporal logic evaluation. The returned
            function computes the robustness metric for the disjunction, which is
            positive when at least one operand is satisfied.

        Example:
            Used for task specifications like "reach goal A OR goal B"::

                goal_A = ox.Norm(x - target_A) <= 1.0
                goal_B = ox.Norm(x - target_B) <= 1.0
                task = ox.Or(goal_A, goal_B)

        See Also:
            - stljax.formula.Or: Underlying STLJax implementation
            - STL robustness: Quantitative measure of constraint satisfaction
        """
        from stljax.formula import Or as STLOr
        from stljax.formula import Predicate

        # Lower each operand to get their functions
        operand_fns = [self.lower(operand) for operand in node.operands]

        # Return a function that evaluates the STLJax Or
        def or_fn(x, u, node, params):
            # Create STLJax predicates for each operand with current params
            predicates = []
            for i, operand_fn in enumerate(operand_fns):
                # Create a predicate function that captures the current params
                def make_pred_fn(fn):
                    return lambda x: fn(x, None, None, params)

                pred_fn = make_pred_fn(operand_fn)
                predicates.append(Predicate(f"pred_{i}", pred_fn))

            # Create and evaluate STLJax Or formula
            stl_or = STLOr(*predicates)
            return stl_or(x)

        return or_fn
