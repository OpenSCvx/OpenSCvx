from typing import Callable, Optional, Tuple, Union

import numpy as np


class Expr:
    """
    Base class for symbolic expressions in optimization problems.

    Note: This class is currently not being used.
    """

    # Give Expr objects higher priority than numpy arrays in operations
    __array_priority__ = 1000

    def __le__(self, other):
        return Inequality(self, to_expr(other))

    def __ge__(self, other):
        return Inequality(to_expr(other), self)

    def __eq__(self, other):
        return Equality(self, to_expr(other))

    def __add__(self, other):
        return Add(self, to_expr(other))

    def __radd__(self, other):
        return Add(to_expr(other), self)

    def __sub__(self, other):
        return Sub(self, to_expr(other))

    def __rsub__(self, other):
        # e.g. 5 - a  ⇒ Sub(Constant(5), a)
        return Sub(to_expr(other), self)

    def __truediv__(self, other):
        return Div(self, to_expr(other))

    def __rtruediv__(self, other):
        # e.g. 10 / a
        return Div(to_expr(other), self)

    def __mul__(self, other):
        return Mul(self, to_expr(other))

    def __rmul__(self, other):
        return Mul(to_expr(other), self)

    def __matmul__(self, other):
        return MatMul(self, to_expr(other))

    def __rmatmul__(self, other):
        return MatMul(to_expr(other), self)

    def __rle__(self, other):
        # other <= self  =>  Inequality(other, self)
        return Inequality(to_expr(other), self)

    def __rge__(self, other):
        # other >= self  =>  Inequality(self, other)
        return Inequality(self, to_expr(other))

    def __req__(self, other):
        # other == self  =>  Equality(other, self)
        return Equality(to_expr(other), self)

    def __neg__(self):
        return Neg(self)

    def __pow__(self, other):
        return Power(self, to_expr(other))

    def __rpow__(self, other):
        return Power(to_expr(other), self)

    def __getitem__(self, idx):
        return Index(self, idx)

    @property
    def T(self):
        from .linalg import Transpose

        return Transpose(self)

    def children(self):
        return []

    def canonicalize(self) -> "Expr":
        """
        Return a canonical (simplified) form of this expression.

        Canonicalization performs algebraic simplifications such as:
        - Constant folding (e.g., 2 + 3 → 5)
        - Identity elimination (e.g., x + 0 → x, x * 1 → x)
        - Flattening nested operations (e.g., Add(Add(a, b), c) → Add(a, b, c))
        - Algebraic rewrites (e.g., constraints to standard form)

        Returns:
            Expr: A canonical version of this expression

        Raises:
            NotImplementedError: If canonicalization is not implemented for this node type
        """
        raise NotImplementedError(f"canonicalize() not implemented for {self.__class__.__name__}")

    def check_shape(self) -> Tuple[int, ...]:
        """
        Compute and validate the shape of this expression.

        This method:
        1. Recursively checks shapes of all child expressions
        2. Validates that operations are shape-compatible (e.g., broadcasting rules)
        3. Returns the output shape of this expression

        For example:
        - A Parameter with shape (3, 4) returns (3, 4)
        - MatMul of (3, 4) @ (4, 5) returns (3, 5)
        - Sum of any shape returns () (scalar)
        - Add broadcasts shapes like NumPy

        Returns:
            tuple: The shape of this expression as a tuple of integers.
                   Empty tuple () represents a scalar.

        Raises:
            NotImplementedError: If shape checking is not implemented for this node type
            ValueError: If the expression has invalid shapes (e.g., incompatible dimensions)
        """
        raise NotImplementedError(f"check_shape() not implemented for {self.__class__.__name__}")

    def pretty(self, indent=0):
        pad = "  " * indent
        pad = "  " * indent
        lines = [f"{pad}{self.__class__.__name__}"]
        for child in self.children():
            lines.append(child.pretty(indent + 1))
        return "\n".join(lines)


class Leaf(Expr):
    """
    Base class for leaf nodes (terminal expressions) in the symbolic expression tree.

    Leaf nodes represent named symbolic variables that don't have child expressions.
    This includes Parameters, Variables, States, and Controls.

    Attributes:
        name (str): Name identifier for the leaf node
        _shape (tuple): Shape of the leaf node
    """

    def __init__(self, name: str, shape: tuple = ()):
        """Initialize a Leaf node.

        Args:
            name (str): Name identifier for the leaf node
            shape (tuple): Shape of the leaf node
        """
        super().__init__()
        self.name = name
        self._shape = shape

    @property
    def shape(self):
        """Get the shape of the leaf node.

        Returns:
            tuple: Shape of the leaf node
        """
        return self._shape

    def children(self):
        """Leaf nodes have no children.

        Returns:
            list: Empty list since leaf nodes are terminal
        """
        return []

    def canonicalize(self) -> "Expr":
        """Leaf nodes are already in canonical form.

        Returns:
            Expr: Returns self since leaf nodes are already canonical
        """
        return self

    def check_shape(self) -> Tuple[int, ...]:
        """Return the shape of this leaf node.

        Returns:
            tuple: The shape of the leaf node
        """
        return self._shape

    def __repr__(self):
        """String representation of the leaf node.

        Returns:
            str: A string describing the leaf node
        """
        return f"{self.__class__.__name__}('{self.name}', shape={self.shape})"


class Parameter(Leaf):
    """Parameter that can be changed at runtime without recompilation.

    Parameters are symbolic variables whose values are provided at solve time
    through the problem's parameter dictionary. They allow for efficient
    parameter sweeps without needing to recompile the optimization problem.
    """

    def __init__(self, name: str, shape: tuple = ()):
        """Initialize a Parameter node.

        Args:
            name (str): Name identifier for the parameter
            shape (tuple): Shape of the parameter
        """
        super().__init__(name, shape)


def to_expr(x: Union[Expr, float, int, np.ndarray]) -> Expr:
    return x if isinstance(x, Expr) else Constant(np.array(x))


def traverse(expr: Expr, visit: Callable[[Expr], None]):
    visit(expr)
    for child in expr.children():
        traverse(child, visit)


class Add(Expr):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Add requires two or more operands")
        self.terms = [to_expr(a) for a in args]

    def children(self):
        return list(self.terms)

    def canonicalize(self) -> "Expr":
        """Canonicalize addition: flatten, fold constants, eliminate zeros."""
        terms = []
        const_vals = []

        for t in self.terms:
            c = t.canonicalize()
            if isinstance(c, Add):
                terms.extend(c.terms)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                terms.append(c)

        if const_vals:
            total = sum(const_vals)
            # If not all-zero, keep it
            if not (isinstance(total, np.ndarray) and np.all(total == 0)):
                terms.append(Constant(total))

        if not terms:
            return Constant(np.array(0))
        if len(terms) == 1:
            return terms[0]
        return Add(*terms)

    def check_shape(self) -> Tuple[int, ...]:
        """Addition broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Add shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        inner = " + ".join(repr(e) for e in self.terms)
        return f"({inner})"


class Sub(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        """Canonicalize subtraction: fold constants if both sides are constants."""
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value - right.value)
        return Sub(left, right)

    def check_shape(self) -> Tuple[int, ...]:
        """Subtraction broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Sub shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class Mul(Expr):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Mul requires two or more operands")
        self.factors = [to_expr(a) for a in args]

    def children(self):
        return list(self.factors)

    def canonicalize(self) -> "Expr":
        """Canonicalize multiplication: flatten, fold constants, eliminate ones."""
        factors = []
        const_vals = []

        for f in self.factors:
            c = f.canonicalize()
            if isinstance(c, Mul):
                factors.extend(c.factors)
            elif isinstance(c, Constant):
                const_vals.append(c.value)
            else:
                factors.append(c)

        if const_vals:
            prod = np.prod(const_vals)
            # If prod != 1, keep it
            if not (isinstance(prod, np.ndarray) and np.all(prod == 1)):
                factors.append(Constant(prod))

        if not factors:
            return Constant(np.array(1))
        if len(factors) == 1:
            return factors[0]
        return Mul(*factors)

    def check_shape(self) -> Tuple[int, ...]:
        """Multiplication broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Mul shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        inner = " * ".join(repr(e) for e in self.factors)
        return f"({inner})"


class Div(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        """Canonicalize division: fold constants if both sides are constants."""
        lhs = self.left.canonicalize()
        rhs = self.right.canonicalize()
        if isinstance(lhs, Constant) and isinstance(rhs, Constant):
            return Constant(lhs.value / rhs.value)
        return Div(lhs, rhs)

    def check_shape(self) -> Tuple[int, ...]:
        """Division broadcasts shapes like NumPy."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Div shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        return f"({self.left!r} / {self.right!r})"


class MatMul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def canonicalize(self) -> "Expr":
        """Canonicalize matrix multiplication."""
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        return MatMul(left, right)

    def check_shape(self) -> Tuple[int, ...]:
        """Check matrix multiplication shape compatibility and return result shape."""
        L, R = self.left.check_shape(), self.right.check_shape()

        # Handle different matmul cases:
        # Matrix @ Matrix: (m,n) @ (n,k) -> (m,k)
        # Matrix @ Vector: (m,n) @ (n,) -> (m,)
        # Vector @ Matrix: (m,) @ (m,n) -> (n,)
        # Vector @ Vector: (m,) @ (m,) -> ()

        if len(L) == 0 or len(R) == 0:
            raise ValueError(f"MatMul requires at least 1D operands: {L} @ {R}")

        if len(L) == 1 and len(R) == 1:
            # Vector @ Vector -> scalar
            if L[0] != R[0]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return ()
        elif len(L) == 1:
            # Vector @ Matrix: (m,) @ (m,n) -> (n,)
            if len(R) < 2 or L[0] != R[-2]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return R[-1:]
        elif len(R) == 1:
            # Matrix @ Vector: (m,n) @ (n,) -> (m,)
            if len(L) < 2 or L[-1] != R[0]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return L[:-1]
        else:
            # Matrix @ Matrix: (...,m,n) @ (...,n,k) -> (...,m,k)
            if len(L) < 2 or len(R) < 2 or L[-1] != R[-2]:
                raise ValueError(f"MatMul incompatible: {L} @ {R}")
            return L[:-1] + (R[-1],)

    def __repr__(self):
        return f"({self.left!r} * {self.right!r})"


class Neg(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize negation: fold if operand is a constant."""
        o = self.operand.canonicalize()
        if isinstance(o, Constant):
            return Constant(-o.value)
        return Neg(o)

    def check_shape(self) -> Tuple[int, ...]:
        """Negation preserves the shape of its operand."""
        return self.operand.check_shape()

    def __repr__(self):
        return f"(-{self.operand!r})"


class Sum(Expr):
    """Sum all elements of an expression (reduction operation)"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def canonicalize(self) -> "Expr":
        """Canonicalize sum operation."""
        operand = self.operand.canonicalize()
        return Sum(operand)

    def check_shape(self) -> Tuple[int, ...]:
        """Sum reduces any shape to a scalar."""
        # Validate that the operand has a valid shape
        self.operand.check_shape()
        # Sum always produces a scalar regardless of input shape
        return ()

    def __repr__(self):
        return f"sum({self.operand!r})"


class Index(Expr):
    """Expr that means "take this Expr and index/slice it." """

    def __init__(self, base: Expr, index: Union[int, slice, tuple]):
        self.base = base
        self.index = index

    def children(self):
        return [self.base]

    def canonicalize(self) -> "Expr":
        """Canonicalize indexing operation."""
        base = self.base.canonicalize()
        return Index(base, self.index)

    def check_shape(self) -> Tuple[int, ...]:
        """Compute the shape after indexing."""
        base_shape = self.base.check_shape()
        dummy = np.zeros(base_shape)
        try:
            result = dummy[self.index]
        except Exception as e:
            raise ValueError(f"Bad index {self.index} for shape {base_shape}") from e
        return result.shape

    def __repr__(self):
        return f"{self.base!r}[{self.index!r}]"


class Concat(Expr):
    """
    Concatenate a sequence of Exprs into one long vector.
    """

    def __init__(self, *exprs: Expr):
        # wrap raw values as Constant if needed
        self.exprs = [to_expr(e) for e in exprs]

    def children(self):
        return list(self.exprs)

    def canonicalize(self) -> "Expr":
        """Canonicalize concatenation operation."""
        exprs = [e.canonicalize() for e in self.exprs]
        return Concat(*exprs)

    def check_shape(self) -> Tuple[int, ...]:
        """Check concatenation shape compatibility and return result shape."""
        shapes = [e.check_shape() for e in self.exprs]
        shapes = [(1,) if len(s) == 0 else s for s in shapes]
        rank = len(shapes[0])
        if any(len(s) != rank for s in shapes):
            raise ValueError(f"Concat rank mismatch: {shapes}")
        if any(s[1:] != shapes[0][1:] for s in shapes[1:]):
            raise ValueError(f"Concat non-0 dims differ: {shapes}")
        return (sum(s[0] for s in shapes),) + shapes[0][1:]

    def __repr__(self):
        inner = ", ".join(repr(e) for e in self.exprs)
        return f"Concat({inner})"


class Power(Expr):
    def __init__(self, base, exponent):
        self.base = to_expr(base)
        self.exponent = to_expr(exponent)

    def children(self):
        return [self.base, self.exponent]

    def canonicalize(self) -> "Expr":
        """Canonicalize power operation."""
        base = self.base.canonicalize()
        exponent = self.exponent.canonicalize()
        return Power(base, exponent)

    def check_shape(self) -> Tuple[int, ...]:
        """Power preserves the broadcasted shape of base and exponent."""
        shapes = [child.check_shape() for child in self.children()]
        try:
            return np.broadcast_shapes(*shapes)
        except ValueError as e:
            raise ValueError(f"Power shapes not broadcastable: {shapes}") from e

    def __repr__(self):
        return f"({self.base!r})**({self.exponent!r})"


class Constant(Expr):
    def __init__(self, value: np.ndarray):
        # Normalize immediately upon construction to ensure consistency
        # This ensures Constant(5.0) and Constant([5.0]) create identical objects
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = np.squeeze(value)

    def canonicalize(self) -> "Expr":
        """Constants are already in canonical form.

        Returns:
            Expr: Returns self since constants are already canonical
        """
        return self

    def check_shape(self) -> Tuple[int, ...]:
        """Return the shape of this constant's value.

        Returns:
            tuple: The shape of the constant's numpy array value
        """
        # Verify the invariant: constants should already be squeezed during construction
        original_shape = self.value.shape
        squeezed_shape = np.squeeze(self.value).shape
        if original_shape != squeezed_shape:
            raise ValueError(
                f"Constant not properly normalized: has shape {original_shape} but should have shape {squeezed_shape}. "
                "Constants should be squeezed during construction."
            )
        return self.value.shape

    def __repr__(self):
        # Show clean representation - always show as Python values, not numpy arrays
        if self.value.ndim == 0:
            # Scalar: show as plain number
            return f"Const({self.value.item()!r})"
        else:
            # Array: show as Python list for readability
            return f"Const({self.value.tolist()!r})"


class Constraint(Expr):
    """
    Abstract base for all constraints.
    """

    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs
        self.is_convex = False

    def children(self):
        return [self.lhs, self.rhs]

    def at(self, nodes: Union[list, tuple]):
        """Apply this constraint only at specific discrete nodes.

        Args:
            nodes: List of node indices where the constraint should be enforced

        Returns:
            NodalConstraint wrapping this constraint with node specification
        """
        from .constraint import NodalConstraint

        return NodalConstraint(self, list(nodes))

    def over(
        self,
        interval: tuple[int, int],
        penalty: str = "squared_relu",
        idx: Optional[int] = None,
        check_nodally: bool = False,
    ):
        """Apply this constraint over a continuous interval using CTCS.

        Args:
            interval: Tuple of (start, end) node indices for the continuous interval
            penalty: Penalty function type ("squared_relu", "huber", "smooth_relu")
            idx: Optional grouping index for multiple augmented states
            check_nodally: Whether to also enforce this constraint nodally

        Returns:
            CTCS constraint wrapping this constraint with interval specification
        """
        from .constraint import CTCS

        return CTCS(self, penalty=penalty, nodes=interval, idx=idx, check_nodally=check_nodally)

    def convex(self) -> "Constraint":
        """Mark this constraint as convex for CVXPy lowering.

        Returns:
            Self with convex flag set to True (enables method chaining)
        """
        self.is_convex = True
        return self


class Equality(Constraint):
    """Represents lhs == rhs."""

    def canonicalize(self) -> "Expr":
        """Canonicalize equality constraint to standard form: (lhs - rhs) == 0."""
        diff = Sub(self.lhs, self.rhs)
        canon_diff = diff.canonicalize()
        new_eq = Equality(canon_diff, Constant(np.array(0)))
        new_eq.is_convex = self.is_convex  # Preserve convex flag
        return new_eq

    def check_shape(self) -> Tuple[int, ...]:
        """Check that constraint operands are broadcastable. Returns scalar shape."""
        L_shape = self.lhs.check_shape()
        R_shape = self.rhs.check_shape()

        # Figure out their broadcasted shape (or error if incompatible)
        try:
            np.broadcast_shapes(L_shape, R_shape)
        except ValueError as e:
            raise ValueError(f"Equality not broadcastable: {L_shape} vs {R_shape}") from e

        # Allow vector constraints - they're interpreted element-wise
        # Return () as constraints always produce a scalar
        return ()

    def __repr__(self):
        return f"{self.lhs!r} == {self.rhs!r}"


class Inequality(Constraint):
    """Represents lhs <= rhs"""

    def canonicalize(self) -> "Expr":
        """Canonicalize inequality constraint to standard form: (lhs - rhs) <= 0."""
        diff = Sub(self.lhs, self.rhs)
        canon_diff = diff.canonicalize()
        new_ineq = Inequality(canon_diff, Constant(np.array(0)))
        new_ineq.is_convex = self.is_convex  # Preserve convex flag
        return new_ineq

    def check_shape(self) -> Tuple[int, ...]:
        """Check that constraint operands are broadcastable. Returns scalar shape."""
        L_shape = self.lhs.check_shape()
        R_shape = self.rhs.check_shape()

        # Figure out their broadcasted shape (or error if incompatible)
        try:
            np.broadcast_shapes(L_shape, R_shape)
        except ValueError as e:
            raise ValueError(f"Inequality not broadcastable: {L_shape} vs {R_shape}") from e

        # Allow vector constraints - they're interpreted element-wise
        # Return () as constraints always produce a scalar
        return ()

    def __repr__(self):
        return f"{self.lhs!r} <= {self.rhs!r}"
