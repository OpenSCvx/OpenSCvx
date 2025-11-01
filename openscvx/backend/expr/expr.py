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

    def __repr__(self):
        inner = " + ".join(repr(e) for e in self.terms)
        return f"({inner})"


class Sub(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class Mul(Expr):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Mul requires two or more operands")
        self.factors = [to_expr(a) for a in args]

    def children(self):
        return list(self.factors)

    def __repr__(self):
        inner = " * ".join(repr(e) for e in self.factors)
        return f"({inner})"


class Div(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} / {self.right!r})"


class MatMul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def children(self):
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left!r} * {self.right!r})"


class Neg(Expr):
    def __init__(self, operand):
        self.operand = operand

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"(-{self.operand!r})"


class Sum(Expr):
    """Sum all elements of an expression (reduction operation)"""

    def __init__(self, operand):
        self.operand = to_expr(operand)

    def children(self):
        return [self.operand]

    def __repr__(self):
        return f"sum({self.operand!r})"


class Index(Expr):
    """Expr that means “take this Expr and index/slice it.”"""

    def __init__(self, base: Expr, index: Union[int, slice, tuple]):
        self.base = base
        self.index = index

    def children(self):
        return [self.base]

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

    def __repr__(self):
        inner = ", ".join(repr(e) for e in self.exprs)
        return f"Concat({inner})"


class Power(Expr):
    def __init__(self, base, exponent):
        self.base = to_expr(base)
        self.exponent = to_expr(exponent)

    def children(self):
        return [self.base, self.exponent]

    def __repr__(self):
        return f"({self.base!r})**({self.exponent!r})"


class Constant(Expr):
    def __init__(self, value: np.ndarray):
        # Normalize immediately upon construction to ensure consistency
        # This ensures Constant(5.0) and Constant([5.0]) create identical objects
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = np.squeeze(value)

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

    def __repr__(self):
        return f"{self.lhs!r} == {self.rhs!r}"


class Inequality(Constraint):
    """Represents lhs <= rhs"""

    def __repr__(self):
        return f"{self.lhs!r} <= {self.rhs!r}"


# Import canonicalization and shape checking systems at the end to avoid circular imports
import numpy as np

from ..canonicalizer import canon_visitor, canonicalize
from ..shape_checker import _broadcast_shape_for, check_shape, shape_visitor


# Shape visitors for leaf nodes
@shape_visitor(Parameter)
@shape_visitor(Constant)
def check_shape_constant(c):
    if isinstance(c, Constant):
        # Verify the invariant: constants should already be squeezed during construction
        original_shape = c.value.shape
        squeezed_shape = np.squeeze(c.value).shape
        if original_shape != squeezed_shape:
            raise ValueError(
                f"Constant not properly normalized: has shape {original_shape} but should have shape {squeezed_shape}. "
                "Constants should be squeezed during construction."
            )
        return c.value.shape
    else:  # Parameter
        return c.shape


# Shape visitors for core operations
@shape_visitor(Add)
@shape_visitor(Sub)
@shape_visitor(Mul)
@shape_visitor(Div)
def check_shape_binary_op(node) -> tuple[int, ...]:
    return _broadcast_shape_for(node)


@shape_visitor(MatMul)
def check_shape_matmul(node: MatMul):
    L, R = check_shape(node.left), check_shape(node.right)

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


@shape_visitor(Concat)
def check_shape_concat(node: Concat):
    shapes = [check_shape(e) for e in node.exprs]
    shapes = [(1,) if len(s) == 0 else s for s in shapes]
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"Concat rank mismatch: {shapes}")
    if any(s[1:] != shapes[0][1:] for s in shapes[1:]):
        raise ValueError(f"Concat non-0 dims differ: {shapes}")
    return (sum(s[0] for s in shapes),) + shapes[0][1:]


@shape_visitor(Sum)
def check_shape_sum(node: Sum) -> tuple[int, ...]:
    """sum() reduces any shape to a scalar"""
    # Validate that the operand has a valid shape
    operand_shape = check_shape(node.operand)
    # Sum always produces a scalar regardless of input shape
    return ()


@shape_visitor(Index)
def check_shape_index(node: Index):
    base_shape = check_shape(node.base)
    dummy = np.zeros(base_shape)
    try:
        result = dummy[node.index]
    except Exception as e:
        raise ValueError(f"Bad index {node.index} for shape {base_shape}") from e
    return result.shape


@shape_visitor(Neg)
def check_shape_neg(node: Neg) -> tuple[int, ...]:
    return check_shape(node.operand)


@shape_visitor(Equality)
@shape_visitor(Inequality)
def check_shape_constraint(node) -> tuple[int, ...]:
    # 1) get the two operand shapes
    L_shape = check_shape(node.lhs)
    R_shape = check_shape(node.rhs)

    # 2) figure out their broadcasted shape (or error if incompatible)
    try:
        np.broadcast_shapes(L_shape, R_shape)
    except ValueError as e:
        op = type(node).__name__
        raise ValueError(f"{op} not broadcastable: {L_shape} vs {R_shape}") from e

    # 3) Allow vector constraints - they're interpreted element-wise
    # 4) return () as usual
    return ()


@shape_visitor(Power)
def check_shape_power(node: Power) -> tuple[int, ...]:
    """power preserves the broadcasted shape of base and exponent"""
    return _broadcast_shape_for(node)


# Canonicalization visitors for leaf nodes
@canon_visitor(Parameter)
@canon_visitor(Constant)
def canon_leaf(node):
    # Leaf nodes are already canonical
    return node


# Canonicalization visitors for core operations
@canon_visitor(Add)
def canon_add(node: Add):
    # Flatten, recurse, fold, eliminate zero, collapse singleton
    terms = []
    const_vals = []

    for t in node.terms:
        c = canonicalize(t)
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


@canon_visitor(Mul)
def canon_mul(node: Mul):
    factors = []
    const_vals = []

    for f in node.factors:
        c = canonicalize(f)
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


@canon_visitor(Sub)
def canon_sub(node: Sub):
    # Canonicalize children, but keep as binary
    left = canonicalize(node.left)
    right = canonicalize(node.right)
    # Maybe special-case Constant-Constant?
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(left.value - right.value)
    return Sub(left, right)


@canon_visitor(Div)
def canon_div(node: Div):
    lhs = canonicalize(node.left)
    rhs = canonicalize(node.right)
    if isinstance(lhs, Constant) and isinstance(rhs, Constant):
        return Constant(lhs.value / rhs.value)
    return Div(lhs, rhs)


@canon_visitor(Neg)
def canon_neg(node: Neg):
    o = canonicalize(node.operand)
    if isinstance(o, Constant):
        return Constant(-o.value)
    return Neg(o)


@canon_visitor(Concat)
def canon_concat(node: Concat):
    exprs = [canonicalize(e) for e in node.exprs]
    return Concat(*exprs)


@canon_visitor(Index)
def canon_index(node: Index):
    base = canonicalize(node.base)
    return Index(base, node.index)


@canon_visitor(Inequality)
def canon_inequality(node: Inequality):
    diff = Sub(node.lhs, node.rhs)
    canon_diff = canonicalize(diff)
    new_ineq = Inequality(canon_diff, Constant(np.array(0)))
    new_ineq.is_convex = node.is_convex  # Preserve convex flag
    return new_ineq


@canon_visitor(Equality)
def canon_equality(node: Equality):
    diff = Sub(node.lhs, node.rhs)
    canon_diff = canonicalize(diff)
    new_eq = Equality(canon_diff, Constant(np.array(0)))
    new_eq.is_convex = node.is_convex  # Preserve convex flag
    return new_eq


@canon_visitor(MatMul)
def canon_matmul(node: MatMul):
    # Canonicalize operands but preserve the operation
    left = canonicalize(node.left)
    right = canonicalize(node.right)
    return MatMul(left, right)


@canon_visitor(Sum)
def canon_sum(node: Sum):
    # Canonicalize the operand
    operand = canonicalize(node.operand)
    return Sum(operand)


@canon_visitor(Power)
def canon_power(node: Power):
    # Canonicalize both operands
    base = canonicalize(node.base)
    exponent = canonicalize(node.exponent)
    return Power(base, exponent)
