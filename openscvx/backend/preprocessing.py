from typing import Callable, Dict, Iterable, Set, Type, Union

import numpy as np

from openscvx.backend.control import Control
from openscvx.backend.expr import (
    Add,
    Concat,
    Constant,
    Constraint,
    Div,
    Equality,
    Expr,
    Index,
    Inequality,
    MatMul,
    Mul,
    Sub,
    traverse,
)
from openscvx.backend.state import State


# TODO: (norrisg) allow `traverse` to take a list of visitors, that way we can combine steps
def validate_variable_names(
    exprs: Iterable[Expr], *, reserved_prefix: str = "_", reserved_names: Set[str] = None
) -> None:
    """
    1) Ensure all State/Control names are unique *across distinct variables*.
    2) Ensure no user‐supplied name starts with `reserved_prefix`.
    3) Ensure no name collides with `reserved_names` if given.
    Raises ValueError on any violation.
    """
    seen_names = set()
    seen_ids = set()
    reserved = set(reserved_names or ())

    def visitor(node):
        if not isinstance(node, (State, Control)):
            return

        node_id = id(node)
        if node_id in seen_ids:
            # we already checked this exact object
            return

        name = node.name

        # 1) uniqueness across *different* variables
        if name in seen_names:
            raise ValueError(f"Duplicate variable name: {name!r}")

        # 2) no leading underscore
        if name.startswith(reserved_prefix):
            raise ValueError(
                f"Variable name {name!r} is reserved (cannot start with {reserved_prefix!r})"
            )

        # 3) no collision with explicit reserved set
        if name in reserved:
            raise ValueError(f"Variable name {name!r} collides with reserved name")

        seen_names.add(name)
        seen_ids.add(node_id)

    for e in exprs:
        traverse(e, visitor)


def collect_and_assign_slices(exprs: Iterable[Expr], *, start_index: int = 0):
    # 1) collect all State/Control nodes
    states, controls = [], []

    def visitor(node):
        # Cannot simply use `if node not in states` check since we have overloaded `__eq__` operator
        if isinstance(node, State) and not any(node is s for s in states):
            states.append(node)

        if isinstance(node, Control) and not any(node is c for c in controls):
            controls.append(node)

    for e in exprs:
        traverse(e, visitor)

    def assign(vars_list, start_index):
        # split into manual vs auto
        manual = [v for v in vars_list if v._slice is not None]
        auto = [v for v in vars_list if v._slice is None]

        if manual:
            # 1) shape‐match check
            for v in manual:
                dim = int(np.prod(v.shape))
                sl = v._slice
                if (sl.stop - sl.start) != dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} is length {sl.stop - sl.start}, "
                        f"but variable has shape {v.shape} (dim {dim})"
                    )
            # sort by the start of their slices
            manual.sort(key=lambda v: v._slice.start)
            # 2a) must start at 0
            if manual[0]._slice.start != start_index:
                raise ValueError("User-defined slices must start at index 0")
            # 2b) check contiguity & no overlaps
            cursor = start_index
            for v in manual:
                sl = v._slice
                dim = int(np.prod(v.shape))
                if sl.start != cursor or sl.stop != cursor + dim:
                    raise ValueError(
                        f"Manual slice for {v.name!r} must be contiguous and non-overlapping"
                    )
                cursor += dim
            offset = cursor
        else:
            offset = start_index

        # 3) auto-assign the rest
        for v in auto:
            dim = int(np.prod(v.shape))
            v._slice = slice(offset, offset + dim)
            offset += dim

    # run separately on states (x) and controls (u)
    assign(states, start_index)
    assign(controls, start_index)


def _traverse_with_depth(expr: Expr, visit: Callable[[Expr, int], None], depth: int = 0):
    visit(expr, depth)
    for child in expr.children():
        _traverse_with_depth(child, visit, depth + 1)


def validate_constraints_at_root(exprs: Union[Expr, list[Expr]]):
    """
    Raise ValueError if any Constraint is found at depth>0.
    Accepts a single Expr or a list of Exprs.
    """
    # normalize to list
    expr_list = exprs if isinstance(exprs, (list, tuple)) else [exprs]

    for expr in expr_list:

        def visit(node: Expr, depth: int):
            if depth > 0 and isinstance(node, Constraint):
                raise ValueError(
                    f"Nested Constraint found at depth {depth!r}: {node!r}; "
                    "constraints must only appear as top‐level roots"
                )

        _traverse_with_depth(expr, visit, depth=0)


_SHAPE_VISITORS: Dict[Type[Expr], Callable[[Expr], tuple[int, ...]]] = {}


def validate_shapes(exprs):
    exprs = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    for e in exprs:
        dispatch(e)  # will raise ValueError if anything’s wrong


def visitor(expr_cls: Type[Expr]):
    def register(fn: Callable[[Expr], tuple[int, ...]]):
        _SHAPE_VISITORS[expr_cls] = fn
        return fn

    return register


def dispatch(expr: Expr) -> tuple[int, ...]:
    fn = _SHAPE_VISITORS.get(type(expr))
    if fn is None:
        raise NotImplementedError(f"No shape rule for {type(expr).__name__}")
    return fn(expr)


def _broadcast_shape_for(node: Expr) -> tuple[int, ...]:
    # gather all child shapes
    shapes = [dispatch(child) for child in node.children()]
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError as e:
        op = type(node).__name__
        raise ValueError(f"{op} shapes not broadcastable: {shapes}") from e


@visitor(Constant)
def visit_constant(c: Constant):
    return c.value.shape


@visitor(Add)
@visitor(Sub)
@visitor(Mul)
@visitor(Div)
def visit_binary_op(node: Expr) -> tuple[int, ...]:
    return _broadcast_shape_for(node)


@visitor(MatMul)
def visit_matmul(node: MatMul):
    L, R = dispatch(node.left), dispatch(node.right)
    if len(L) < 2 or len(R) < 2 or L[-1] != R[-2]:
        raise ValueError(f"MatMul incompatible: {L} @ {R}")
    return L[:-1] + (R[-1],)


@visitor(Concat)
def visit_concat(node: Concat):
    shapes = [dispatch(e) for e in node.exprs]
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"Concat rank mismatch: {shapes}")
    if any(s[1:] != shapes[0][1:] for s in shapes[1:]):
        raise ValueError(f"Concat non-0 dims differ: {shapes}")
    return (sum(s[0] for s in shapes),) + shapes[0][1:]


@visitor(Index)
def visit_index(node: Index):
    base_shape = dispatch(node.base)
    dummy = np.zeros(base_shape)
    try:
        result = dummy[node.index]
    except Exception as e:
        raise ValueError(f"Bad index {node.index} for shape {base_shape}") from e
    return result.shape


@visitor(Equality)
@visitor(Inequality)
def visit_constraint(node: Constraint) -> tuple[int, ...]:
    # 1) get the two operand shapes
    L_shape = dispatch(node.lhs)
    R_shape = dispatch(node.rhs)

    # 2) figure out their broadcasted shape (or error if incompatible)
    try:
        out_shape = np.broadcast_shapes(L_shape, R_shape)
    except ValueError as e:
        op = type(node).__name__
        raise ValueError(f"{op} not broadcastable: {L_shape} vs {R_shape}") from e

    # 3) ensure that broadcast result is “scalar” in the sense that total size == 1
    total_size = int(np.prod(out_shape))
    if total_size != 1:
        op = type(node).__name__
        raise ValueError(
            f"{op} must be scalar-valued (total size==1), but got broadcast shape {out_shape}"
        )

    # 4) return () as usual
    return ()
