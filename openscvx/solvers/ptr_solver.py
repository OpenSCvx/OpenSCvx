"""Abstract base class for Penalized Trust-Region (PTR) convex subproblem solvers.

The PTR formulation — its variables, slack structure, cost terms, and
linearization contract — is shared across backends. This module defines that
contract; concrete backends (CVXPy, QPAX, Moreau, ICORE) live in sibling modules and
implement the assembly/dispatch that each backend's modeling layer requires.

Two per-iteration entry points coexist:

* The historical NumPy contract — four :meth:`PTRSolver.update_*` stages
  followed by :meth:`PTRSolver.solve`, returning a :class:`PTRSolveResult`.
  Used by today's Python-side SCP loop and by direct interactive use of
  :class:`~openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`.
* The JAX-pure contract — :meth:`PTRSolver.iteration_callback`, which returns
  a ``(state, SubproblemData) -> SubproblemSolution`` callable built once at
  :meth:`~openscvx.solvers.base.ConvexSolver.initialize`. All backends emit
  the same input/output pytree shape so the callable composes with
  ``jax.jit`` / ``jax.vmap`` and, downstream, ``lax.while_loop``-driven SCP
  iteration.

Backends:
    :class:`openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`
        DCP graph assembled via CVXPy, dispatched to a conic solver
        (QOCO, CLARABEL, ...). The JAX-pure callback wraps the host solve
        in :func:`jax.pure_callback` (``vmap_method="sequential"``).
    :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`
        Flat ``(Q, q, A, b, G, h)`` assembled as JAX arrays and solved with
        ``qpax.solve_qp`` (NumPy path) or the differentiable
        ``qpax.solve_qp_primal`` (JAX-pure path). Enables an end-to-end
        JAX-differentiable SCP loop in follow-up work.
    :class:`openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver`
        Sparse conic program assembled as CSR JAX arrays and solved with
        ``moreau.jax.Solver``. The NumPy path warm-starts between SCP
        iterations. The JAX-pure path calls that solver's jitted kernel,
        which returns ``(solution, info)`` and does not warm-start. Uses SOC
        epigraphs for the L1 / pos
        PTR penalties instead of QPAX-style slack expansion.
    :class:`openscvx.solvers.icore_ptr_solver.ICOREPTRSolver`
        Stage-wise conic OCP (augmented state ``[x, u, t_vb]``, uniform
        per-stage cones, equalities eliminated through the dynamics) solved
        with ICORE's Riccati-structured JAX interior-point kernel. Cold-start;
        float64 only.

QPAX, Moreau and ICORE share their non-assembly plumbing (argument merging,
scaling, guards, the ``update_*`` stash and its packing into
:class:`SubproblemData`) through :class:`JaxPTRSolver`.
"""

from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np

from .base import ConvexSolver
from .cones import ConeConstraint

if TYPE_CHECKING:
    import hashlib

    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState
    from openscvx.symbolic.constraint_set import ConstraintSet


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------


class StatusCode(IntEnum):
    """Subproblem solve outcome, encoded as ``int32`` for JAX traceability.

    ``iteration_callback`` returns a numeric status on the JAX side rather than
    a Python string so the result pytree stays valid under ``jax.jit`` /
    ``jax.vmap``. The SCP loop maps the code back to a label via
    :func:`status_code_to_str` only on the Python-loop printing path.
    """

    OPTIMAL = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
    UNKNOWN = 3


_STATUS_NAMES = {
    StatusCode.OPTIMAL: "optimal",
    StatusCode.INFEASIBLE: "infeasible",
    StatusCode.UNBOUNDED: "unbounded",
    StatusCode.UNKNOWN: "unknown",
}


def status_code_to_str(code: Union[int, jnp.ndarray, np.ndarray]) -> str:
    """Map a :class:`StatusCode` value (int or 0-d array) to its label."""
    return _STATUS_NAMES[StatusCode(int(code))]


_STATUS_STR_TO_CODE = {
    "optimal": StatusCode.OPTIMAL,
    "infeasible": StatusCode.INFEASIBLE,
    "unbounded": StatusCode.UNBOUNDED,
}


def status_str_to_code(status: str) -> StatusCode:
    """Map a backend-emitted status string to a :class:`StatusCode`.

    Used by :class:`CVXPyPTRSolver.iteration_callback` to coerce CVXPy's
    ``problem.status`` into the int32 form the JAX-pure result pytree
    requires. Only the three definite outcomes (``"optimal"``,
    ``"infeasible"``, ``"unbounded"``) are recognized — every other label
    CVXPy emits (``"optimal_inaccurate"``, ``"solver_error"``, ...) collapses
    to :attr:`StatusCode.UNKNOWN`.
    """
    return _STATUS_STR_TO_CODE.get(status, StatusCode.UNKNOWN)


# ---------------------------------------------------------------------------
# Per-iteration JAX-pure I/O pytrees
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SubproblemData:
    """JAX-pure inputs to :meth:`PTRSolver.iteration_callback`.

    Packs the per-iteration linearization, penalty weights, and boundary
    conditions into a single registered pytree so a single SCP iteration —
    assemble → solve → unpack — happens on the JAX boundary instead of
    bouncing through NumPy each step.

    Attributes:
        x_bar: Previous nodal state, shape ``(N, n_x)``.
        u_bar: Previous nodal control, shape ``(N, n_u)``.
        A_d, B_d, C_d: Continuous-step discretized Jacobians,
            shapes ``(N-1, n_x, n_x)`` / ``(N-1, n_x, n_u)`` / ``(N-1, n_x, n_u)``.
        x_prop: Propagated state, shape ``(N-1, n_x)``.
        x_prop_plus: Impulsive propagated state, shape ``(N, n_x)``. Zeros when
            no impulsive component is present.
        D_d, E_d: Impulsive Jacobians, shapes ``(N, n_x, n_x)`` /
            ``(N, n_x, n_u)``. Zeros when no impulsive component is present.
        nodal_g: Nodal constraint values stacked to fixed shape
            ``(N, n_nodal)``. Rows for nodes not in a constraint's ``nodes`` set
            are filled with zeros (mask-by-zero — backend assembly closes over
            each constraint's static ``nodes`` tuple to skip them).
        nodal_grad_x: Nodal constraint state gradients, shape
            ``(N, n_nodal, n_x)``.
        nodal_grad_u: Nodal constraint control gradients, shape
            ``(N, n_nodal, n_u)``.
        cross_g: Cross-node constraint values, shape ``(n_cross,)``.
        cross_grad_X: Cross-node state gradients, shape ``(n_cross, N, n_x)``.
        cross_grad_U: Cross-node control gradients, shape ``(n_cross, N, n_u)``.
        lam_prox: Trust-region weights, shape ``(N, n_x + n_u)``.
        lam_cost: Cost weight, scalar or shape ``(n_x,)``.
        lam_vc: Virtual-control penalty weights, shape ``(N-1, n_x)``.
        lam_vb_nodal: Nodal virtual-buffer weights, shape ``(N, n_nodal)``.
        lam_vb_cross: Cross-node virtual-buffer weights, shape ``(n_cross,)``.
        x_init: Initial state, shape ``(n_x,)``. ``jnp.nan`` sentinel where free.
        x_term: Terminal state, shape ``(n_x,)``. ``jnp.nan`` sentinel where free.
        params: Runtime parameters dict from the outer solve loop.  Threaded
            here so JAX-pure assembly functions see the per-batch vmapped
            parameter values (e.g. ``initial_position[b]``) rather than the
            frozen ``_parameters_dict`` captured at initialisation.
    """

    x_bar: jnp.ndarray
    u_bar: jnp.ndarray
    A_d: jnp.ndarray
    B_d: jnp.ndarray
    C_d: jnp.ndarray
    x_prop: jnp.ndarray
    x_prop_plus: jnp.ndarray
    D_d: jnp.ndarray
    E_d: jnp.ndarray
    nodal_g: jnp.ndarray
    nodal_grad_x: jnp.ndarray
    nodal_grad_u: jnp.ndarray
    cross_g: jnp.ndarray
    cross_grad_X: jnp.ndarray
    cross_grad_U: jnp.ndarray
    lam_prox: jnp.ndarray
    lam_cost: jnp.ndarray
    lam_vc: jnp.ndarray
    lam_vb_nodal: jnp.ndarray
    lam_vb_cross: jnp.ndarray
    x_init: jnp.ndarray
    x_term: jnp.ndarray
    #: Runtime parameters dict forwarded from the outer solve loop so that
    #: JAX-pure assembly paths (``_assemble_conic_jax``, ``_assemble_qp_jax``)
    #: see the per-batch vmapped values rather than the frozen init-time copy.
    params: dict

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SubproblemSolution:
    """JAX-pure output of :meth:`PTRSolver.iteration_callback`.

    All backends produce structurally-identical pytrees so the result composes
    with ``lax.while_loop`` in the downstream batchable-problem work.

    Attributes:
        x: Nodal state, shape ``(N, n_x)``. Unscaled.
        u: Nodal control, shape ``(N, n_u)``. Unscaled.
        nu: Virtual-control slack, shape ``(N-1, n_x)``.
        nu_vb: Nodal virtual-buffer slacks stacked to ``(N, n_nodal)`` (the
            CVXPy / QPAX list-of-arrays layout is collapsed at the
            ``iteration_callback`` output boundary).
        nu_vb_cross: Cross-node virtual-buffer slacks, shape ``(n_cross,)``.
        cost: Optimal objective value (scalar).
        status_code: :class:`StatusCode` value as ``int32``.
    """

    x: jnp.ndarray
    u: jnp.ndarray
    nu: jnp.ndarray
    nu_vb: jnp.ndarray
    nu_vb_cross: jnp.ndarray
    cost: jnp.ndarray
    status_code: jnp.ndarray

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@dataclass
class PTRSolveResult:
    """Result from solving a PTR convex subproblem.

    Contains the solution trajectories and slack variables from a single
    SCP iteration. All trajectories are unscaled (physical units).

    Attributes:
        x: State trajectory, shape (N, n_states). Unscaled.
        u: Control trajectory, shape (N, n_controls). Unscaled.
        nu: Virtual control slack for dynamics defects, shape (N-1, n_states).
        nu_vb: Nonconvex nodal constraint violation slacks. List of arrays,
            one per nodal constraint.
        nu_vb_cross: Cross-node constraint violation slacks. List of scalars,
            one per cross-node constraint.
        cost: Optimal objective value.
        status: Solver status string (e.g., "optimal", "infeasible").
    """

    x: np.ndarray
    u: np.ndarray
    nu: np.ndarray
    nu_vb: List[np.ndarray]
    nu_vb_cross: List[float]
    cost: float
    status: str


class PTRSolver(ConvexSolver):
    """Abstract base class for Penalized Trust-Region convex subproblem solvers.

    Defines the contract every PTR backend must satisfy: per-iteration entry
    points for updating the linearization point, constraint gradients, penalty
    weights, and boundary conditions, plus a ``solve()`` returning a
    :class:`PTRSolveResult`. The set of variables (state, control, virtual
    control ``nu``, per-constraint virtual buffer ``nu_vb``, cross-node slack
    ``nu_vb_cross``) is fixed by the PTR formulation; how each backend
    realizes those variables is an implementation detail.

    Subclasses must additionally implement :meth:`create_variables` and
    :meth:`initialize` from :class:`ConvexSolver`.

    Attributes:
        SUPPORTED_CONE_TYPES: The set of
            :class:`~openscvx.solvers.cones.ConeConstraint` subclasses this
            backend can assemble.  An empty frozenset (the default) means the
            backend rejects **all** user ``.convex()`` constraints.  Concrete
            subclasses declare their capabilities by overriding this attribute.
    """

    # Subclasses declare which cone types they support.
    SUPPORTED_CONE_TYPES: ClassVar[FrozenSet[Type[ConeConstraint]]] = frozenset()

    # Registry populated via __init_subclass__ so _raise_unsupported_cone_error
    # can suggest which backends do support a given cone type.
    _solver_registry: ClassVar[List[Type["PTRSolver"]]] = []

    #: Backend-specific solve options (tolerances, iteration caps, ...). Each
    #: concrete backend assigns its own in ``__init__``; declared here with a
    #: ``None`` default so the base :meth:`_hash_into` contract is self-contained
    #: — a subclass that omits it hashes as an empty mapping rather than raising.
    solver_args: Optional[dict] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        PTRSolver._solver_registry.append(cls)

    # ------------------------------------------------------------------
    # Principled convex-constraint lowering (shared by QPAX and Moreau)
    # ------------------------------------------------------------------

    def lower_convex_constraints(
        self,
        constraints: "ConstraintSet",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Lower user ``.convex()`` constraints into cone constraints.

        Separates auto-generated impulsive zero-pin equalities from genuine
        user constraints, canonicalises the latter via
        :func:`~openscvx.symbolic.canonicalize.canonicalize_nodal_constraint`,
        and verifies each cone type against :attr:`SUPPORTED_CONE_TYPES`.

        Cross-node convex constraints are not yet supported by the JAX
        backends and raise :class:`NotImplementedError`.

        Raises:
            NotImplementedError: For cross-node convex constraints or
                unsupported cone types (with a message listing which backends
                do support the cone type).
            ValueError: If a constraint is not affine in state/control.
        """
        from openscvx.symbolic.canonicalize import canonicalize_nodal_constraint

        if constraints.cross_node_convex:
            raise NotImplementedError(
                f"{type(self).__name__} does not support cross-node .convex() "
                f"constraints ({len(constraints.cross_node_convex)} defined). "
                "Use CVXPyPTRSolver."
            )

        pins, user_nodal = self._partition_nodal_convex(constraints)
        self._impulsive_pins = pins

        user_cones: List[ConeConstraint] = []
        for nc in user_nodal:
            cones = canonicalize_nodal_constraint(nc)
            for cone in cones:
                if type(cone) not in self.SUPPORTED_CONE_TYPES:
                    self._raise_unsupported_cone_error(cone)
                user_cones.append(cone)

        self._user_cone_constraints: List[ConeConstraint] = user_cones
        self._parameters_dict: Dict[str, Any] = parameters or {}
        return [], {}

    @staticmethod
    def _partition_nodal_convex(
        constraints: "ConstraintSet",
    ) -> Tuple[List[Tuple[List[int], slice]], List]:
        """Separate auto-generated impulsive zero-pin constraints from genuine
        user-defined convex constraints.

        Auto-generated pins match the pattern ``Control == 0`` injected by
        :func:`~openscvx.symbolic.lower._augment_impulsive_constraints`.  All
        other ``nodal_convex`` entries are returned as user-defined constraints.

        Returns:
            tuple: ``(pins, user_nodal)`` where *pins* is a list of
            ``(nodes, ctrl_slice)`` pairs and *user_nodal* is the list of
            :class:`~openscvx.symbolic.expr.constraint.NodalConstraint`
            objects that need full canonicalization.
        """
        from openscvx.symbolic.expr.constraint import Equality, NodalConstraint
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.expr import Constant

        pins: List[Tuple[List[int], slice]] = []
        user_nodal: List = []

        for entry in constraints.nodal_convex:
            if (
                isinstance(entry, NodalConstraint)
                and isinstance(entry.constraint, Equality)
                and isinstance(entry.constraint.rhs, Constant)
                and np.all(np.asarray(entry.constraint.rhs.value) == 0)
                and isinstance(entry.constraint.lhs, Control)
                and entry.constraint.lhs._slice is not None
            ):
                pins.append(([int(k) for k in entry.nodes], entry.constraint.lhs._slice))
            else:
                user_nodal.append(entry)

        return pins, user_nodal

    def _raise_unsupported_cone_error(self, cone: ConeConstraint) -> None:
        """Raise :class:`NotImplementedError` naming which backends support *cone*.

        Scans :attr:`_solver_registry` to find alternatives.
        """
        cone_cls = type(cone)
        alternatives = [
            cls.__name__
            for cls in PTRSolver._solver_registry
            if cone_cls in cls.SUPPORTED_CONE_TYPES
        ]
        msg = f"{type(self).__name__} does not support {cone_cls.__name__} constraints."
        if alternatives:
            msg += f"  Backends that do: {', '.join(alternatives)}."
        else:
            msg += (
                "  No registered JAX backend supports this cone type; use CVXPyPTRSolver instead."
            )
        raise NotImplementedError(msg)

    @abstractmethod
    def update_dynamics_linearization(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_prop: np.ndarray,
        x_prop_plus: np.ndarray | None = None,
        D_d: np.ndarray | None = None,
        E_d: np.ndarray | None = None,
    ) -> None:
        """Update dynamics linearization point and discrete-time matrices.

        Args:
            x_bar: Previous state trajectory, shape (N, n_states).
            u_bar: Previous control trajectory, shape (N, n_controls).
            A_d: Discretized state Jacobian, shape (N-1, n_states, n_states).
            B_d: Discretized control Jacobian (current node),
                shape (N-1, n_states, n_controls).
            C_d: Discretized control Jacobian (next node),
                shape (N-1, n_states, n_controls).
            x_prop: Propagated state from continuous dynamics,
                shape (N-1, n_states).
            x_prop_plus: Optional impulsive/discrete propagated state,
                shape (N, n_states).
            D_d: Optional impulsive/discrete Jacobian wrt state,
                shape (N, n_states, n_states).
            E_d: Optional impulsive/discrete Jacobian wrt control,
                shape (N, n_states, n_controls).
        """
        raise NotImplementedError

    @abstractmethod
    def update_constraint_linearizations(
        self,
        nodal: List[dict] = None,
        cross_node: List[dict] = None,
    ) -> None:
        """Update linearized constraint values and gradients.

        Args:
            nodal: List of dicts, one per nodal constraint, each containing:
                ``g`` (value at linearization point),
                ``grad_g_x`` (gradient w.r.t. state),
                ``grad_g_u`` (gradient w.r.t. control).
            cross_node: List of dicts, one per cross-node constraint, each
                containing: ``g``, ``grad_g_X`` (gradient w.r.t. full state
                trajectory), ``grad_g_U`` (gradient w.r.t. full control
                trajectory).
        """
        raise NotImplementedError

    @abstractmethod
    def update_penalties(
        self,
        lam_prox: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        """Update SCP penalty weights.

        Args:
            lam_prox: Trust region weights, shape (N, n_states + n_controls).
            lam_cost: Cost function weight. Scalar or shape (n_states,).
            lam_vc: Virtual control penalty weights, shape (N-1, n_states).
            lam_vb_nodal: Virtual buffer penalty weights for nodal constraints,
                shape (N, n_nodal_constraints).
            lam_vb_cross: Virtual buffer penalty weights for cross-node
                constraints, shape (n_cross_node_constraints,).
        """
        raise NotImplementedError

    @abstractmethod
    def update_boundary_conditions(
        self,
        x_init: np.ndarray = None,
        x_term: np.ndarray = None,
    ) -> None:
        """Update initial and/or terminal state parameters.

        Args:
            x_init: Initial state vector, shape (n_states,). Optional.
            x_term: Terminal state vector, shape (n_states,). Optional.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> PTRSolveResult:
        """Solve the convex subproblem and return a :class:`PTRSolveResult`.

        Call the four ``update_*`` methods first to set the linearization
        point, constraint gradients, penalties, and boundary conditions.
        """
        raise NotImplementedError

    @abstractmethod
    def iteration_callback(self) -> Callable[..., SubproblemSolution]:
        """Return a JAX-friendly ``(state, SubproblemData) -> SubproblemSolution``.

        Built once at :meth:`initialize`; called once per SCP iteration. The
        callable consumes the algorithm-state pytree plus an assembled
        :class:`SubproblemData`, performs one ``assemble → solve → unpack``
        cycle on the JAX boundary, and returns a :class:`SubproblemSolution`.

        Replaces the four ``update_*`` methods + :meth:`solve` for the
        JAX-pure SCP path. All backends share the same input/output pytree
        shape so the SCP loop downstream can wrap the callback in
        ``lax.while_loop`` and compose with ``jax.jit`` / ``jax.vmap``.
        """
        raise NotImplementedError

    @property
    def exportable(self) -> bool:
        """Whether this backend's solve survives ``jax.export`` serialization.

        :meth:`~openscvx.problem.Problem.solve_batched` exports the whole
        vmapped SCP loop to disk under ``save_compiled=True``, which requires
        every op in the loop — the backend solve included — to lower to
        serializable XLA. The pure-JAX backends (QPAX, Moreau) qualify, so the
        base default is ``True``. CVXPy's :meth:`iteration_callback` is a
        :func:`jax.pure_callback`, whose host call ``jax.export`` cannot
        serialize, so it overrides this to ``False`` and ``solve_batched``
        raises a teaching error rather than silently degrading to an in-process
        solve.
        """
        return True

    def _hash_into(self, hasher: "hashlib._Hash") -> None:
        """Contribute this backend's identity to the ``solve_batched`` cache key.

        The exported batched loop bakes in the backend class and its
        ``solver_args`` (tolerances, iteration caps, ...), neither of which the
        symbolic problem hash covers. Mirrors the symbolic ``_hash_into``
        protocol (see :meth:`~openscvx.symbolic.expr.expr.Expr._hash_into`) so a
        backend that adds a field affecting the artifact updates the hash right
        next to that field.
        """
        from openscvx.utils.caching import hash_value_into

        hasher.update(type(self).__name__.encode())
        for key in sorted(self.solver_args or {}):
            hasher.update(str(key).encode())
            hash_value_into(hasher, self.solver_args[key])

    @staticmethod
    def _extract_impulsive_pins(
        constraints: "ConstraintSet",
    ) -> Optional[List[Tuple[List[int], slice]]]:
        """Recognize auto-generated impulsive zero-pin constraints.

        :func:`openscvx.symbolic.lower._augment_impulsive_constraints` injects
        a ``Control == 0`` equality at every non-impulse node for each
        impulsive control. CVXPy lowers these alongside user ``.convex()``
        constraints, but JAX backends that otherwise refuse user
        ``.convex()`` constraints still need to honor them.

        This helper detects that exact shape — a ``NodalConstraint`` wrapping
        ``Equality(Control, Constant(0))`` over a list of nodes — and
        returns the implied ``(nodes, slice)`` pin list. If any
        ``nodal_convex`` entry doesn't match the auto-augmentation shape,
        returns ``None`` so the caller can fall back to the default
        refusal.

        Cross-node convex constraints are never produced by the
        auto-augmentation, so any presence aborts recognition.
        """
        from openscvx.symbolic.expr.constraint import Equality, NodalConstraint
        from openscvx.symbolic.expr.control import Control
        from openscvx.symbolic.expr.expr import Constant

        if constraints.cross_node_convex:
            return None

        pins: List[Tuple[List[int], slice]] = []
        for entry in constraints.nodal_convex:
            if not isinstance(entry, NodalConstraint):
                return None
            inner = entry.constraint
            if not isinstance(inner, Equality):
                return None
            rhs = inner.rhs
            if not isinstance(rhs, Constant) or not np.all(np.asarray(rhs.value) == 0):
                return None
            lhs = inner.lhs
            if not isinstance(lhs, Control) or lhs._slice is None:
                return None
            pins.append(([int(k) for k in entry.nodes], lhs._slice))
        return pins

    @staticmethod
    def _scaling(unified: Union["UnifiedState", "UnifiedControl"]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the affine scaling matrices ``(S, c)`` for a unified
        state or control interface.

        The PTR formulation works on scaled variables ``z = S⁻¹ (x - c)``;
        ``S`` and ``c`` are chosen so the scaled variables sit roughly on
        ``[-1, 1]``. Bounds come from ``scaling_min``/``scaling_max`` when
        provided, otherwise from ``min``/``max``. Shared by every backend.
        """
        from openscvx.config import get_affine_scaling_matrices

        n = len(unified.max)
        lower = np.array(
            unified.scaling_min if unified.scaling_min is not None else unified.min,
            dtype=float,
        )
        upper = np.array(
            unified.scaling_max if unified.scaling_max is not None else unified.max,
            dtype=float,
        )
        return get_affine_scaling_matrices(n, lower, upper)


# ---------------------------------------------------------------------------
# Shared base for the pure-JAX backends
# ---------------------------------------------------------------------------


class JaxPTRSolver(PTRSolver):
    """Shared plumbing for PTR backends whose subproblem solve is pure JAX.

    QPAX, Moreau and ICORE differ only in how they assemble, dispatch and
    unpack the subproblem. This class owns everything else they have in
    common:

    * named-argument / ``solver_args`` merging (:meth:`_merge_solver_args`);
    * the affine ``x = S z + c`` scaling, in NumPy and JAX form
      (:meth:`_setup_scaling`);
    * the supported-subset guards (:meth:`_reject_nodal_equalities`,
      :meth:`_check_initialize`);
    * the four NumPy ``update_*`` hooks, which stash their inputs (with
      ``D_d`` absorbed into ``A_d`` / ``B_d`` / ``C_d``) for :meth:`solve`;
    * packing that stash into a :class:`SubproblemData`
      (:meth:`_subproblem_data_from_stash`), so a backend can implement
      :meth:`solve` by running its own :meth:`iteration_callback`
      (:meth:`_solve_via_callback`);
    * vmapped linearization of user ``.convex()`` cone functions
      (:meth:`_linearize_at_nodes`).

    Subclasses provide ``layout`` (any object with ``N``, ``n_x``, ``n_u`` and
    ``n_nodal``), :meth:`create_variables`, :meth:`initialize`,
    :meth:`iteration_callback` and :meth:`solve`.
    """

    def __init__(self, solver_args: Dict[str, Any]):
        self.solver_args = solver_args

        self.layout: Any = None
        self._settings = None
        self._jax_constraints = None

        # Populated by PTRSolver.lower_convex_constraints.
        self._impulsive_pins: List[Tuple[List[int], slice]] = []
        self._user_cone_constraints: List[ConeConstraint] = []
        self._parameters_dict: Dict[str, Any] = {}

        # Per-iteration NumPy stash, written by the update_* hooks.
        self._dyn: dict = {}
        self._cons: dict = {}
        self._pen: dict = {}
        self._x_init: Optional[np.ndarray] = None
        self._x_term: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construction / setup helpers
    # ------------------------------------------------------------------

    def _merge_solver_args(
        self,
        named: Dict[str, Any],
        solver_args: Optional[Dict[str, Any]],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge ``defaults``, explicitly passed named arguments and ``solver_args``.

        ``None`` values in ``named`` mean "not passed". A key that is both
        passed by name and present in ``solver_args`` is ambiguous and raises.
        """
        named = {k: v for k, v in named.items() if v is not None}
        extra = dict(solver_args) if solver_args else {}
        overlap = named.keys() & extra.keys()
        if overlap:
            raise ValueError(
                f"{type(self).__name__} settings {sorted(overlap)} appear as both named "
                "arguments and inside solver_args; use one or the other."
            )
        return {**(defaults or {}), **named, **extra}

    def _setup_scaling(
        self, x_unified: "UnifiedState", u_unified: "UnifiedControl"
    ) -> Tuple[int, int]:
        """Store the affine scaling in NumPy and JAX form; return ``(n_x, n_u)``.

        The JAX copies use ``float64`` when ``jax_enable_x64`` is on and
        ``float32`` otherwise, so assembled arrays land in the dtype the
        backend kernel expects.
        """
        self._S_x, self._c_x = self._scaling(x_unified)
        self._S_u, self._c_u = self._scaling(u_unified)
        self._S_x_diag = np.diag(self._S_x)
        self._S_u_diag = np.diag(self._S_u)
        self._inv_S_x_diag = 1.0 / self._S_x_diag
        self._inv_S_u_diag = 1.0 / self._S_u_diag

        f = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        self._jax_dtype = f
        self._S_x_diag_j = jnp.asarray(self._S_x_diag, dtype=f)
        self._S_u_diag_j = jnp.asarray(self._S_u_diag, dtype=f)
        self._inv_S_x_diag_j = jnp.asarray(self._inv_S_x_diag, dtype=f)
        self._inv_S_u_diag_j = jnp.asarray(self._inv_S_u_diag, dtype=f)
        self._c_x_j = jnp.asarray(self._c_x, dtype=f)
        self._c_u_j = jnp.asarray(self._c_u, dtype=f)
        self._x_unified = x_unified
        return len(x_unified.max), len(u_unified.max)

    def _reject_nodal_equalities(self, jax_constraints: "LoweredJaxConstraints") -> None:
        """Raise if any nodal constraint is an L1-penalized equality."""
        if any(c.is_equality for c in jax_constraints.nodal):
            raise NotImplementedError(
                f"{type(self).__name__} does not support L1-penalized equality "
                "constraints. Use CVXPyPTRSolver."
            )

    def _check_initialize(self, lowered: "LoweredProblem") -> None:
        """Common :meth:`initialize` preconditions.

        Raises:
            RuntimeError: If :meth:`create_variables` was not called first.
            NotImplementedError: If cross-node constraints are present.
        """
        name = type(self).__name__
        if self.layout is None:
            raise RuntimeError(
                f"{name}.initialize() called before create_variables(). "
                "Call create_variables() first."
            )
        if lowered.jax_constraints.cross_node:
            raise NotImplementedError(
                f"{name} does not support cross-node constraints "
                f"({len(lowered.jax_constraints.cross_node)} defined). "
                "Use CVXPyPTRSolver."
            )

    def _check_callback_ready(self) -> None:
        """Raise unless :meth:`initialize` has run."""
        if self.layout is None or self._settings is None:
            raise RuntimeError(
                f"{type(self).__name__}.iteration_callback() requires initialize() to "
                "have been called first."
            )

    # ------------------------------------------------------------------
    # NumPy update_* hooks (stash for solve())
    # ------------------------------------------------------------------

    def update_dynamics_linearization(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        A_d: np.ndarray,
        B_d: np.ndarray,
        C_d: np.ndarray,
        x_prop: np.ndarray,
        x_prop_plus: Optional[np.ndarray] = None,
        D_d: Optional[np.ndarray] = None,
        E_d: Optional[np.ndarray] = None,
    ) -> None:
        A_eff = np.asarray(A_d, dtype=float)
        B_eff = np.asarray(B_d, dtype=float)
        C_eff = np.asarray(C_d, dtype=float)

        # Absorb the impulsive state Jacobian into the continuous step
        # matrices so the assembly keeps a single A_d·dx + B_d·du term (same
        # recipe as CVXPyPTRSolver.update_dynamics_linearization).
        if D_d is not None:
            D_arr = np.asarray(D_d, dtype=float)
            n_steps = A_eff.shape[0]
            if D_arr.ndim == 3 and D_arr.shape[0] in (n_steps, n_steps + 1):
                D_steps = D_arr[D_arr.shape[0] - n_steps :]
            else:
                raise ValueError(
                    "Unexpected D_d shape for dynamics update: "
                    f"{D_arr.shape}, expected "
                    f"{(n_steps + 1, A_eff.shape[1], A_eff.shape[2])} "
                    f"or {(n_steps, A_eff.shape[1], A_eff.shape[2])}."
                )
            A_eff = np.einsum("kij,kjl->kil", D_steps, A_eff)
            B_eff = np.einsum("kij,kjl->kil", D_steps, B_eff)
            C_eff = np.einsum("kij,kjl->kil", D_steps, C_eff)

        self._dyn = {
            "x_bar": np.asarray(x_bar, dtype=float),
            "u_bar": np.asarray(u_bar, dtype=float),
            "A_d": A_eff,
            "B_d": B_eff,
            "C_d": C_eff,
            "x_prop": np.asarray(x_prop, dtype=float),
            "x_prop_plus": (
                np.asarray(x_prop_plus, dtype=float) if x_prop_plus is not None else None
            ),
            "E_d": np.asarray(E_d, dtype=float) if E_d is not None else None,
        }

    def update_constraint_linearizations(
        self,
        nodal: Optional[List[dict]] = None,
        cross_node: Optional[List[dict]] = None,
    ) -> None:
        if cross_node:
            raise NotImplementedError(
                f"{type(self).__name__} received cross-node linearization data; "
                "cross-node constraints are not supported. Use CVXPyPTRSolver."
            )
        self._cons = {
            "nodal": [
                {
                    "g": np.asarray(d["g"], dtype=float),
                    "grad_g_x": np.asarray(d["grad_g_x"], dtype=float),
                    "grad_g_u": np.asarray(d["grad_g_u"], dtype=float),
                }
                for d in (nodal or [])
            ],
        }

    def update_penalties(
        self,
        lam_prox: np.ndarray,
        lam_cost: Union[float, np.ndarray],
        lam_vc: np.ndarray,
        lam_vb_nodal: np.ndarray,
        lam_vb_cross: np.ndarray,
    ) -> None:
        del lam_vb_cross  # cross-node constraints are rejected at initialize()
        self._pen = {
            "lam_prox": np.asarray(lam_prox, dtype=float),
            "lam_cost": np.asarray(lam_cost, dtype=float),
            "lam_vc": np.asarray(lam_vc, dtype=float),
            "lam_vb_nodal": np.asarray(lam_vb_nodal, dtype=float),
        }

    def update_boundary_conditions(
        self,
        x_init: Optional[np.ndarray] = None,
        x_term: Optional[np.ndarray] = None,
    ) -> None:
        if x_init is not None:
            self._x_init = np.asarray(x_init, dtype=float)
        if x_term is not None:
            self._x_term = np.asarray(x_term, dtype=float)

    # ------------------------------------------------------------------
    # Stash → SubproblemData, and solve() through the JAX callback
    # ------------------------------------------------------------------

    def _subproblem_data_from_stash(self) -> SubproblemData:
        """Pack the ``update_*`` stash into the :class:`SubproblemData` pytree.

        ``D_d`` was absorbed into ``A_d`` / ``B_d`` / ``C_d`` at update time,
        so the packed ``D_d`` is the identity.
        """
        if not (self._dyn and self._cons and self._pen):
            raise RuntimeError(
                f"{type(self).__name__}.solve() requires update_dynamics_linearization, "
                "update_constraint_linearizations, and update_penalties to all "
                "have been called this iteration."
            )
        L = self.layout
        N, n_x, n_u, C = L.N, L.n_x, L.n_u, L.n_nodal
        dyn, pen = self._dyn, self._pen

        # Nodal linearizations stacked to (N, C, ...), zero outside each
        # constraint's node set.
        nodal_g = np.zeros((N, C))
        nodal_grad_x = np.zeros((N, C, n_x))
        nodal_grad_u = np.zeros((N, C, n_u))
        for c, (constraint, entry) in enumerate(
            zip(self._jax_constraints.nodal, self._cons["nodal"])
        ):
            nodes = list(constraint.nodes)
            nodal_g[nodes, c] = entry["g"][nodes]
            nodal_grad_x[nodes, c] = entry["grad_g_x"][nodes]
            nodal_grad_u[nodes, c] = entry["grad_g_u"][nodes]

        def _or(value, shape, fill=0.0):
            return value if value is not None else np.full(shape, fill)

        return SubproblemData(
            x_bar=jnp.asarray(dyn["x_bar"]),
            u_bar=jnp.asarray(dyn["u_bar"]),
            A_d=jnp.asarray(dyn["A_d"]),
            B_d=jnp.asarray(dyn["B_d"]),
            C_d=jnp.asarray(dyn["C_d"]),
            x_prop=jnp.asarray(dyn["x_prop"]),
            x_prop_plus=jnp.asarray(_or(dyn["x_prop_plus"], (N, n_x))),
            D_d=jnp.broadcast_to(jnp.eye(n_x), (N, n_x, n_x)),
            E_d=jnp.asarray(_or(dyn["E_d"], (N, n_x, n_u))),
            nodal_g=jnp.asarray(nodal_g),
            nodal_grad_x=jnp.asarray(nodal_grad_x),
            nodal_grad_u=jnp.asarray(nodal_grad_u),
            cross_g=jnp.zeros((0,)),
            cross_grad_X=jnp.zeros((0, N, n_x)),
            cross_grad_U=jnp.zeros((0, N, n_u)),
            lam_prox=jnp.asarray(pen["lam_prox"]),
            lam_cost=jnp.asarray(pen["lam_cost"]),
            lam_vc=jnp.asarray(pen["lam_vc"]),
            lam_vb_nodal=jnp.asarray(pen["lam_vb_nodal"]),
            lam_vb_cross=jnp.zeros((0,)),
            x_init=jnp.asarray(_or(self._x_init, n_x, np.nan)),
            x_term=jnp.asarray(_or(self._x_term, n_x, np.nan)),
            params=self._parameters_dict,
        )

    def _solve_via_callback(self) -> PTRSolveResult:
        """Implement :meth:`solve` by running :meth:`iteration_callback` on the stash.

        Lets a backend share one assembly between the NumPy and JAX-pure
        entry points instead of maintaining a separate NumPy path.
        """
        sol = self.iteration_callback()(None, self._subproblem_data_from_stash())
        ok = int(sol.status_code) == int(StatusCode.OPTIMAL)
        nu_vb = np.asarray(sol.nu_vb)
        return PTRSolveResult(
            x=np.asarray(sol.x),
            u=np.asarray(sol.u),
            nu=np.asarray(sol.nu),
            nu_vb=list(nu_vb.T),
            nu_vb_cross=[],
            cost=float(sol.cost),
            status="optimal" if ok else "infeasible",
        )

    # ------------------------------------------------------------------
    # User cone linearization
    # ------------------------------------------------------------------

    def _linearize_at_nodes(
        self, fn: Callable, nodes: Tuple[int, ...], data: SubproblemData, rows: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Value and *scaled* Jacobians of a per-node function at ``nodes``.

        ``fn(x, u, node, params)`` is evaluated and differentiated (one
        forward-mode pass for both arguments) at every node in a single
        ``jax.vmap``, instead of one trace per node. Jacobians are
        right-multiplied by ``S_x`` / ``S_u`` so they act on scaled variables.

        Returns:
            ``(val, J_x, J_u)`` with shapes ``(K, rows)``, ``(K, rows, n_x)``
            and ``(K, rows, n_u)`` for ``K = len(nodes)``.
        """
        idx = np.asarray(nodes, dtype=np.int32)
        params = data.params

        def at_node(x, u, k):
            f = lambda x_, u_: jnp.reshape(fn(x_, u_, k, params), (rows,))  # noqa: E731
            J_x, J_u = jax.jacfwd(f, argnums=(0, 1))(x, u)
            return f(x, u), J_x, J_u

        val, J_x, J_u = jax.vmap(at_node)(data.x_bar[idx], data.u_bar[idx], jnp.asarray(idx))
        return (
            val,
            J_x * self._S_x_diag_j[None, None, :],
            J_u * self._S_u_diag_j[None, None, :],
        )
