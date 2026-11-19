"""Riccati-structured conic backend for the PTR convex subproblem.

Assembles each SCP subproblem as a *stage-wise* conic optimal control problem
and dispatches it to ICORE (https://github.com/sametuzun781/ICORE), a
Mehrotra predictor-corrector interior-point method whose Newton systems are
solved by a Riccati recursion instead of a general sparse factorization. This
backend sits alongside
:class:`openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver` (general sparse
conic program) and :class:`openscvx.solvers.qpax_ptr_solver.QPAXPTRSolver`
(dense QP), and shares their plumbing through
:class:`~openscvx.solvers.ptr_solver.JaxPTRSolver`.

ICORE's JAX kernel (``icore.jax.core.make_solver``) is a pure ``jax.jit``
function of the stage arrays, so :meth:`ICOREPTRSolver.iteration_callback`
composes with ``jax.jit`` / ``jax.vmap`` exactly like the Moreau backend.

ICORE problem form:
    For stages ``k = 0, ..., K-1`` with stage variable ``w_k = [v_k, x̃_k]``
    (control first)::

        min  Σ_k 0.5 w_kᵀ H_k w_k + c_kᵀ w_k  (+ terminal 0.5 x̃_Kᵀ H_K x̃_K + c_Kᵀ x̃_K)
        s.t. x̃_{k+1} = A_k x̃_k + B_k v_k + b_k,   x̃_0 = x0  (fully fixed)
             h_k − G_k w_k ∈ 𝒦                    (same cone layout every stage)
             F x̃_K = f                            (optional terminal equality)

    where ``𝒦`` is a product of nonnegative orthants followed by second-order
    cones ``(radius, vector)``.

Mapping the PTR subproblem onto that form:
    The PTR subproblem does not fit the stage form directly. First-order-hold
    dynamics couple ``u_{k+1}`` into ``x_{k+1}``, the initial state is only
    partly fixed, and several constraints are equalities, which ICORE's stage
    cones cannot express. The assembly closes each gap *exactly*, without
    relaxations:

    * **Augmented state** ``x̃ = [x_k, u_k, t_vb_k]`` (scaled ``x`` / ``u``
      plus one virtual-buffer epigraph per nodal constraint). The stage
      control ``v_k = [t_vc_k, u_{k+1}, t_vb_{k+1}, ν_k]`` holds the
      ``|ν|`` epigraph ``t_vc_k``, the next control and buffer, and the
      dynamics virtual control ``ν_k``. FOH dynamics become
      ``x_{k+1} = A(x_k − x̄_k) + B(u_k − ū_k) + C(u_{k+1} − ū_{k+1}) + ν_k + r_k``,
      which is affine in ``(x̃_k, v_k)``.
    * **Epigraph controls.** ``t_vc`` comes first in ``v`` and appears only in
      the two rows ``±ν − t_vc ≤ 0``, which is exactly ICORE's
      ``epigraph_controls`` contract. ICORE then eliminates the ``n_x``
      ``t_vc`` slots analytically instead of carrying them through the
      Riccati recursion.
    * **Pre-stage.** ICORE fixes ``x̃_0`` entirely, so stage 0 starts from a
      dummy zero state and its control chooses the free components of
      ``x̃_1 = [x_0, u_0, t_vb_0]``. Free initial states enter as
      ``x_0 = x̄_0 + ν_0``. Fixed initial components (``Fix`` boundary
      conditions, CTCS augmented states, and the impulsive post-jump state)
      enter through ``b_0``.
    * **Equalities are eliminated through the dynamics.** Uniform time grid
      (``u_{k+1}[td] = u_k[td]``) and impulsive zero pins
      (``u_k[j] = const``) are wired into the ``u`` rows of ``A_k / B_k /
      b_k``. Terminal ``Fix`` conditions use ICORE's ``F x̃_K = f``.
    * **Uniform cone layout.** Every stage carries the same rows. Rows that are
      inactive at a stage (CTCS intervals, nodal/user constraints restricted
      to a node subset, controls at the terminal stage) are padded with the
      trivially-satisfied ``0 ≤ 1`` (orthant) or ``(1, 0, …)`` (SOC). Box
      rows on structurally fixed components are padded too: a fixed value
      sitting exactly on its bound would otherwise produce a zero-slack row
      with no strict interior.
    * Penalties: ``λ_vc t_vc`` with ``|ν| ≤ t_vc``; ``λ_vb t_vb`` with
      ``g + ∇g·Δ ≤ t_vb``, ``t_vb ≥ 0``. CTCS difference rows
      ``|x_{k+1} − x_k|`` substitute ``x_{k+1}`` from the stage dynamics so
      they stay stage-local.

Scope:
    Supported: state/control box, dynamics linearization (continuous and
    impulsive), boundary Fix / Minimize / Maximize, uniform time grid,
    linearized nodal nonconvex inequalities, CTCS rows, and user
    ``.convex()`` affine inequality and second-order-cone constraints.

    Not supported: cross-node constraints, L1-penalized nodal equalities, and
    user ``.convex()`` equality (zero-cone) constraints. ICORE has no
    stage-wise equality cone. All of these raise :class:`NotImplementedError`
    pointing at :class:`~openscvx.solvers.cvxpy_ptr_solver.CVXPyPTRSolver`
    or :class:`~openscvx.solvers.moreau_ptr_solver.MoreauPTRSolver`.

Precision:
    ICORE's kernel is float64-only. Pass ``float_dtype="float64"`` to
    :class:`openscvx.problem.Problem`; :meth:`ICOREPTRSolver.create_variables`
    raises otherwise.

Note:
    ``icore.jax.core`` enables ``jax_enable_x64`` as an import side effect.
    This module restores the caller's setting immediately after importing it,
    so loading OpenSCvx never silently changes the global JAX dtype.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .cones import NonnegConeConstraint, SOCConstraint
from .ptr_solver import (
    JaxPTRSolver,
    PTRSolveResult,
    StatusCode,
    SubproblemData,
    SubproblemSolution,
)

_x64_before_import = jax.config.read("jax_enable_x64")
try:
    from icore.jax.core import make_solver as _icore_make_solver

    _ICORE_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by the install-error test
    _icore_make_solver = None  # type: ignore[assignment]
    _ICORE_AVAILABLE = False
finally:
    # icore.jax.core flips jax_enable_x64 on at import time; undo that so the
    # global dtype stays under Problem(float_dtype=...)'s control.
    jax.config.update("jax_enable_x64", _x64_before_import)

if TYPE_CHECKING:
    from openscvx.config import Config
    from openscvx.lowered import LoweredProblem
    from openscvx.lowered.jax_constraints import LoweredJaxConstraints
    from openscvx.lowered.unified import UnifiedControl, UnifiedState

_INSTALL_HINT = (
    "ICOREPTRSolver requires the `icore` package "
    "(https://github.com/sametuzun781/ICORE). ICORE is not yet published on "
    "PyPI; clone the repository and add it to PYTHONPATH."
)

# Tiny diagonal regularisation on every non-epigraph decision slot (matches
# Moreau's _P_DIAG_EPS). Keeps the Riccati control blocks positive definite
# when a trust-region weight is zero.
_DIAG_EPS = 1e-10

# Hessian placed on control slots that no dynamics row or cone row touches
# (pre-stage padding, pinned / time-grid-copied controls). Such a slot is
# fully decoupled, so any positive weight drives it to exactly zero; using
# 1.0 instead of _DIAG_EPS keeps ICORE's Cholesky well conditioned.
_UNUSED_SLOT_HESS = 1.0

# Row padding for inactive orthant rows: h − G w = 1 > 0 with G = 0.
_PAD_H = 1.0

# Native ICORE settings accepted in ``solver_args`` and their defaults (the
# same defaults as ``icore.Solver``).
_ICORE_DEFAULTS: Dict[str, Any] = {
    "tolerance": 1e-8,
    "max_iterations": 200,
    "refinement_steps": 5,
    "regularization": 1e-11,
    "step_fraction": 0.99,
    "refinement_tolerance": 0.0,
    "riccati_mode": 0,
    "step_policy": "auto",
}
_STEP_POLICIES = {"auto": 0, "fixed": 1, "adaptive": 2}


def _icore_status_to_code(status: "jnp.ndarray") -> "jnp.ndarray":
    """Map ICORE's integer status to a :class:`StatusCode`.

    ICORE reports ``0`` (solved), ``1`` (iteration limit), ``2`` (numerical
    failure) or ``3`` (invalid model). It has no infeasibility certificate,
    so every non-zero status maps to :attr:`StatusCode.UNKNOWN`.
    """
    return jnp.where(
        jnp.asarray(status) == 0,
        jnp.int32(int(StatusCode.OPTIMAL)),
        jnp.int32(int(StatusCode.UNKNOWN)),
    )


# ---------------------------------------------------------------------------
# Stage layout
# ---------------------------------------------------------------------------


@dataclass
class _StageLayout:
    """Static slot layout of ICORE's stage variable ``w = [v, x̃]``.

    Attributes:
        N: Number of PTR discretization nodes. ICORE runs ``N`` transitions
            (one pre-stage plus ``N-1`` PTR intervals) over ``N+1`` stages.
            Stage ``i ≥ 1`` holds PTR node ``i-1``.
        n_x: PTR state dimension.
        n_u: PTR control dimension.
        n_nodal: Number of nodal nonconvex constraints.
        n_xa: Augmented state width ``n_x + n_u + n_nodal``.
        n_v: Stage control width ``2 n_x + n_u + n_nodal``.
        n_w: Stage variable width ``n_v + n_xa``.
    """

    N: int
    n_x: int
    n_u: int
    n_nodal: int

    # Augmented-state slots, in x̃ coordinates (rows of A / B / b).
    sx: slice = field(init=False)
    su: slice = field(init=False)
    stvb: slice = field(init=False)
    # Control slots, in v coordinates (also the first n_v columns of w).
    # t_vc must come first: ICORE's epigraph_controls are the leading controls.
    v_tvc: slice = field(init=False)
    v_u: slice = field(init=False)
    v_tvb: slice = field(init=False)
    v_nu: slice = field(init=False)
    # Augmented-state slots, in w coordinates (offset by n_v).
    w_x: slice = field(init=False)
    w_u: slice = field(init=False)
    w_tvb: slice = field(init=False)

    n_xa: int = field(init=False)
    n_v: int = field(init=False)
    n_w: int = field(init=False)

    def __post_init__(self):
        n_x, n_u, C = self.n_x, self.n_u, self.n_nodal
        self.n_xa = n_x + n_u + C
        self.n_v = 2 * n_x + n_u + C
        self.n_w = self.n_v + self.n_xa

        self.sx = slice(0, n_x)
        self.su = slice(n_x, n_x + n_u)
        self.stvb = slice(n_x + n_u, self.n_xa)

        self.v_tvc = slice(0, n_x)
        self.v_u = slice(n_x, n_x + n_u)
        self.v_tvb = slice(n_x + n_u, n_x + n_u + C)
        self.v_nu = slice(n_x + n_u + C, self.n_v)

        o = self.n_v
        self.w_x = slice(o, o + n_x)
        self.w_u = slice(o + n_x, o + n_x + n_u)
        self.w_tvb = slice(o + n_x + n_u, self.n_w)


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------


class ICOREPTRSolver(JaxPTRSolver):
    """Riccati-structured conic backend for the PTR convex subproblem.

    Recasts each SCP subproblem as a stage-wise conic OCP and solves it with
    ICORE's JAX interior-point kernel. See the module docstring for the
    augmented-state mapping.

    Because the solve is a pure ``jax.jit`` function, the backend composes
    with ``jax.jit`` / ``jax.vmap`` and is exportable via ``jax.export``.
    Each solve starts cold (ICORE has no warm start).

    Class attributes:
        SUPPORTED_CONE_TYPES: Accepts
            :class:`~openscvx.solvers.cones.NonnegConeConstraint` and
            :class:`~openscvx.solvers.cones.SOCConstraint`. Zero-cone
            (equality) user constraints are not supported because ICORE has
            no stage-wise equality cone.

    Args:
        solver_tol: Absolute KKT tolerance (ICORE ``tolerance``). ICORE stops
            when the max of primal residual, dual residual and complementarity
            gap falls below it. Defaults to ``1e-8``.
        max_iter: Maximum IPM iterations (ICORE ``max_iterations``).
            Defaults to ``200``.
        solver_args: Further ICORE settings, using ICORE's native names:
            ``refinement_steps``, ``regularization``, ``step_fraction``,
            ``refinement_tolerance``, ``riccati_mode`` (0 auto, 1 classical,
            2 square-root) and ``step_policy`` (``"auto"``, ``"fixed"``,
            ``"adaptive"``). ``tolerance`` / ``max_iterations`` are also
            accepted but conflict with the named arguments above.

    Attributes:
        layout: :class:`_StageLayout` describing ICORE's stage variable.
            Populated by :meth:`create_variables`.
    """

    SUPPORTED_CONE_TYPES = frozenset({NonnegConeConstraint, SOCConstraint})

    def __init__(
        self,
        *,
        solver_tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        solver_args: Optional[Dict[str, Any]] = None,
    ):
        if not _ICORE_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        unknown = (solver_args or {}).keys() - _ICORE_DEFAULTS.keys()
        if unknown:
            raise ValueError(
                f"Unknown ICORE settings {sorted(unknown)} in solver_args; "
                f"valid keys are {sorted(_ICORE_DEFAULTS)}."
            )
        merged = self._merge_solver_args(
            {"tolerance": solver_tol, "max_iterations": max_iter}, solver_args, _ICORE_DEFAULTS
        )
        policy = merged["step_policy"]
        if isinstance(policy, str) and policy not in _STEP_POLICIES:
            raise ValueError(
                f"step_policy must be one of {sorted(_STEP_POLICIES)}; got {policy!r}."
            )
        super().__init__(merged)

        self._static: Dict[str, Any] = {}
        self._solve_fn = None
        self._callback = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def create_variables(
        self,
        N: int,
        x_unified: "UnifiedState",
        u_unified: "UnifiedControl",
        jax_constraints: "LoweredJaxConstraints",
        dynamics_sparsity: Optional[tuple] = None,
        constraint_sparsity: Optional[list] = None,
    ) -> None:
        """Compute scaling and the static stage layout.

        Sparsity hints are accepted for interface symmetry and ignored: ICORE
        works with dense per-stage blocks.

        Raises:
            ValueError: If JAX is not running in float64.
            NotImplementedError: If any nodal constraint is an L1-penalized
                equality.
        """
        del dynamics_sparsity, constraint_sparsity
        if not jax.config.read("jax_enable_x64"):
            raise ValueError(
                "ICOREPTRSolver requires float64: ICORE's interior-point kernel "
                "runs in double precision only. Pass float_dtype='float64' to Problem."
            )
        self._reject_nodal_equalities(jax_constraints)
        n_x, n_u = self._setup_scaling(x_unified, u_unified)
        self.layout = _StageLayout(N=N, n_x=n_x, n_u=n_u, n_nodal=len(jax_constraints.nodal))
        self._jax_constraints = jax_constraints

    def initialize(self, lowered: "LoweredProblem", settings: "Config") -> None:
        """Precompute every static stage block and build ICORE's jitted kernel.

        Raises:
            RuntimeError: If :meth:`create_variables` was not called first.
            NotImplementedError: If cross-node constraints are present.
        """
        self._check_initialize(lowered)
        self._settings = settings
        self._static = self._build_static(settings)

        args = dict(self.solver_args)
        policy = args["step_policy"]
        args["step_policy"] = _STEP_POLICIES[policy] if isinstance(policy, str) else int(policy)
        self._solve_fn = _icore_make_solver(
            self._static["n_orth"],
            self._static["soc_sizes"],
            epigraph_controls=self.layout.n_x,
            **args,
        )
        self._callback = None

    # ------------------------------------------------------------------
    # Static structure
    # ------------------------------------------------------------------

    def _build_static(self, settings: "Config") -> Dict[str, Any]:
        """Assemble everything that does not depend on the SCP iterate.

        Returns a dict of NumPy arrays and Python metadata closed over by the
        JAX assembly: boundary-condition classification, control maps for
        pins / time grid, the always-static constraint rows (boxes,
        ``t_vb ≥ 0``, ``|ν| ≤ t_vc``), active-stage masks for the
        iterate-dependent rows, and the cone layout.
        """
        L = self.layout
        sim = settings.sim
        N, n_x, n_u, C, n_w = L.N, L.n_x, L.n_u, L.n_nodal, L.n_w
        S_x, S_u, c_x, c_u = self._S_x_diag, self._S_u_diag, self._c_x, self._c_u
        true_x = np.arange(sim.true_state_slice.start, sim.true_state_slice.stop)

        slice_imp = sim.u.slice_impulsive
        has_impulsive = slice_imp.stop > slice_imp.start
        imp_mask = np.zeros(n_u)
        imp_mask[slice_imp] = 1.0

        # ---- Boundary classification -----------------------------------
        # Initial components are free (chosen by the pre-stage), Fix (x_init,
        # or the post-jump state coupled to u_0 when impulsive), or CTCS
        # (zero integral at node 0).
        init_fix = np.zeros(n_x, dtype=bool)
        init_fix[true_x] = np.asarray(sim.x.initial_type, dtype=object)[true_x] == "Fix"
        ctcs0 = np.zeros(n_x, dtype=bool)
        ctcs0[sim.ctcs_slice] = True
        free0 = ~init_fix & ~ctcs0
        fixN = np.zeros(n_x, dtype=bool)
        fixN[true_x] = np.asarray(sim.x.final_type, dtype=object)[true_x] == "Fix"

        sign = {"Minimize": 1.0, "Maximize": -1.0}
        init_sign = np.zeros(n_x)
        final_sign = np.zeros(n_x)
        init_sign[true_x] = [sign.get(sim.x.initial_type[i], 0.0) for i in true_x]
        final_sign[true_x] = [sign.get(sim.x.final_type[i], 0.0) for i in true_x]

        # ---- Control maps: u_k = P_k ⊙ v_u + U_k ⊙ u_{k-1} + p_k ----------
        pinned = np.zeros((N, n_u), dtype=bool)
        for nodes, ctrl_slice in self._impulsive_pins:
            pinned[list(nodes), ctrl_slice] = True
        copy_prev = np.zeros((N, n_u), dtype=bool)
        if sim._uniform_time_grid:
            copy_prev[1:, sim.time_dilation_slice] = True
        copy_prev &= ~pinned
        P_map = (~pinned & ~copy_prev).astype(float)
        U_map = copy_prev.astype(float)
        p_map = np.where(pinned, (-c_u / S_u)[None, :], 0.0)  # scaled physical zero

        # ---- Static orthant rows -----------------------------------------
        stage = np.arange(1, N + 1)[:, None]  # stage of node k is k + 1

        def block(rows):
            return np.zeros((N + 1, rows, n_w)), np.full((N + 1, rows), _PAD_H)

        def box(on, col0, hi, lo):
            """Two-sided box rows ``lo ≤ w[col0 + j] ≤ hi`` where ``on[k, j]``."""
            n = on.shape[1]
            G, h = block(2 * n)
            k, j = np.nonzero(on)
            G[k + 1, j, col0 + j] = 1.0
            h[k + 1, j] = hi[j]
            G[k + 1, n + j, col0 + j] = -1.0
            h[k + 1, n + j] = -lo[j]
            return G, h

        # Box rows on components with no freedom (fixed initial states other
        # than the impulsive post-jump state, terminal Fix states, pinned
        # controls) are padded: a fixed value at its bound would give a
        # zero-slack row with no strict interior.
        x_box_on = np.ones((N, n_x), dtype=bool)
        x_box_on[0] = free0 | (init_fix & has_impulsive)
        x_box_on[N - 1] &= ~fixN
        x_max = (np.asarray(sim.x.max, dtype=float) - c_x) / S_x
        x_min = (np.asarray(sim.x.min, dtype=float) - c_x) / S_x
        u_max = (np.asarray(sim.u.max, dtype=float) - c_u) / S_u
        u_min = (np.asarray(sim.u.min, dtype=float) - c_u) / S_u

        blocks = [
            box(x_box_on, L.w_x.start, x_max, x_min),
            box(~pinned, L.w_u.start, u_max, u_min),
        ]
        # Nodal linearized rows are iterate-dependent; reserve their slot.
        nodal_row0 = 2 * n_x + 2 * n_u
        blocks.append(block(C))

        # t_vb ≥ 0 at every node.
        G, h = block(C)
        cc = np.arange(C)
        G[stage, cc, L.w_tvb.start + cc] = -1.0
        h[1:] = 0.0
        blocks.append((G, h))

        # Epigraph pairs ±ν_j − t_vc_j ≤ 0 at every transition stage 0..N-1
        # (ICORE's epigraph_controls contract).
        G, h = block(2 * n_x)
        s = np.arange(N)[:, None]
        jj = np.arange(n_x)
        G[s, jj, L.v_nu.start + jj] = 1.0
        G[s, n_x + jj, L.v_nu.start + jj] = -1.0
        G[s, jj, L.v_tvc.start + jj] = -1.0
        G[s, n_x + jj, L.v_tvc.start + jj] = -1.0
        h[:N] = 0.0
        blocks.append((G, h))

        G_static = np.concatenate([b[0] for b in blocks], axis=1)
        h_static = np.concatenate([b[1] for b in blocks], axis=1)

        # ---- Iterate-dependent rows: active-stage masks ------------------
        nodal_active = np.zeros((N, C), dtype=bool)
        for c, constraint in enumerate(self._jax_constraints.nodal):
            nodal_active[list(constraint.nodes), c] = True

        # CTCS rows: stage s ↔ transition into node s, active on [start, end).
        ctcs_idx = np.arange(sim.ctcs_slice.start, sim.ctcs_slice.stop)
        ctcs_active = np.zeros((N, ctcs_idx.size), dtype=bool)
        for g, (n0, n1) in enumerate(sim.ctcs_node_intervals):
            ctcs_active[max(n0, 1) : n1, g] = True

        nonneg = [c for c in self._user_cone_constraints if isinstance(c, NonnegConeConstraint)]
        socs = [c for c in self._user_cone_constraints if isinstance(c, SOCConstraint)]
        n_orth = G_static.shape[1] + 2 * ctcs_idx.size + sum(int(c.m) for c in nonneg)

        # ---- Static cost pieces -----------------------------------------
        # Hessian diagonal: _DIAG_EPS everywhere, 1.0 on control slots no
        # dynamics / cone row touches, 0 on epigraph slots (ICORE requires
        # it) and on the terminal stage's control columns (it has none).
        H_base = np.full((N + 1, n_w), _DIAG_EPS)
        H_base[0, L.v_nu][~free0] = _UNUSED_SLOT_HESS
        H_base[:N, L.v_u][P_map == 0.0] = _UNUSED_SLOT_HESS
        H_base[:, L.v_tvc] = 0.0
        H_base[N, : L.n_v] = 0.0

        F = np.zeros((int(fixN.sum()), L.n_xa))
        F[np.arange(F.shape[0]), L.sx.start + np.flatnonzero(fixN)] = 1.0

        return {
            "has_impulsive": has_impulsive,
            "imp_mask": imp_mask,
            "free0": np.flatnonzero(free0),
            "fix0": np.flatnonzero(init_fix & ~has_impulsive),
            "imp0": np.flatnonzero(init_fix & has_impulsive),
            "ctcs0": np.flatnonzero(ctcs0),
            "fixN": np.flatnonzero(fixN),
            "F": F,
            "init_sign": init_sign,
            "final_sign": final_sign,
            "P_map": P_map,
            "U_map": U_map,
            "p_map": p_map,
            "H_base": H_base,
            "G_static": G_static,
            "h_static": h_static,
            "nodal_row0": nodal_row0,
            "nodal_active": nodal_active,
            "ctcs_idx": ctcs_idx,
            "ctcs_active": ctcs_active,
            "ctcs_x_max": np.asarray(sim.x.max, dtype=float)[ctcs_idx],
            "nonneg": nonneg,
            "socs": socs,
            "n_orth": int(n_orth),
            "soc_sizes": tuple(int(c.m_arg) + 1 for c in socs),
        }

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> PTRSolveResult:
        """Solve the subproblem described by the ``update_*`` stash.

        Runs the same jitted :meth:`iteration_callback` the SCP loop uses, so
        both entry points share one assembly.
        """
        return self._solve_via_callback()

    def iteration_callback(self):
        """JAX-pure ``(state, SubproblemData) -> SubproblemSolution``.

        Composes :meth:`_assemble_ocp_jax`, ICORE's jitted kernel and
        :meth:`_build_solution_jax` into a single ``jax.jit`` closure, built
        once and cached. ICORE has no warm start, so ``state`` is accepted
        only for cross-backend signature uniformity.
        """
        self._check_callback_ready()
        if self._callback is None:
            solve_fn = self._solve_fn

            def step(state, data: SubproblemData) -> SubproblemSolution:
                del state  # ICORE solves cold.
                result = solve_fn(self._assemble_ocp_jax(data))
                return self._build_solution_jax(result, data)

            self._callback = jax.jit(step)
        return self._callback

    # ------------------------------------------------------------------
    # JAX-pure assembly
    # ------------------------------------------------------------------

    def _scaled_reference(self, data: SubproblemData):
        """Scaled linearization point ``(x̄, ū)``, shapes ``(N, n_x)``, ``(N, n_u)``."""
        xs = self._inv_S_x_diag_j * (data.x_bar - self._c_x_j)
        us = self._inv_S_u_diag_j * (data.u_bar - self._c_u_j)
        return xs, us

    def _dynamics_blocks(self, data: SubproblemData):
        """Scaled PTR step matrices ``(A, B, C', r)`` for transitions k → k+1.

        ``C'`` folds the impulsive ``E_d`` into the next-node control column
        and ``r`` is the scaled propagated state, both mirroring
        :meth:`MoreauPTRSolver._assemble_conic_jax`.
        """
        st = self._static
        S_x, S_u = self._S_x_diag_j, self._S_u_diag_j
        inv_S_x = self._inv_S_x_diag_j[:, None]
        A_d, B_d, C_d = data.A_d, data.B_d, data.C_d

        if st["has_impulsive"]:
            D = data.D_d[1:]
            A_d, B_d, C_d = (jnp.einsum("kij,kjl->kil", D, M) for M in (A_d, B_d, C_d))
            C_d = C_d + data.E_d[1:] * st["imp_mask"]
            x_next = data.x_prop_plus[1:]
        else:
            x_next = data.x_prop
        return (
            inv_S_x * A_d * S_x,
            inv_S_x * B_d * S_u,
            inv_S_x * C_d * S_u,
            self._inv_S_x_diag_j * (x_next - self._c_x_j),
        )

    def _assemble_ocp_jax(self, data: SubproblemData) -> Dict[str, "jnp.ndarray"]:
        """Build ICORE's stage arrays ``H, c, A, B, b, G, h, x0, F, f``."""
        L = self.layout
        st = self._static
        N, n_x, C = L.N, L.n_x, L.n_nodal
        n_xa, n_v, n_w = L.n_xa, L.n_v, L.n_w
        f64 = jnp.float64
        S_x, S_u = self._S_x_diag_j, self._S_u_diag_j
        inv_S_x, c_x = self._inv_S_x_diag_j, self._c_x_j
        xs, us = self._scaled_reference(data)
        P_map, U_map, p_map = st["P_map"], st["U_map"], st["p_map"]
        diag = jax.vmap(jnp.diag)
        bmv = lambda M, v: jnp.einsum("kij,kj->ki", M, v)  # noqa: E731

        # ================= Dynamics =================
        # Pre-stage (k = 0): x̃_1 = B_0 v_0 + b_0 from a zero dummy state.
        B0 = jnp.zeros((n_xa, n_v), dtype=f64)
        b0 = jnp.zeros((n_xa,), dtype=f64)
        free0, fix0, ctcs0, imp0 = st["free0"], st["fix0"], st["ctcs0"], st["imp0"]
        B0 = B0.at[free0, L.v_nu.start + free0].set(1.0)
        b0 = b0.at[free0].set(xs[0, free0])  # x_0 = x̄_0 + ν_0
        b0 = b0.at[fix0].set(inv_S_x[fix0] * (data.x_init[fix0] - c_x[fix0]))
        b0 = b0.at[ctcs0].set(-c_x[ctcs0] * inv_S_x[ctcs0])
        if imp0.size:
            # x_0[i] = inv_S_x(x⁺_0 − c_x) + E0s·(u_0 − ū_0), u_0 = P_0 v + p_0.
            E0s = inv_S_x[imp0, None] * data.E_d[0, imp0] * S_u * st["imp_mask"]
            B0 = B0.at[imp0, L.v_u].set(E0s * P_map[0])
            b0 = b0.at[imp0].set(
                inv_S_x[imp0] * (data.x_prop_plus[0, imp0] - c_x[imp0]) + E0s @ (p_map[0] - us[0])
            )
        B0 = B0.at[L.su, L.v_u].set(jnp.diag(P_map[0])).at[L.stvb, L.v_tvb].set(jnp.eye(C))
        b0 = b0.at[L.su].set(p_map[0])

        # PTR transitions (stage s = k+1, node k → k+1), all stages at once.
        A_blk, B_blk, C_blk, r = self._dynamics_blocks(data)
        P_n, U_n, p_n = P_map[1:], U_map[1:], p_map[1:]
        A = jnp.zeros((N - 1, n_xa, n_xa), dtype=f64)
        A = A.at[:, L.sx, L.sx].set(A_blk)
        A = A.at[:, L.sx, L.su].set(B_blk + C_blk * U_n[:, None, :])
        A = A.at[:, L.su, L.su].set(diag(U_n))
        B = jnp.zeros((N - 1, n_xa, n_v), dtype=f64)
        B = B.at[:, L.sx, L.v_u].set(C_blk * P_n[:, None, :])
        B = B.at[:, L.sx, L.v_nu].set(jnp.eye(n_x))
        B = B.at[:, L.su, L.v_u].set(diag(P_n))
        B = B.at[:, L.stvb, L.v_tvb].set(jnp.eye(C))
        bx = r - bmv(A_blk, xs[:-1]) - bmv(B_blk, us[:-1]) - bmv(C_blk, us[1:] - p_n)
        b = jnp.concatenate([bx, p_n, jnp.zeros((N - 1, C))], axis=1)

        A = jnp.concatenate([jnp.zeros((1, n_xa, n_xa)), A])
        B = jnp.concatenate([B0[None], B])
        b = jnp.concatenate([b0[None], b])

        # ================= Cost (diagonal H) =================
        lam_cost = jnp.broadcast_to(jnp.asarray(data.lam_cost, dtype=f64), (n_x,))
        w_x = 2.0 * data.lam_prox[:, :n_x] + _DIAG_EPS
        w_u = 2.0 * data.lam_prox[:, n_x:] + _DIAG_EPS
        Hd = jnp.asarray(st["H_base"]).at[1:, L.w_x].set(w_x).at[1:, L.w_u].set(w_u)
        c = jnp.zeros((N + 1, n_w), dtype=f64)
        c = c.at[1:, L.w_x].set(-w_x * xs).at[1:, L.w_u].set(-w_u * us)
        c = c.at[1, L.w_x].add(st["init_sign"] * lam_cost)
        c = c.at[N, L.w_x].add(st["final_sign"] * lam_cost)
        c = c.at[1:, L.w_tvb].set(data.lam_vb_nodal[:, :C])
        # Stage 0's epigraph prices |x_0 − x̄_0| on free initial states at
        # _DIAG_EPS (the same order as the slot regularisation it replaces).
        c = c.at[0, L.v_tvc].set(_DIAG_EPS).at[1:N, L.v_tvc].set(data.lam_vc)

        # ================= Cones =================
        G = jnp.asarray(st["G_static"])
        h = jnp.asarray(st["h_static"])

        # Linearized nodal rows: ∇g_x S_x (x − x̄) + ∇g_u S_u (u − ū) + g ≤ t_vb.
        if C:
            r0, act = st["nodal_row0"], st["nodal_active"]
            gx = data.nodal_grad_x[:, :C] * S_x
            gu = data.nodal_grad_u[:, :C] * S_u
            h_nodal = -data.nodal_g[:, :C] + bmv(gx, xs) + bmv(gu, us)
            G_nodal = jnp.concatenate(
                [jnp.zeros((N, C, L.w_x.start)), gx, gu, -jnp.broadcast_to(jnp.eye(C), (N, C, C))],
                axis=2,
            )
            G = G.at[1:, r0 : r0 + C].set(jnp.where(act[..., None], G_nodal, 0.0))
            h = h.at[1:, r0 : r0 + C].set(jnp.where(act, h_nodal, _PAD_H))

        extra_G: List[jnp.ndarray] = []
        extra_h: List[jnp.ndarray] = []

        # CTCS difference rows: S_x[i]·|x_{k+1}[i] − x_k[i]| ≤ x_max[i], with
        # x_{k+1}[i] substituted from stage s's dynamics row. Stage s holds
        # x_{s-1}, so rows live on stages 0..N-1; stage N is padded.
        idx = st["ctcs_idx"]
        if idx.size:
            n_g = idx.size
            row = jnp.concatenate([B[:, idx, :], A[:, idx, :]], axis=2)  # (N, n_g, n_w)
            row = row.at[:, np.arange(n_g), n_v + idx].add(-1.0)
            scale = S_x[idx]
            x_max = st["ctcs_x_max"]
            act = st["ctcs_active"]
            G_p = jnp.where(act[..., None], scale[:, None] * row, 0.0)
            off = scale * b[:, idx]
            h_p = jnp.where(act, x_max - off, _PAD_H)
            h_m = jnp.where(act, x_max + off, _PAD_H)
            extra_G.append(
                jnp.concatenate(
                    [jnp.concatenate([G_p, -G_p], axis=1), jnp.zeros((1, 2 * n_g, n_w))]
                )
            )
            extra_h.append(
                jnp.concatenate(
                    [jnp.concatenate([h_p, h_m], axis=1), jnp.full((1, 2 * n_g), _PAD_H)]
                )
            )

        # User cones, linearized at every active node in one vmap. Both kinds
        # use G = σJ, h = σ(J z̄ − f(z̄)): σ = +1 for f ≤ 0 (orthant), σ = −1
        # for the SOC slack s = f(z̄) + J(z − z̄).
        def user_rows(fn, nodes, m, sigma, h_pad):
            val, J_x, J_u = self._linearize_at_nodes(fn, nodes, data, m)
            k = np.asarray(nodes)
            h_val = sigma * (bmv(J_x, xs[k]) + bmv(J_u, us[k]) - val)
            G_rows = jnp.zeros((N, m, n_w)).at[k, :, L.w_x].set(sigma * J_x)
            G_rows = G_rows.at[k, :, L.w_u].set(sigma * J_u)
            h_rows = jnp.broadcast_to(h_pad, (N, m)).at[k].set(h_val)
            extra_G.append(jnp.concatenate([jnp.zeros((1, m, n_w)), G_rows]))
            extra_h.append(jnp.concatenate([h_pad[None], h_rows]))

        for cc in st["nonneg"]:
            m = int(cc.m)
            user_rows(cc.jax_fn, cc.nodes, m, 1.0, jnp.full((m,), _PAD_H))

        # Orthant rows (CTCS, nonneg) precede SOC blocks, which come last.
        for cc in st["socs"]:
            m = int(cc.m_arg) + 1

            def soc_fn(x, u, k, p, cc=cc, m=m):
                bound = jnp.reshape(cc.bound_fn(x, u, k, p), (1,))
                return jnp.concatenate([bound, jnp.reshape(cc.arg_fn(x, u, k, p), (m - 1,))])

            user_rows(soc_fn, cc.nodes, m, -1.0, jnp.zeros((m,)).at[0].set(_PAD_H))

        if extra_G:
            G = jnp.concatenate([G, *extra_G], axis=1)
            h = jnp.concatenate([h, *extra_h], axis=1)

        # ================= Boundary =================
        fixN = st["fixN"]
        f = inv_S_x[fixN] * (data.x_term[fixN] - c_x[fixN])

        return dict(
            H=diag(Hd),
            c=c,
            A=A,
            B=B,
            b=b,
            G=G,
            h=h,
            x0=jnp.zeros((n_xa,), dtype=f64),
            F=jnp.asarray(st["F"]),
            f=f,
        )

    # ------------------------------------------------------------------
    # JAX-pure unpack
    # ------------------------------------------------------------------

    def _build_solution_jax(self, result: dict, data: SubproblemData) -> SubproblemSolution:
        """Map ICORE's stage solution back to PTR's :class:`SubproblemSolution`."""
        L = self.layout
        st = self._static
        n_x, C = L.n_x, L.n_nodal
        f64 = jnp.float64
        S_x, S_u = self._S_x_diag_j, self._S_u_diag_j

        xa = result["x"][1:]  # (N, n_xa): stage i ↔ node i-1
        nu = result["u"][1:, L.v_nu]  # (N-1, n_x): transition stages
        x_sc, u_sc, t_vb = xa[:, L.sx], xa[:, L.su], xa[:, L.stvb]
        xs, us = self._scaled_reference(data)
        dx, du = x_sc - xs, u_sc - us

        # Virtual buffer = linearized constraint value at the solution.
        nu_vb = (
            data.nodal_g[:, :C]
            + jnp.einsum("kcj,kj->kc", data.nodal_grad_x[:, :C] * S_x, dx)
            + jnp.einsum("kcj,kj->kc", data.nodal_grad_u[:, :C] * S_u, du)
        )
        nu_vb = jnp.where(st["nodal_active"], nu_vb, 0.0)

        lam_cost = jnp.broadcast_to(jnp.asarray(data.lam_cost, dtype=f64), (n_x,))
        cost = (
            jnp.sum(data.lam_prox[:, :n_x] * dx**2)
            + jnp.sum(data.lam_prox[:, n_x:] * du**2)
            + jnp.dot(st["init_sign"] * lam_cost, x_sc[0])
            + jnp.dot(st["final_sign"] * lam_cost, x_sc[-1])
            + jnp.sum(data.lam_vc * jnp.abs(nu))
            + jnp.sum(data.lam_vb_nodal[:, :C] * t_vb)
        )

        return SubproblemSolution(
            x=x_sc * S_x + self._c_x_j,
            u=u_sc * S_u + self._c_u_j,
            nu=nu,
            nu_vb=nu_vb,
            nu_vb_cross=jnp.zeros((0,), dtype=f64),
            cost=cost.astype(f64),
            status_code=_icore_status_to_code(result["status"]),
        )

    # ------------------------------------------------------------------
    # Misc API
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return stage-OCP dimensions for the diagnostics summary box."""
        if self.layout is None:
            return {"n_variables": 0, "n_parameters": 0, "n_constraints": 0}
        L = self.layout
        n_vars = L.N * L.n_v + (L.N + 1) * L.n_xa
        if not self._static:
            return {"n_variables": n_vars, "n_parameters": 0, "n_constraints": -1}
        m = self._static["n_orth"] + sum(self._static["soc_sizes"])
        n_con = (L.N + 1) * m + L.N * L.n_xa + int(self._static["fixN"].size)
        return {"n_variables": n_vars, "n_parameters": 0, "n_constraints": n_con}

    def citation(self) -> List[str]:
        """Return BibTeX citation entries for ICORE."""
        return [
            r"""@software{icore,
  title={{ICORE}: A High-Performance Interior-Point Method for Conic {OCPs} via Riccati Equations},
  author={Uzun, Samet},
  url={https://github.com/sametuzun781/ICORE},
  year={2026}
}"""
        ]
