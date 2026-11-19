"""Tests for the ICORE-backed PTR convex subproblem solver.

Covers:
  * Instantiation guard when ``icore`` isn't importable.
  * ``PTRSolverSpec`` / constructor option handling for ``backend='icore'``.
  * Rejection of unsupported features (float32, zero-cone ``.convex()``).
  * Subproblem parity against ``CVXPyPTRSolver`` on an identical
    :class:`SubproblemData` (CTCS and nodal formulations) — this checks the
    whole augmented-state / pre-stage reformulation, not just convergence.
  * NumPy ``solve()`` vs ``iteration_callback`` agreement, ``jit`` / ``vmap``
    composition.
  * End-to-end SCP solves, including user affine-inequality and SOC
    ``.convex()`` constraints restricted to a node subset (exercises the
    uniform-cone-layout padding).

The brachistochrone and Hohmann parametrized-backend tests in ``tests/e2e/``
exercise ICORE on richer nonlinear CTCS / nodal / impulsive problems.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openscvx as ox
from openscvx import Problem
from openscvx.solvers import ICOREPTRSolver, PTRSolver, PTRSolveResult
from openscvx.solvers.ptr_solver import StatusCode, SubproblemSolution
from tests.solvers._iteration_callback_helpers import (
    build_brachistochrone,
    populate_numpy_stash,
    subproblem_data_from_numpy_stash,
)

pytestmark = [pytest.mark.e2e, pytest.mark.icore]

# ============================================================================
# Helpers
# ============================================================================


def _make_double_integrator_problem(n=6, backend="icore", k_max=20, constraints=(), **kw):
    """2-D double integrator with box bounds. Auto-CTCS for time bounds still
    applies, exercising the CTCS row assembly; the uniform time grid exercises
    the control-copy elimination."""
    pos = ox.State("pos", shape=(2,))
    pos.min = np.array([-10.0, -10.0])
    pos.max = np.array([10.0, 10.0])
    pos.initial = np.array([0.0, 0.0])
    pos.final = np.array([3.0, 3.0])

    vel = ox.State("vel", shape=(2,))
    vel.min = np.array([-5.0, -5.0])
    vel.max = np.array([5.0, 5.0])
    vel.initial = np.array([0.0, 0.0])
    vel.final = [("free", 0.0), ("free", 0.0)]

    u = ox.Control("u", shape=(2,))
    u.min = np.array([-3.0, -3.0])
    u.max = np.array([3.0, 3.0])
    u.guess = np.zeros((n, 2))

    time = ox.Time(initial=0.0, final=("minimize", 2.0), min=0.0, max=10.0, uniform_time_grid=True)
    cons = [c(pos, vel, u) for c in constraints]

    prob = Problem(
        dynamics={"pos": vel, "vel": u},
        states=[pos, vel],
        controls=[u],
        time=time,
        constraints=cons,
        N=n,
        float_dtype=kw.pop("float_dtype", "float64"),
        algorithm={
            "lam_prox": 1.0,
            "lam_cost": 0.5,
            "k_max": k_max,
            "ep_tr": 1e-5,
            "ep_vb": 1e-5,
            "ep_vc": 1e-8,
        },
        solver={"backend": backend, **kw},
    )
    prob.settings.dev.printing = False
    return prob


def _cvxpy_data(data):
    # CVXPy's callback declares lam_vb_cross with shape (1,) when there are no
    # cross-node constraints; the JAX backends use (0,).
    return dataclasses.replace(data, lam_vb_cross=jnp.zeros((1,)))


# ============================================================================
# Construction / dependency-guard / config tests
# ============================================================================


def test_icore_solver_is_a_PTRSolver():
    assert isinstance(ICOREPTRSolver(), PTRSolver)


def test_icore_missing_icore_raises_clear_error(monkeypatch):
    """Without icore importable, instantiation should explain where to get it."""
    import openscvx.solvers.icore_ptr_solver as mod

    monkeypatch.setattr(mod, "_ICORE_AVAILABLE", False)
    with pytest.raises(ImportError, match="github.com/sametuzun781/ICORE"):
        ICOREPTRSolver()


def test_icore_import_does_not_change_global_x64():
    """icore.jax.core enables jax_enable_x64 at import; our module must undo
    that so Problem(float_dtype=...) stays in charge of the global dtype.

    Runs in a fresh interpreter: the import side effect only fires once."""
    import subprocess
    import sys

    code = (
        "import jax; jax.config.update('jax_enable_x64', False); "
        "import openscvx.solvers.icore_ptr_solver as m; assert m._ICORE_AVAILABLE; "
        "print(jax.config.read('jax_enable_x64'))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_icore_named_params_populate_solver_args():
    solver = ICOREPTRSolver(solver_tol=1e-9, max_iter=50, solver_args={"riccati_mode": 2})
    assert solver.solver_args["tolerance"] == 1e-9
    assert solver.solver_args["max_iterations"] == 50
    assert solver.solver_args["riccati_mode"] == 2
    # Untouched settings keep ICORE's defaults.
    assert solver.solver_args["step_policy"] == "auto"


def test_icore_named_param_overlap_with_solver_args_raises():
    with pytest.raises(ValueError, match="tolerance"):
        ICOREPTRSolver(solver_tol=1e-8, solver_args={"tolerance": 1e-6})
    with pytest.raises(ValueError, match="max_iterations"):
        ICOREPTRSolver(max_iter=10, solver_args={"max_iterations": 20})


def test_icore_rejects_unknown_solver_args():
    with pytest.raises(ValueError, match="Unknown ICORE settings"):
        ICOREPTRSolver(solver_args={"abstol": 1e-6})
    with pytest.raises(ValueError, match="step_policy"):
        ICOREPTRSolver(solver_args={"step_policy": "greedy"})


def test_icore_spec_build_and_field_validation():
    from openscvx.solvers import resolve_solver_config

    solver = resolve_solver_config({"backend": "icore", "solver_tol": 1e-9, "max_iter": 80}).build()
    assert isinstance(solver, ICOREPTRSolver)
    assert solver.solver_args["tolerance"] == 1e-9
    assert solver.solver_args["max_iterations"] == 80

    with pytest.raises(ValueError, match="only valid for backend='cvxpy'"):
        resolve_solver_config({"backend": "icore", "cvx_solver": "CLARABEL"})
    with pytest.raises(ValueError, match="only valid for backend='moreau'"):
        resolve_solver_config({"backend": "icore", "device": "cpu"})


def test_icore_requires_float64():
    with pytest.raises(ValueError, match="float64"):
        _make_double_integrator_problem(k_max=1, float_dtype="float32").initialize()


def test_icore_rejects_zero_cone_convex_constraints():
    """ICORE has no stage-wise equality cone, so a user .convex() equality
    must be rejected at lowering with a pointer to backends that support it."""
    with pytest.raises(NotImplementedError, match="ZeroConeConstraint"):
        _make_double_integrator_problem(
            k_max=1, constraints=[lambda pos, vel, u: (vel[0] == 0.5).convex().at([2])]
        )


# ============================================================================
# Subproblem parity with CVXPy (identical SubproblemData)
# ============================================================================


@pytest.mark.parametrize("constraint_style", ["ctcs", "nodal"])
def test_icore_subproblem_matches_cvxpy(constraint_style):
    """Feed one identical SubproblemData to both backends; the convex
    subproblem is the same, so the minimizers must agree to IPM tolerance.

    The nodal variant has non-identity state scaling on bounded positions,
    which pins down the scaling of linearized nodal gradients."""
    n = 6
    prob_i = build_brachistochrone("icore", n=n, constraint_style=constraint_style)
    prob_i.initialize()
    populate_numpy_stash(prob_i)
    data = subproblem_data_from_numpy_stash(prob_i.solver)

    prob_c = build_brachistochrone("cvxpy", n=n, constraint_style=constraint_style)
    prob_c.initialize()
    prob_c.solver.solver_args = {"abstol": 1e-9, "reltol": 1e-9}

    si = prob_i.solver.iteration_callback()(None, data)
    sc = prob_c.solver.iteration_callback()(None, _cvxpy_data(data))

    assert int(si.status_code) == int(StatusCode.OPTIMAL)
    np.testing.assert_allclose(np.asarray(si.x), np.asarray(sc.x), atol=5e-5)
    np.testing.assert_allclose(np.asarray(si.u), np.asarray(sc.u), atol=5e-5)
    np.testing.assert_allclose(np.asarray(si.nu), np.asarray(sc.nu), atol=5e-5)
    np.testing.assert_allclose(np.asarray(si.nu_vb), np.asarray(sc.nu_vb), atol=5e-5)
    np.testing.assert_allclose(float(si.cost), float(sc.cost), rtol=1e-5)


def test_icore_user_cone_subproblem_matches_cvxpy_off_origin():
    """Binding affine and SOC .convex() rows, linearized about a reference far
    from the origin, so ``val − J·z̄`` offsets in the cone rows are exercised
    (at a near-zero reference a sign error there is invisible)."""
    nodes = [1, 2, 3]
    cons = [
        lambda pos, vel, u: (vel[0] <= 0.05).convex().at(nodes),
        lambda pos, vel, u: (ox.linalg.Norm(vel) <= 0.1).convex().at(nodes),
    ]
    prob_i = _make_double_integrator_problem(k_max=1, constraints=cons)
    prob_i.initialize()
    populate_numpy_stash(prob_i)
    data = subproblem_data_from_numpy_stash(prob_i.solver)
    data = dataclasses.replace(data, x_bar=data.x_bar + 0.7, u_bar=data.u_bar + 0.3)

    prob_c = _make_double_integrator_problem(backend="cvxpy", k_max=1, constraints=cons)
    prob_c.initialize()
    prob_c.solver.solver_args = {"abstol": 1e-9, "reltol": 1e-9}

    si = prob_i.solver.iteration_callback()(None, data)
    sc = prob_c.solver.iteration_callback()(None, _cvxpy_data(data))

    assert int(si.status_code) == int(StatusCode.OPTIMAL)
    np.testing.assert_allclose(np.asarray(si.x), np.asarray(sc.x), atol=1e-5)
    np.testing.assert_allclose(np.asarray(si.u), np.asarray(sc.u), atol=1e-5)
    np.testing.assert_allclose(float(si.cost), float(sc.cost), rtol=1e-5)
    vel = np.asarray(si.x)[nodes, 2:4]
    assert np.max(np.linalg.norm(vel, axis=1)) > 0.1 - 1e-4  # SOC binds


def test_icore_numpy_solve_matches_iteration_callback():
    prob = build_brachistochrone("icore", n=5, constraint_style="nodal")
    prob.initialize()
    populate_numpy_stash(prob)
    solver = prob.solver

    res = solver.solve()
    assert isinstance(res, PTRSolveResult)
    assert res.status == "optimal"
    sol = solver.iteration_callback()(None, subproblem_data_from_numpy_stash(solver))
    np.testing.assert_allclose(res.x, np.asarray(sol.x), atol=1e-10)
    np.testing.assert_allclose(res.u, np.asarray(sol.u), atol=1e-10)
    np.testing.assert_allclose(res.nu, np.asarray(sol.nu), atol=1e-10)
    assert len(res.nu_vb) == solver.layout.n_nodal


def test_icore_iteration_callback_jit_and_vmap():
    prob = build_brachistochrone("icore", n=4)
    prob.initialize()
    populate_numpy_stash(prob)
    solver = prob.solver
    base = subproblem_data_from_numpy_stash(solver)
    callback = solver.iteration_callback()

    # jit(callback) traces cleanly.
    out = jax.jit(callback)(None, base)
    assert isinstance(out, SubproblemSolution)
    assert out.x.shape == base.x_bar.shape

    # vmap over per-element trust-region weights matches unbatched calls.
    scales = jnp.array([0.5, 1.0, 2.0])
    batch = jax.tree.map(lambda a: jnp.stack([a] * len(scales)), base)
    batch = dataclasses.replace(batch, lam_prox=batch.lam_prox * scales[:, None, None])
    batched = jax.vmap(callback, in_axes=(None, 0))(None, batch)
    for i, s in enumerate(scales):
        single = callback(None, dataclasses.replace(base, lam_prox=base.lam_prox * s))
        np.testing.assert_allclose(np.asarray(batched.x[i]), np.asarray(single.x), atol=1e-8)
        np.testing.assert_allclose(float(batched.cost[i]), float(single.cost), atol=1e-8)


# ============================================================================
# End-to-end SCP
# ============================================================================


def _solve_both(**kw):
    """Run the same SCP problem with CVXPy and ICORE; return both results.

    Each SCP iterate depends only on the previous one and the convex
    subproblem, so matching subproblem solutions mean the two trajectories
    track each other iteration-for-iteration — whether or not the SCP loop
    has converged by ``k_max``."""
    prob_c = _make_double_integrator_problem(backend="cvxpy", **kw)
    prob_c.initialize()
    prob_c.solver.solver_args = {"abstol": 1e-9, "reltol": 1e-9}
    res_c = prob_c.solve()

    prob_i = _make_double_integrator_problem(**kw)
    prob_i.initialize()
    res_i = prob_i.solve()
    return prob_i, res_i, res_c


def test_icore_scp_tracks_cvxpy_on_double_integrator():
    prob, res, res_cvx = _solve_both(k_max=10)
    x, x_cvx = res.get("x"), res_cvx.get("x")
    np.testing.assert_allclose(x, x_cvx, atol=1e-3)
    np.testing.assert_allclose(res.get("u"), res_cvx.get("u"), atol=1e-3)
    np.testing.assert_allclose(x[-1, :2], [3.0, 3.0], atol=1e-6)
    # Uniform time grid: time dilation identical at every node.
    u = res.get("u")
    td = prob.settings.sim.time_dilation_slice
    np.testing.assert_allclose(u[:, td], np.broadcast_to(u[:1, td], u[:, td].shape), atol=1e-9)


def test_icore_user_nonneg_and_soc_constraints_track_cvxpy():
    """Affine-inequality and SOC .convex() constraints on node subsets — the
    inactive stages get padded rows, so this checks the uniform cone layout."""
    nodes = [1, 2, 3]
    _, res, res_cvx = _solve_both(
        k_max=10,
        constraints=[
            lambda pos, vel, u: (vel[0] <= 1.8).convex().at(nodes),
            lambda pos, vel, u: (ox.linalg.Norm(vel) <= 1.6).convex().at(nodes),
        ],
    )
    x, x_cvx = res.get("x"), res_cvx.get("x")
    np.testing.assert_allclose(x, x_cvx, atol=1e-3)
    vel = x[:, 2:4]
    assert np.all(vel[nodes, 0] <= 1.8 + 1e-6)
    assert np.all(np.linalg.norm(vel[nodes], axis=1) <= 1.6 + 1e-6)
    # The SOC binds somewhere, so the cone rows are really in play.
    assert np.max(np.linalg.norm(vel[nodes], axis=1)) > 1.6 - 1e-3
