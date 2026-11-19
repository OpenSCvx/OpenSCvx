"""Native ICORE solves obey the same PTR data and result contracts as JAX.

These tests exercise real C and C++ libraries, including the CTCS structure
whose epigraph prefix is shorter than the state dimension. Numerical references
come from CVXPy on the same subproblem data. Reusing a callback with different
inputs checks the native workspace update path, not just the first solve.
"""

import dataclasses
import hashlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openscvx as ox
from openscvx.solvers import ICOREPTRSolver, resolve_solver_config
from openscvx.solvers.ptr_solver import StatusCode
from tests.solvers._iteration_callback_helpers import (
    build_brachistochrone,
    populate_numpy_stash,
    subproblem_data_from_numpy_stash,
)
from tests.solvers.test_icore_ptr_solver import _cvxpy_data, _make_double_integrator_problem

pytestmark = [pytest.mark.e2e, pytest.mark.icore]


def _assert_solution(actual, expected, atol=5e-5):
    assert int(actual.status_code) == int(StatusCode.OPTIMAL)
    assert int(expected.status_code) == int(StatusCode.OPTIMAL)
    for name in ("x", "u", "nu", "nu_vb", "nu_vb_cross"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)), atol=atol
        )
    np.testing.assert_allclose(float(actual.cost), float(expected.cost), atol=atol, rtol=1e-5)


def _initialize_pair(native, cvxpy):
    native.initialize()
    populate_numpy_stash(native)
    # Retain the default warm_start=False: each numerical reference must solve
    # the current matrices independently, rather than reuse a QOCO workspace.
    cvxpy.solver.solver_args.update({"abstol": 1e-9, "reltol": 1e-9})
    cvxpy.initialize()
    return SimpleNamespace(
        problem=native,
        reference=cvxpy,
        data=subproblem_data_from_numpy_stash(native.solver),
        callback=native.solver.iteration_callback(),
        reference_callback=cvxpy.solver.iteration_callback(),
    )


@pytest.fixture(scope="module")
def native_cases():
    """Reuse small initialized fixtures and compiled callbacks across tests."""
    cases = {}

    def get(backend, style="nodal"):
        key = backend, style
        if key not in cases:
            native = build_brachistochrone("icore", n=6, constraint_style=style)
            # Solver options are documented as mutable before initialize().
            native.solver.solver_args["backend"] = backend
            cvxpy = build_brachistochrone("cvxpy", n=6, constraint_style=style)
            cases[key] = _initialize_pair(native, cvxpy)
        return cases[key]

    yield get
    for case in cases.values():
        case.problem.solver.close()


def test_native_selection_capabilities_and_cache_identity():
    assert ICOREPTRSolver().solver_args["backend"] == "jax"
    digests = set()
    for backend in ("jax", "c", "cpp"):
        solver = resolve_solver_config(
            {"backend": "icore", "solver_args": {"backend": backend}}
        ).build()
        assert solver.solver_args["backend"] == backend
        assert solver.exportable is (backend == "jax")
        hasher = hashlib.sha256()
        solver._hash_into(hasher)
        digests.add(hasher.hexdigest())
    assert len(digests) == 3
    with pytest.raises(ValueError, match="backend"):
        ICOREPTRSolver(solver_args={"backend": "cuda"})


@pytest.mark.parametrize("backend", ["c", "cpp"])
@pytest.mark.parametrize("style", ["ctcs", "nodal"])
def test_native_subproblem_matches_cvxpy(native_cases, backend, style):
    case = native_cases(backend, style)
    actual = case.callback(None, case.data)
    expected = case.reference_callback(None, _cvxpy_data(case.data))
    _assert_solution(actual, expected)
    assert not case.problem.solver.exportable


def _changed_data(data):
    """Change matrices, offsets, costs, references and fixed boundary values."""
    nx = data.x_bar.shape[1]
    return dataclasses.replace(
        data,
        x_bar=data.x_bar + 0.12,
        u_bar=data.u_bar + 0.03,
        A_d=data.A_d + 0.004 * jnp.eye(nx)[None],
        B_d=data.B_d * 0.97,
        C_d=data.C_d * 1.02,
        x_prop=data.x_prop + 0.02,
        nodal_g=data.nodal_g + 0.03,
        nodal_grad_x=data.nodal_grad_x * 1.04,
        # Preserve the CVXPy model's declared sparsity. These constraints
        # depend only on state, so zero control derivatives must remain zero.
        nodal_grad_u=data.nodal_grad_u * 1.03,
        lam_prox=data.lam_prox * 1.3,
        lam_cost=data.lam_cost * 0.9,
        lam_vc=data.lam_vc * 1.1,
        lam_vb_nodal=data.lam_vb_nodal * 1.2,
        x_init=data.x_init.at[0].set(0.15).at[1].set(9.85),
        x_term=data.x_term.at[0].set(9.7).at[1].set(5.25),
    )


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_reused_callback_updates_all_coefficients_and_batches(native_cases, backend):
    case = native_cases(backend)
    first, changed = case.data, _changed_data(case.data)
    callback = jax.jit(case.callback)
    references = [case.reference_callback(None, _cvxpy_data(data)) for data in (first, changed)]
    assert np.max(np.abs(np.asarray(references[0].x - references[1].x))) > 0.1

    # A -> B -> A must update a reused workspace, including restoring old data.
    reference_order = (references[0], references[1], references[0])
    for data, expected in zip((first, changed, first), reference_order):
        _assert_solution(callback(None, data), expected)

    # Native vmap is a sequence of host solves; each item still has its own data.
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b, a]), first, changed)
    result = jax.jit(jax.vmap(callback, in_axes=(None, 0)))(None, batch)
    for index, expected in enumerate((references[0], references[1], references[0])):
        _assert_solution(jax.tree.map(lambda a: a[index], result), expected)

    # Concurrent invocations must not mix update/solve pairs in one workspace.
    inputs = (first, changed, changed, first)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(lambda data: jax.block_until_ready(callback(None, data)), inputs)
        )
    reference_order = (references[0], references[1], references[1], references[0])
    for actual, expected in zip(outputs, reference_order):
        _assert_solution(actual, expected)


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_user_cones_and_runtime_parameters_match_cvxpy(backend):
    radius = ox.Parameter("radius", shape=(1,), value=np.array([0.1]))
    nodes = [1, 2, 3]
    constraints = [
        lambda pos, vel, u: (vel[0] <= radius[0] / 2).convex().at(nodes),
        lambda pos, vel, u: (ox.linalg.Norm(vel) <= radius[0]).convex().at(nodes),
    ]
    native = _make_double_integrator_problem(
        k_max=1, constraints=constraints, solver_args={"backend": backend}
    )
    cvxpy = _make_double_integrator_problem(backend="cvxpy", k_max=1, constraints=constraints)
    case = _initialize_pair(native, cvxpy)
    base = dataclasses.replace(case.data, x_bar=case.data.x_bar + 0.7, u_bar=case.data.u_bar + 0.3)
    try:
        for value in (0.1, 0.2, 0.1):
            params = {**base.params, "radius": jnp.array([value])}
            data = dataclasses.replace(base, params=params)
            cvxpy.parameters["radius"] = np.array([value])
            actual = case.callback(None, data)
            expected = case.reference_callback(None, _cvxpy_data(data))
            _assert_solution(actual, expected, atol=2e-5)
            norms = np.linalg.norm(np.asarray(actual.x)[nodes, 2:4], axis=1)
            assert np.all(norms <= value + 1e-6)
            assert np.max(norms) > value - 1e-4
    finally:
        native.solver.close()


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_iteration_limit_is_unknown_in_both_interfaces(backend):
    problem = _make_double_integrator_problem(
        n=4, k_max=1, max_iter=0, solver_args={"backend": backend}
    )
    problem.initialize()
    populate_numpy_stash(problem)
    try:
        data = subproblem_data_from_numpy_stash(problem.solver)
        solution = problem.solver.iteration_callback()(None, data)
        assert int(solution.status_code) == int(StatusCode.UNKNOWN)
        assert problem.solver.solve().status == "unknown"
    finally:
        problem.solver.close()


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_invalid_update_does_not_poison_next_solve(native_cases, backend):
    case = native_cases(backend)
    expected = case.callback(None, case.data)
    invalid = dataclasses.replace(case.data, A_d=case.data.A_d.at[0, 0, 0].set(jnp.nan))
    failed = case.callback(None, invalid)
    assert int(failed.status_code) == int(StatusCode.UNKNOWN)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(failed))
    _assert_solution(case.callback(None, case.data), expected, atol=1e-9)


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_fixed_boundary_bounds_and_free_nan_sentinels(native_cases, backend):
    case = native_cases(backend, "ctcs")
    # Initial/final position components lie exactly on their declared bounds;
    # free final components carry NaN sentinels. Both must remain admissible.
    free = np.asarray(case.problem.settings.sim.x.final_type) != "Fix"
    assert np.any(free)
    data = dataclasses.replace(
        case.data, x_term=jnp.where(jnp.asarray(free), jnp.nan, case.data.x_term)
    )
    assert int(case.callback(None, data).status_code) == int(StatusCode.OPTIMAL)
    for invalid in (
        dataclasses.replace(case.data, x_init=case.data.x_init.at[1].set(11.0)),
        dataclasses.replace(case.data, x_term=case.data.x_term.at[0].set(11.0)),
    ):
        assert int(case.callback(None, invalid).status_code) == int(StatusCode.INFEASIBLE)
    nonfinite = dataclasses.replace(case.data, x_term=case.data.x_term.at[0].set(jnp.nan))
    assert int(case.callback(None, nonfinite).status_code) == int(StatusCode.UNKNOWN)
    assert int(case.callback(None, case.data).status_code) == int(StatusCode.OPTIMAL)


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_batched_export_reports_host_callback_limitation(native_cases, backend):
    problem = native_cases(backend, "ctcs").problem
    problem.settings.sim.save_compiled = True
    guesses = jnp.stack([problem.state.x, problem.state.x])
    try:
        with pytest.raises(ValueError, match="pure_callback"):
            problem.solve_batched(x_guess=guesses)
    finally:
        problem.settings.sim.save_compiled = False


def test_native_close_and_reinitialize_do_not_reuse_old_workspace():
    problem = _make_double_integrator_problem(n=4, k_max=1, solver_args={"backend": "c"})
    problem.initialize()
    populate_numpy_stash(problem)
    solver = problem.solver
    data = subproblem_data_from_numpy_stash(solver)
    old_callback = solver.iteration_callback()
    expected = jax.block_until_ready(old_callback(None, data))
    solver.solver_args["max_iterations"] = 0
    # The settings snapshot changes only on initialize(), as for JAX backends.
    _assert_solution(old_callback(None, data), expected, atol=1e-9)
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.iteration_callback()
    # JAX versions differ in the wrapper type for a host callback exception.
    with pytest.raises((RuntimeError, ValueError), match="closed"):
        jax.block_until_ready(old_callback(None, data))
    solver.initialize(problem.lowered, problem.settings)
    try:
        assert int(solver.iteration_callback()(None, data).status_code) == int(StatusCode.UNKNOWN)
        with pytest.raises((RuntimeError, ValueError), match="closed"):
            jax.block_until_ready(old_callback(None, data))
    finally:
        solver.close()


def test_native_impulsive_bounds_check_post_jump_states_and_zero_pins():
    """An initial impulse may repair a pre-jump state; a fixed zero must fit its box."""

    def make_problem(control_lower):
        x = ox.State("x", shape=(1,), min=np.array([-1.0]), max=np.array([1.0]))
        x.initial = np.array([0.0])
        x.final = [("free", 0.0)]
        impulse = ox.Control("impulse", shape=(1,), parameterization="impulsive", nodes=[0, 2])
        impulse.min = np.array([control_lower])
        impulse.max = np.array([3.0])
        impulse.guess = np.full((3, 1), max(0.0, control_lower))
        problem = ox.Problem(
            dynamics={"x": 0.0 * x},
            dynamics_discrete={"x": x + impulse},
            states=[x],
            controls=[impulse],
            constraints=[],
            time=ox.Time(initial=0.0, final=1.0, min=0.0, max=10.0),
            N=3,
            float_dtype="float64",
            solver={"backend": "icore", "solver_args": {"backend": "c"}},
        )
        problem.settings.dev.printing = False
        problem.initialize()
        populate_numpy_stash(problem)
        return problem, subproblem_data_from_numpy_stash(problem.solver)

    problem, data = make_problem(-3.0)
    try:
        # The input boundary is pre-jump. The post-jump equation is x[0] = 2 + u[0].
        data = dataclasses.replace(
            data,
            x_init=data.x_init.at[0].set(2.0),
            x_prop_plus=data.x_prop_plus.at[0, 0].set(2.0),
        )
        solution = problem.solver.iteration_callback()(None, data)
        assert int(solution.status_code) == int(StatusCode.OPTIMAL)
        assert abs(float(solution.x[0, 0])) <= 1.0 + 1e-7
        impulse_index = problem.settings.sim.u.slice_impulsive.start
        np.testing.assert_allclose(solution.x[0, 0], 2.0 + solution.u[0, impulse_index], atol=1e-7)
        np.testing.assert_allclose(solution.u[1, impulse_index], 0.0, atol=1e-10)
    finally:
        problem.solver.close()

    # The middle node is pinned to zero because it is not an impulse node.
    # Bounds requiring a strictly positive impulse make that model infeasible.
    problem, data = make_problem(0.2)
    try:
        solution = problem.solver.iteration_callback()(None, data)
        assert int(solution.status_code) == int(StatusCode.INFEASIBLE)
    finally:
        problem.solver.close()


@pytest.mark.parametrize("backend", ["c", "cpp"])
def test_native_scp_loop_tracks_cvxpy(backend):
    native = _make_double_integrator_problem(n=4, k_max=3, solver_args={"backend": backend})
    cvxpy = _make_double_integrator_problem(backend="cvxpy", n=4, k_max=3)
    case = _initialize_pair(native, cvxpy)
    try:
        actual = case.problem.solve()
        expected = case.reference.solve()
        np.testing.assert_allclose(actual["x"], expected["x"], atol=1e-3)
        np.testing.assert_allclose(actual["u"], expected["u"], atol=1e-3)
        np.testing.assert_allclose(actual["x"][-1, :2], [3.0, 3.0], atol=1e-6)
    finally:
        native.solver.close()
