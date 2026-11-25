"""Measure the complete OpenSCvx ICORE callback, including native host crossings.

Run from this checkout with its runtime dependencies and a source checkout of
ICORE (the independent numerical audit lives in ICORE's example_tools)::

    python scripts/benchmark_icore_backends.py --icore-root ../ICORE --cpu 8 \
        --output docs/benchmarks/icore-native.json

The fixtures are the repository's deterministic CTCS brachistochrone cases.
Every timed callback assembles current coefficients, solves, and unpacks all
outputs. All backends receive identical changed inputs and alternate call order.
Compilation/first execution, numerical audits, and optional full SCP solves are
reported separately. No result from the earlier host-only prototype is used.
"""

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = ("jax", "c", "cpp")


def fingerprint(root, files):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def commit(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def stats(samples):
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def timed(function, argument):
    import jax

    start = time.perf_counter_ns()
    result = function(argument)
    jax.block_until_ready(result)
    return result, (time.perf_counter_ns() - start) / 1e6


def interleaved(functions, arguments, repetitions, warmup, check):
    """Cycle all six backend permutations and both changing input variants."""
    import itertools

    orders = tuple(itertools.permutations(BACKENDS))
    samples = {backend: [] for backend in BACKENDS}
    accepted = {backend: True for backend in BACKENDS}
    for i in range(warmup):
        for backend in BACKENDS:
            result, _ = timed(functions[backend], arguments[i % len(arguments)])
            accepted[backend] &= bool(check(result))
    for i in range(repetitions):
        argument = arguments[(i // len(orders)) % len(arguments)]
        for backend in orders[i % len(orders)]:
            result, elapsed = timed(functions[backend], argument)
            samples[backend].append(elapsed)
            accepted[backend] &= bool(check(result))
    return {
        backend: {**stats(samples[backend]), "all_checks_passed": accepted[backend]}
        for backend in BACKENDS
    }


def fixture(n, backend, tolerance, k_max=1):
    from tests.solvers._iteration_callback_helpers import build_brachistochrone

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        problem = build_brachistochrone("icore", n=n, k_max=k_max, constraint_style="ctcs")
        problem.solver.solver_args.update(backend=backend, tolerance=tolerance)
        problem.initialize()
    return problem, time.perf_counter() - start


def difference(a, b):
    import numpy as np

    return {
        key + "_max_abs": float(
            np.max(np.abs(np.asarray(getattr(a, key)) - getattr(b, key)), initial=0)
        )
        for key in ("x", "u", "nu", "cost")
    }


def ocp_blocks(p):
    nu, n = p["B"].shape[-1], p["A"].shape[0]
    return {
        **{key: p[key] for key in ("A", "B", "b", "G", "h", "x0", "F", "f")},
        "Q": p["H"][:, nu:, nu:],
        "R": p["H"][:n, :nu, :nu],
        "S": p["H"][:n, :nu, nu:],
        "q": p["c"][:, nu:],
        "r": p["c"][:n, :nu],
    }


def callback_case(n, args):
    import jax
    import numpy as np
    from example_tools.audit import audit_physical
    from icore import Solver

    from tests.solvers._iteration_callback_helpers import (
        populate_numpy_stash,
        subproblem_data_from_numpy_stash,
    )

    problems, calls, cold = {}, {}, {}
    original = None
    for backend in BACKENDS:
        problem, initialize_seconds = fixture(n, backend, args.tolerance)
        problems[backend] = problem
        if original is None:
            populate_numpy_stash(problem)
            original = subproblem_data_from_numpy_stash(problem.solver)
            jax.block_until_ready(original)
        started = time.perf_counter()
        executable = problem.solver.iteration_callback().lower(None, original).compile()
        compile_seconds = time.perf_counter() - started
        calls[backend] = lambda data, executable=executable: executable(None, data)
        result, first_ms = timed(calls[backend], original)
        cold[backend] = {
            "initialize_seconds": initialize_seconds,
            "callback_compile_seconds": compile_seconds,
            "first_execution_seconds": first_ms / 1000,
            "first_status": int(result.status_code),
        }
        print(f"  N={n} {backend} compiled; first status={int(result.status_code)}", flush=True)

    variants = [
        dataclasses.replace(
            original,
            A_d=original.A_d + delta * np.eye(original.A_d.shape[-1])[None, :, :],
            lam_prox=original.lam_prox * scale,
        )
        for delta, scale in ((-1e-4, 0.99), (1e-4, 1.01))
    ]
    jax.block_until_ready(variants)
    # Timings precede raw-solve audit compilation; every backend is warmed first.
    measurements = interleaved(
        calls, variants, args.repetitions, args.warmup, lambda r: int(r.status_code) == 0
    )
    reference_solver = problems["jax"].solver
    assemble = jax.jit(reference_solver._assemble_ocp_jax)
    arrays = [assemble(data) for data in [original, *variants]]
    jax.block_until_ready(arrays)
    host = [jax.tree.map(np.asarray, p) for p in arrays]
    raw_jax = reference_solver._solve_fn.lower(arrays[0]).compile()
    raw_native = {}
    validation = {backend: [] for backend in BACKENDS}
    parity = {backend: [] for backend in BACKENDS if backend != "jax"}
    try:
        for backend in ("c", "cpp"):
            solver = problems[backend].solver
            settings = {k: v for k, v in solver.solver_args.items() if k != "backend"}
            active = np.any(solver._static["ctcs_active"], axis=0)
            ne = int(np.min(solver._static["ctcs_idx"][active], initial=solver.layout.n_x))
            raw_native[backend] = Solver(
                host[0],
                backend=backend,
                nonnegative=solver._static["n_orth"],
                soc_sizes=solver._static["soc_sizes"],
                epigraph_controls=ne,
                structural_presolve=False,
                **settings,
            )
        for i, data in enumerate([original, *variants]):
            physical = {backend: calls[backend](data) for backend in BACKENDS}
            jax.block_until_ready(physical)
            for backend in BACKENDS:
                raw = raw_jax(arrays[i]) if backend == "jax" else raw_native[backend].solve(host[i])
                jax.block_until_ready(raw)
                audit = audit_physical(
                    ocp_blocks(host[i]),
                    reference_solver._static["n_orth"],
                    reference_solver._static["soc_sizes"],
                    raw,
                    tolerance=args.tolerance,
                    status_ok=int(raw["status"]) == 0,
                )
                decoded = reference_solver._build_solution_jax(raw, data)
                jax.block_until_ready(decoded)
                decoded_difference = difference(decoded, physical[backend])
                validation[backend].append(
                    {
                        "input_variant": i,
                        "status": int(physical[backend].status_code),
                        "raw_status": int(raw["status"]),
                        "ipm_iterations": int(raw["iterations"]),
                        "reported_residuals": np.asarray(raw["residuals"]).tolist(),
                        "independent_audit": audit,
                        "raw_callback_difference": decoded_difference,
                        "accepted": bool(
                            audit["accepted"]
                            and int(physical[backend].status_code) == 0
                            and max(decoded_difference.values()) <= 2e-7
                        ),
                    }
                )
                if backend != "jax":
                    delta = difference(physical[backend], physical["jax"])
                    parity[backend].append(
                        {
                            "input_variant": i,
                            **delta,
                            "accepted": bool(max(delta.values()) <= 1e-5),
                        }
                    )
    finally:
        for solver in raw_native.values():
            solver.close()
        for problem in problems.values():
            problem.solver.close()
    accepted = all(row["all_checks_passed"] for row in measurements.values())
    accepted &= all(row["accepted"] for rows in validation.values() for row in rows)
    accepted &= all(row["accepted"] for rows in parity.values() for row in rows)
    return {
        "name": f"brachistochrone_ctcs_n{n}",
        "kind": "iteration_callback",
        "scope": "assembly + solve (including native pure_callback) + unpack; all leaves blocked",
        "variants": "A_d +/- 1e-4 I and lam_prox times 0.99/1.01; same inputs for every backend",
        "shapes": {key: list(value.shape) for key, value in host[0].items()},
        "settings": {k: v for k, v in reference_solver.solver_args.items() if k != "backend"},
        "epigraph_controls": {"jax": reference_solver.layout.n_x, "c": ne, "cpp": ne},
        "cold": cold,
        "changing_inputs": measurements,
        "validation": validation,
        "native_vs_jax": parity,
        "accepted": bool(accepted),
    }


def scp_case(args):
    """Time public solve_jax, auditing its iterations outside the timed window."""
    import jax
    import numpy as np

    problems, functions, cold, validation = {}, {}, {}, {}
    thresholds = {"ep_tr": 0.0, "ep_vb": 0.0, "ep_vc": 0.0}
    for backend in BACKENDS:
        problem, setup = fixture(6, backend, args.tolerance, k_max=args.scp_iterations)
        problems[backend] = problem
        functions[backend] = lambda _, p=problem: p.solve_jax(
            max_iters=args.scp_iterations, algorithm=thresholds
        )
        result, elapsed = timed(functions[backend], None)
        cold[backend] = {"initialize_seconds": setup, "first_solve_jax_seconds": elapsed / 1000}
        state = problem._resolve_initial_state(
            None, None, max_iters=args.scp_iterations, algorithm=thresholds
        )
        statuses, objectives = [], []
        while int(state.k) <= args.scp_iterations and not bool(problem.algorithm.converged(state)):
            state, diagnostics = problem._iteration_fn(state, problem._parameters)
            jax.block_until_ready((state, diagnostics))
            statuses.append(int(diagnostics.status))
            objectives.append(float(diagnostics.J_lin))
        delta = max(
            float(np.max(np.abs(np.asarray(state.x) - np.asarray(result.x)))),
            float(np.max(np.abs(np.asarray(state.u) - np.asarray(result.u)))),
        )
        validation[backend] = {
            "scp_iterations": len(statuses),
            "subproblem_statuses": statuses,
            "subproblem_objectives": objectives,
            "nonlinear_objective": float(state.J_nonlin),
            "t_final": np.asarray(result.t_final).tolist(),
            "converged": bool(result.converged),
            "public_vs_audited_final_max_abs": delta,
            "accepted": bool(
                len(statuses) == args.scp_iterations
                and all(s == 0 for s in statuses)
                and delta <= 1e-6
            ),
        }
        print(f"  solve_jax {backend}: {len(statuses)} iterations, statuses={statuses}", flush=True)
    measurements = interleaved(
        functions,
        [None],
        args.scp_repetitions,
        min(args.warmup, 3),
        lambda r: all(np.all(np.isfinite(a)) for a in (r.x, r.u, r.t_final)),
    )
    final = {backend: functions[backend](None) for backend in BACKENDS}
    jax.block_until_ready(final)
    parity = {}
    for backend in ("c", "cpp"):
        parity[backend] = {
            "x_max_abs": float(np.max(np.abs(np.asarray(final[backend].x) - final["jax"].x))),
            "u_max_abs": float(np.max(np.abs(np.asarray(final[backend].u) - final["jax"].u))),
            "subproblem_objective_max_abs": float(
                np.max(
                    np.abs(
                        np.asarray(validation[backend]["subproblem_objectives"])
                        - validation["jax"]["subproblem_objectives"]
                    )
                )
            ),
        }
        parity[backend]["accepted"] = max(parity[backend].values()) <= 1e-4
    for problem in problems.values():
        problem.solver.close()
    return {
        "name": "brachistochrone_ctcs_n6_solve_jax",
        "kind": "solve_jax",
        "scope": "public solve_jax including state/result construction and all SCP iterations",
        "thresholds": thresholds,
        "note": "Fixed iteration workload; not a time-to-convergence claim",
        "cold": cold,
        "warm": measurements,
        "validation": validation,
        "native_vs_jax": parity,
        "accepted": bool(
            all(v["accepted"] for v in validation.values())
            and all(v["accepted"] for v in parity.values())
            and all(v["all_checks_passed"] for v in measurements.values())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--nodes", type=int, nargs="+", default=[6, 20])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--scp-repetitions", type=int, default=10, help="0 skips full SCP timing")
    parser.add_argument("--scp-iterations", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    args.icore_root = args.icore_root.resolve()
    sys.path[:0] = [str(root), str(args.icore_root)]
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import icore
    import jax
    import jaxlib
    import numpy as np
    from icore._build import build_info

    if Path(icore.__file__).resolve().parent.parent != args.icore_root:
        raise RuntimeError(f"Unexpected ICORE import: {icore.__file__}")
    files = [
        root / name
        for name in (
            "scripts/benchmark_icore_backends.py",
            "openscvx/solvers/icore_ptr_solver.py",
            "openscvx/solvers/ptr_solver.py",
            "openscvx/solvers/base.py",
            "openscvx/problem.py",
            "openscvx/algorithms/scvx/iteration.py",
            "openscvx/algorithms/loop.py",
            "openscvx/algorithms/state.py",
            "openscvx/algorithms/optimization_results.py",
            "tests/solvers/_iteration_callback_helpers.py",
        )
    ]
    icore_files = sorted(
        p
        for p in (args.icore_root / "icore").rglob("*")
        if p.suffix in (".py", ".c", ".h") and "vendor" not in p.parts
    ) + [args.icore_root / "example_tools/audit.py"]
    native_libraries = {
        backend: [
            {"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted((args.icore_root / ".build" / ("icore_" + backend)).glob("libicore*"))
            if p.suffix in (".so", ".dylib")
        ]
        for backend in ("c", "cpp")
    }
    report = {
        "schema": "openscvx-icore-backends-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "openscvx_commit": commit(root),
        "icore_commit": commit(args.icore_root),
        "openscvx_sha256": fingerprint(root, files),
        "icore_sha256": fingerprint(args.icore_root, icore_files),
        "source_note": "OpenSCvx changes were uncommitted; the file hashes identify tested source.",
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            )
            if Path("/proc/cpuinfo").exists()
            else platform.processor(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "devices": [str(d) for d in jax.devices()],
            "thread_environment": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")
            },
        },
        "protocol": (
            "Same changed input, all six backend call permutations; synchronize every result"
        ),
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "tolerance": args.tolerance,
        "native_library_present_before_run": {
            backend: bool(inventory) for backend, inventory in native_libraries.items()
        },
        "native_library_files_before_run": native_libraries,
        "cold_note": (
            "XLA compilation excludes execution. First native execution may build/load native "
            "code. Initialization/cache reuse differs by call order; "
            "not a fresh-machine startup test."
        ),
        "cases": [],
        "complete": False,
    }
    write_report(args.output, report)
    for n in args.nodes:
        print(f"Measuring complete callbacks N={n}", flush=True)
        row = callback_case(n, args)
        report["cases"].append(row)
        print({k: v["median_ms"] for k, v in row["changing_inputs"].items()}, flush=True)
        write_report(args.output, report)
    if args.scp_repetitions:
        print("Measuring complete solve_jax", flush=True)
        report["cases"].append(scp_case(args))
    report["native_builds"] = {backend: build_info(backend) for backend in ("c", "cpp")}
    report["source_unchanged"] = report["openscvx_sha256"] == fingerprint(root, files) and report[
        "icore_sha256"
    ] == fingerprint(args.icore_root, icore_files)
    report["complete"] = True
    report["all_accepted"] = report["source_unchanged"] and all(
        r["accepted"] for r in report["cases"]
    )
    write_report(args.output, report)
    print(f"Saved {args.output}; all_accepted={report['all_accepted']}", flush=True)
    return 0 if report["all_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
