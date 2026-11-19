# ICORE native callback measurements

The native adapter is a compatibility option, with a size-dependent performance
tradeoff. In this CPU run, C and C++ were slower than optimized JAX for the small
callback and the small complete SCP workload. They were faster for the larger
callback. These results include the real host callback boundary; they do not use
the earlier prototype timings that excluded that boundary and result unpacking.

The complete measurements, samples, numerical audits, dependency information, and
source hashes are in [icore-native-2026-11-19.json](icore-native-2026-11-19.json).

## Warm execution

All values below are wall-clock medians in milliseconds. Positive percentages
mean more time than JAX; negative percentages mean less time.

| Workload | Samples per backend | JAX | C | C++ | C time change | C++ time change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CTCS brachistochrone, 6 nodes, complete callback | 100 | 1.0562 | 1.2452 | 1.2351 | +17.9% | +16.9% |
| CTCS brachistochrone, 20 nodes, complete callback | 100 | 2.3333 | 1.7796 | 1.7370 | −23.7% | −25.6% |
| CTCS brachistochrone, 6 nodes, public `solve_jax`, 3 SCP steps | 10 | 11.7509 | 12.2407 | 12.0927 | +4.2% | +2.9% |

The callback includes JAX assembly, coefficient transfer to the native callback,
native validation/update/solve, transfer back, and JAX result unpacking. Every
output leaf is synchronized before stopping the timer. Native workspaces are
reused, while each numerical solve starts cold.

The two callback inputs alternate `A_d ± 1e-4 I` and trust-region weights scaled
by `0.99` or `1.01`. Thus native updates cannot rely on unchanged dynamics or
Hessians. All three backends receive the same current input. The order cycles
through all six backend permutations after five warmup calls per backend.

The full SCP measurement calls the public `Problem.solve_jax` method, including
state/result construction and nonlinear discretization. All backends execute
exactly three SCP steps from the same initial guess. Convergence thresholds are
set to zero to obtain a fixed workload; these runs do **not** establish a
time-to-convergence advantage. Three warmups precede each backend's ten samples.

## Compilation and initialization

Compilation and first execution are excluded from the warm table. Selected
one-time observations, in seconds:

| Operation | JAX | C | C++ |
| --- | ---: | ---: | ---: |
| Compile standalone 6-node callback after `initialize()` | 9.820 | 0.866 | 0.798 |
| Compile standalone 20-node callback after `initialize()` | 10.756 | 0.873 | 0.864 |
| First public 3-step `solve_jax` call after `initialize()` | 14.991 | 4.597 | 4.429 |

Both native shared libraries already existed before this run. Their filenames,
sizes, and modification times are recorded in the JSON. A machine without the
compiled native dependencies also pays the C/C++ and BLASFEO build cost.

`Problem.initialize()` itself executes a warmup fused SCP iteration. Consequently,
the JSON's callback `first_execution_seconds` means the first invocation of the
separately compiled callback **after initialization**, rather than the process's
first native solve. Initialization times are recorded separately. They include
other compilation and warmup work, with cache reuse affected by backend order.
The table is not a fresh-machine startup comparison and should not be added to
the warm timings as a universal startup cost.

## Numerical checks

All 600 timed callback calls returned `OPTIMAL`. Outside timing, the original
input and both changing variants were checked for each of the three backends at
both sizes: **18 independent original-model KKT audits**. All passed the same
`1e-8` tolerance; the largest recomputed primal residual, dual residual, or total
complementarity gap was **4.2031e-9**. The audit uses the original OCP arrays and
original input-stage algebra, independently of the solver's cached factors.

The raw primal/dual solve is also decoded and compared with the actual timed
callback. Native versus JAX physical state/control/virtual-control/cost
differences were at most **1.046e-9** across the six input variants. Raw epigraph
auxiliaries are not used as a proxy for physical trajectory agreement. IPM
iteration counts matched across all backends: `9/10/9` for the three 6-node inputs
and `11/11/11` for the 20-node inputs.

For the complete SCP case, all three independently replayed subproblem steps
returned `OPTIMAL` for every backend. The replay and public `solve_jax` final
trajectories agreed exactly within each backend. Across native and JAX, the
largest final state difference was **4.98e-12**, and the largest difference in
the three subproblem objectives was **1.86e-12**. All 30 timed public calls
returned finite trajectories.

JAX eliminates five virtual-control epigraphs in these fixtures. Native ICORE's
stricter pair contract permits four because the remaining virtual control also
appears in CTCS rows. Both solve the same original optimization arrays; the
native per-update validator remains active. Native structural presolve is
disabled by this adapter.

## Environment, provenance, and scope

This run used Python 3.12.14, JAX/jaxlib 0.12.1, NumPy 2.5.5, Linux x86-64, and
one pinned CPU (logical CPU 8) on an AMD EPYC 9V74. BLAS thread counts were one,
and XLA CPU Eigen multithreading was disabled. Both native builds used Debian
GCC 14.2.0, BLASFEO's `X64_INTEL_HASWELL` target, and LTO.

The ICORE dependency was **`52764b7a7e0d13ba41d68adcb7da6a9365c75b43`**, the local
revision containing the JAX optimizations. This differs from the public upstream
revision pinned by this PR's CI,
`a63ebffb005725a02d2c3e4247fb73aec453e590`. The native implementation is unchanged
between those revisions; the JAX baseline is different. These timings therefore
must not be presented as measurements of the CI dependency's JAX implementation.

The OpenSCvx implementation was uncommitted during measurement. Its recorded Git
HEAD identifies the PR baseline; the JSON's SHA-256 fingerprints identify the
actual measured source, including the benchmark script and fixtures. All checked
source hashes were unchanged from the beginning to the end of the run.

This is a small deterministic CPU sample, not a general solver ranking. It does
not measure GPU transfers, batched throughput, long-horizon convergence, or an
FFI implementation. Raw samples and means are retained to expose variability;
the small C/C++ differences do not establish a general language advantage. The
broader correctness tests remain necessary to assess other problem mappings.

## Reproduce

Use this OpenSCvx checkout with its runtime dependencies and an ICORE checkout at
the revision above. The ICORE source checkout also provides `example_tools.audit`
for the independent numerical checks. Choose an available logical CPU for your
machine; the script sets single-thread environment variables before importing
JAX.

```bash
python scripts/benchmark_icore_backends.py \
  --icore-root /path/to/ICORE \
  --cpu 8 \
  --nodes 6 20 \
  --repetitions 100 \
  --warmup 5 \
  --tolerance 1e-8 \
  --scp-iterations 3 \
  --scp-repetitions 10 \
  --output docs/benchmarks/icore-native-local.json
```

The script exits unsuccessfully if its audits, status/parity checks, or source
stability check fail. Set `--scp-repetitions 0` to run only the complete callback
comparison. Results from different hardware or dependency revisions should be
saved as separate records.
