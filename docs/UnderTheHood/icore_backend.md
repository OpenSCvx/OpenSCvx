---
title: ICORE solver backends
description: >-
  Use ICORE's JAX, C, or C++ implementation through one stage model adapter,
  with explicit batching, export, and native workspace contracts.
---

# ICORE solver backends

`ICOREPTRSolver` converts each PTR convex subproblem into ICORE's stage-wise
conic optimal control form. One assembly serves the JAX, C, and C++
implementations. Changing the implementation does not require a new model
definition or a converter for each example.

## Select an implementation

The outer `backend` selects the OpenSCvx adapter. The nested `backend`
selects the ICORE implementation:

```python
import openscvx as ox

solver = {
    "backend": "icore",
    "solver_args": {"backend": "cpp", "tolerance": 1e-8},
}
problem = ox.Problem(..., solver=solver, float_dtype="float64")
```

Use `"c"` for native C or `"jax"` for JAX. Omitting the nested selector keeps
the JAX default. Direct construction uses the same settings:

```python
solver = ox.ICOREPTRSolver(solver_args={"backend": "c"})
```

All three implementations require `float_dtype="float64"`. Native C and C++
compile the same ICORE numerical sources with different compiler modes;
neither name implies a different optimization model or a guaranteed speed
advantage. See [installation](../installation.md#icore-solver-backend) for
the tested ICORE revision and native build requirements.

## Supported model class

The adapter supports:

- Continuous and impulsive dynamics, including first-order-hold control coupling.
- State and control bounds, boundary `Fix` / `Minimize` / `Maximize`, and a
  uniform time grid.
- Linearized nodal nonconvex inequalities and CTCS constraints.
- User `.convex()` affine inequalities and second-order-cone constraints,
  including constraints restricted to selected nodes.

Cross-node constraints, L1-penalized nodal equalities, and user `.convex()`
equalities are outside this adapter's supported class and raise
`NotImplementedError`. Use `CVXPyPTRSolver` for those models. The native
selector preserves these checks; choosing C or C++ does not add a stage
equality cone to ICORE.

State and control counts, horizon length, and node subsets come from the
model. They are not tied to a particular example. An initialized adapter
has a fixed layout, including cone structure and CTCS activity. Changing
that structure requires initializing the corresponding new model. Numerical
coefficients, PTR penalty weights, and supported boundary values can change
between solves without changing the layout.

## One stage assembly

OpenSCvx already lowers symbolic dynamics and constraints and computes
their per-iteration linearizations. The ICORE adapter consumes that
`SubproblemData` and assembles numerical arrays:

- `H, c` describe the quadratic and linear objective in `[control, state]` order.
- `A, B, b, x0` describe stage dynamics and the fixed initial state.
- `G, h` describe nonnegative and second-order cones with a uniform stage layout.
- `F, f` describe optional terminal equalities.

The augmented state carries the original state, control, and virtual-buffer
epigraphs. A pre-stage handles partially free initial conditions; dynamics
carry control copies and eliminate supported equalities. Inactive constraint
rows receive feasible padding. The result unpacker maps ICORE's original
stage coordinates back to OpenSCvx's state, control, and virtual controls.
Both assembly and unpacking are shared by all three implementations.

Native ICORE requires both an epigraph and its paired virtual control to be
absent from other cone rows. In this assembly, only active CTCS difference
rows introduce additional occurrences of a virtual control. At initialization,
the adapter therefore chooses a safe leading epigraph prefix: it stops before
the first state coordinate with an active CTCS row, or includes all `n_x`
epigraphs when there are no active CTCS rows. It passes that explicit count
as `epigraph_controls` to native ICORE.

This certificate follows from the adapter's fixed layout and identity virtual
control block. It remains valid as the numerical coefficients change; the
remaining epigraphs stay in the general formulation. Native ICORE still
validates the coefficients on every create or update. The adapter disables
the separate generic OCP-to-chain structural presolve
(`structural_presolve=False`), avoiding repeated structure searches for this
known stage assembly.

## Native execution and workspace reuse

The native path calls ICORE's public `Solver` API. The first concrete
subproblem creates a workspace; later calls update its coefficients and solve
again in the same initialized layout. ICORE returns solutions in the original
stage coordinates, including the epigraphs eliminated internally. Workspace
reuse is distinct from warm starting: every ICORE solve follows its
cold-start policy.
The [ICORE API guide](https://github.com/sametuzun781/ICORE/blob/a63ebffb005725a02d2c3e4247fb73aec453e590/docs/API.md)
documents the coefficient convention and the underlying `update` / `solve`
lifecycle.

An adapter owns its native workspace. Updates and solves are serialized
because the workspace and its output buffers are mutable. Every callback
uses its current array arguments; a previous trajectory is not an implicit
input to the next solve.

Layout and ICORE settings, including implementation, tolerance, and iteration
limit, are captured at initialization. Editing `solver_args` does not change
an existing callback; reinitialize to apply new settings. Reinitialization
closes the previous native workspace and creates a new callback. Explicit
`ICOREPTRSolver.close()` also releases native storage after any active solve
finishes. An old native callback remains attached to its closed workspace
and cannot be reused after closing or reinitializing the solver.

Only ICORE status zero maps to `OPTIMAL`. Other statuses indicate an iteration
limit, numerical failure, invalid input, or allocation failure; they do not establish
infeasibility. The OpenSCvx SCP loop receives a non-optimal status and applies
its normal failure handling.

### Guard for omitted fixed-variable bounds

The stage assembly omits box rows on structurally fixed values so that a
fixed value exactly at its bound does not create a zero-slack cone row.
The callback checks these values against their original physical bounds:
non-impulsive fixed initial states, fixed terminal states, zero initial
CTCS states, and pinned zero controls. It includes the runtime boundary
values in this check; unused free-boundary placeholders are ignored.

A finite fixed value outside its bound produces `INFEASIBLE`; a nonfinite
fixed value or a NaN bound produces `UNKNOWN`. This guard checks the omitted
box rows only. It does not turn a general ICORE numerical failure into an
infeasibility certificate.

## JIT, batching, and export

| Property | JAX implementation | Native C / C++ implementation |
|---|---|---|
| Convex solve | JAX/XLA kernel | CPU library through `jax.pure_callback` |
| `jax.jit` composition | Supported | Supported, with a host call |
| `jax.vmap` composition | JAX batching | Sequential native calls per batch element |
| Whole-solve `jax.export` | Exportable | `exportable=False` |
| Precision | Float64 | Float64 |

For native in-process batches, set `problem.settings.sim.save_compiled = False`
before initialization. Exporting the complete batched solve with
`save_compiled=True` is rejected because it would contain a Python host
callback. This does not prevent JAX from compiling discretization, assembly,
and the outer SCP loop.

Native ICORE executes on the CPU. Running the surrounding JAX computation
on an accelerator therefore introduces host transfers. Compare complete
subproblem or SCP latency, including assembly, transfers, and coefficient
updates, when choosing an implementation; native solver time alone omits
that work.

The native callback provides no custom JVP/VJP, so differentiation through
the native solve is unsupported. JAX remains responsible for the dynamics
and constraint Jacobians used to construct each subproblem. See the official
[JAX callback guide](https://docs.jax.dev/en/latest/external-callbacks.html)
for callback transformation semantics.

## Adapter, code generation, and FFI

This integration is a numerical model adapter. It passes changing arrays to
a generic compiled solver. It does not parse Python source or emit a new C
solver for each OpenSCvx problem.

Solver code generation is a separate feature. For example,
[OSQP code generation](https://osqp.org/docs/codegen/index.html) specializes
C code for a family of QPs with fixed dimensions; matrix updates retain
the selected sparsity pattern.
[QOCOGEN](https://qoco-org.github.io/qoco/qocogen/index.html) generates a
custom C solver for an SOCP family with fixed dimensions, cone layout, and
matrix sparsity patterns. Numerical parameters can still vary within that
family. Using ICORE's existing C/C++ library performs neither of these
code-generation steps, and it does not export OpenSCvx's nonlinear dynamics
or its full SCP algorithm as standalone C.

A [JAX FFI](https://docs.jax.dev/en/latest/ffi.html) wrapper could call the
native library directly from XLA and reduce Python callback overhead. That
would require a compiled wrapper, explicit workspace ownership, platform
registration, and batching rules. Differentiation and portable export would
need their own contracts. The current adapter uses the public Python API
and a host callback; it does not claim those FFI capabilities.
