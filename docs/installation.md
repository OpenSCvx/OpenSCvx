---
title: Installation
description: >-
  How to install OpenSCvx from PyPI with uv or pip, or from source, including
  the optional extras and supported Python versions.
---

# Installation

OpenSCvx is published on PyPI as [`openscvx`](https://pypi.org/project/openscvx/)
and requires **Python 3.11 or newer**. Its core depends on JAX and CVXPY; the
optional extras below pull in plotting, additional solvers, dynamics backends,
and modeling features only when you need them.

## How to install OpenSCvx

=== "uv"

    ```sh
    uv pip install openscvx

    # Optional: plotting, cvxpygen, stljax, jaxlie, mujoco-mjx extras
    uv pip install "openscvx[plotting,cvxpygen,stl,lie,mjx]"
    ```

=== "pip"

    ```sh
    pip install openscvx

    # Optional: plotting, cvxpygen, stljax, jaxlie, mujoco-mjx extras
    pip install "openscvx[plotting,cvxpygen,stl,lie,mjx]"
    ```

=== "From source (editable)"

    ```sh
    git clone https://github.com/OpenSCvx/OpenSCvx.git
    cd OpenSCvx
    uv pip install -e .

    # Optional: plotting, cvxpygen, stljax, jaxlie, mujoco-mjx extras
    uv pip install -e ".[plotting,cvxpygen,stl,lie,mjx]"
    ```

Install from source if you want to modify the library or track the development
branch. Running the [examples](examples.md) needs the `plotting` extra as well,
because those scripts import Plotly and Viser at startup.

## Optional extras

Extras are installed with the `openscvx[extra1,extra2]` syntax shown above.

| Extra | Enables |
|-------|---------|
| `plotting` | 2D plots and 3D scenes (`matplotlib`, `plotly`, `viser`). |
| `cvxpygen` | Generated C solvers via `cvxpygen` + `qocogen`. |
| `qpax` | The `qpax` differentiable QP backend. |
| `moreau` | The `moreau` conic solver backend. |
| `stl` | Signal Temporal Logic constraints via `stljax`. |
| `lie` | Lie-group operators via `jaxlie`. |
| `mjx` | MuJoCo MJX dynamics (`mujoco`, `mujoco-mjx`). |
| `frax` | Fractal / `frax` dynamics support. |

### ICORE solver backend

The [ICORE](https://github.com/sametuzun781/ICORE) backend
(`solver={"backend": "icore"}` or `ox.ICOREPTRSolver()`) is not yet published
on PyPI, so there is no `icore` extra. Clone the repository and put it on
`PYTHONPATH`. The revision below is pinned by OpenSCvx's ICORE CI job:

```bash
git clone https://github.com/sametuzun781/ICORE.git
git -C ICORE checkout a63ebffb005725a02d2c3e4247fb73aec453e590
export PYTHONPATH="$PWD/ICORE${PYTHONPATH:+:$PYTHONPATH}"
```

ICORE runs in double precision only, so construct problems with
`float_dtype="float64"`. The default ICORE implementation is JAX; select
the native C or C++ implementation through `solver_args`:

```python
solver = {"backend": "icore", "solver_args": {"backend": "cpp"}}  # or "c"
problem = ox.Problem(..., solver=solver, float_dtype="float64")
```

The JAX implementation needs JAX and NumPy. Native execution additionally needs
a C/C++ compiler, `make`, and the build tools described in
[ICORE's build guide](https://github.com/sametuzun781/ICORE/blob/a63ebffb005725a02d2c3e4247fb73aec453e590/docs/BUILDING.md).
For the tested source revision, install CMake with:

```bash
python -m pip install "cmake>=3.18,<4"
```

Use your system's development toolchain on Linux, Xcode Command Line Tools on
macOS, or WSL on Windows. ICORE builds and caches its native library on first
use. Native solves run on the CPU, including when called from JAX.

See [ICORE solver backends](UnderTheHood/icore_backend.md) for the supported
model class, workspace reuse, batching, and export limits.

## Next steps

- Follow the [Users Guide](UsersGuide/00_introduction.md) from a first model to a solve.
- Browse runnable [examples](examples.md) organized by topic.
- Read the [API reference](Reference/index.md) generated from source.
