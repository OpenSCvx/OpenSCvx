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

## Next steps

- Follow the [Users Guide](UsersGuide/00_introduction.md) from a first model to a solve.
- Browse runnable [examples](examples.md) organized by topic.
- Read the [API reference](Reference/index.md) generated from source.
