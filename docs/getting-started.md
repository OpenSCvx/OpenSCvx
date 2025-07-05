# Getting Started

OpenSCvx is a JAX-based Python library for trajectory optimization using Successive Convexification (SCvx). It provides a simple interface for formulating and solving trajectory optimization problems with continuous-time constraint satisfaction.

!!! danger "Important"
    The library is currently in beta testing. Please report any issues on the GitHub repository.

## Key Features

- **JAX-based**: Automatic differentiation, vectorization, and compilation
- **Continuous-time constraints**: Support for path constraints that must be satisfied at all times
- **Successive Convexification**: Robust optimization algorithm for non-convex problems
- **Multiple constraint types**: Continuous-time, nodal, and boundary constraints
- **Interactive visualization**: 3D plotting and real-time optimization visualization
- **Code generation**: Automatic C++ code generation for optimization problems
- **Faster solver performance through compiled code for smaller problems
- **Support for customized solver backends like QOCOGen

## Installation

You can install OpenSCvx using pip. For the most common use case, which includes support for interactive plotting and code generation, you can install the library with the `gui` and `cvxpygen` extras:

```sh
pip install openscvx[gui,cvxpygen]
```

If you only need the core library without the optional features, you can run:

```sh
pip install openscvx
```

For the latest development version, you can clone the repository and install it in editable mode:

```sh
# Clone the repo
git clone https://github.com/haynec/OpenSCvx.git
cd OpenSCvx

# Install in editable mode with all optional dependencies
pip install -e ".[gui,cvxpygen]"
```

### Dependencies

OpenSCvx has a few optional dependency groups:

The core dependencies are installed automatically with `openscvx`:

- `cvxpy` - for convex optimization
- `jax` - for fast linear algebra, automatic differentiation, and vectorization
- `numpy` - for numerical operations
- `diffrax` - for automatic differentiation
- `termcolor` - for colored terminal output
- `plotly` - for basic interactive 3D plotting


- **`gui`**: For interactive 3D plotting and real-time visualization. This includes:
    - `pyqtgraph` - for realtime 3D plotting
    - `PyQt5` - for GUI
    - `scipy` - for spatial operations
    - `PyOpenGL` - for 3D plotting
    - `PyOpenGL_accelerate` (optional, for speed) - for 3D plotting

- **`cvxpygen`**: For C++ code generation, enabling faster solver performance on smaller problems. This includes:
    - `cvxpygen` - for C++ code generation
    - `qocogen` - fast SOCP solver

### Local Development

For setting up a local development environment, we recommend using Conda to manage environments.

<details>
<summary>Via Conda</summary>

1.  Clone the repository:
    ```sh
    git clone https://github.com/haynec/OpenSCvx.git
    cd OpenSCvx
    ```
2.  Create and activate the conda environment from the provided file:
    ```sh
    conda env create -f environment.yml
    conda activate openscvx
    ```
3.  Install the package in editable mode with all optional dependencies:
    ```sh
    pip install -e ".[gui,cvxpygen]"
    ```
</details>

<details>
<summary>Via pip and venv</summary>

1.  Clone the repository:
    ```sh
    git clone https://github.com/haynec/OpenSCvx.git
    cd OpenSCvx
    ```
2.  Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the package in editable mode with all optional dependencies:
    ```sh
    pip install -e ".[gui,cvxpygen]"
    ```
</details>

## Quick Example

Here's a simple example to get you started with OpenSCvx:

```python
import openscvx as ocvx
import jax.numpy as jnp

# Create a simple trajectory optimization problem
problem = ocvx.TrajOptProblem()

# Add state variables
x = problem.add_state("x", dim=3)
v = problem.add_state("v", dim=3)

# Add control variables
u = problem.add_control("u", dim=3)

# Define dynamics
def dynamics(x, v, u):
    return v, u

problem.set_dynamics(dynamics)

# Add constraints
problem.add_constraint(x[0] >= 0)  # Keep x position positive
problem.add_constraint(jnp.linalg.norm(u) <= 1.0)  # Control magnitude limit

# Set initial and final conditions
problem.set_initial_condition(x, [0, 0, 0])
problem.set_final_condition(x, [10, 0, 0])

# Solve the problem
result = problem.solve()
```

!!! note "Note"
    This is a basic example. For more complex problems, see the [Examples](examples.md) section.

## Next Steps

- **[Examples](examples.md)**: Explore the comprehensive set of example problems
- **[Basic Problem Setup](Usage/basic_problem_setup.md)**: Learn how to set up your first optimization problem
- **[Advanced Problem Setup](Usage/advanced_problem_setup.md)**: Learn how to set up a more complex optimization problem
- **[API Reference](Usage/api_state.md)**: Detailed documentation of all classes and functions
- **[Citation](citation.md)**: Information for citing OpenSCvx in your research
