# Getting Started

OpenSCvx is a JAX-based Python library for trajectory optimization using Successive Convexification (SCP). It provides a simple interface for formulating and solving trajectory optimization problems with continuous-time constraint satisfaction.

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

<details>
<summary>Stable</summary>

To grab the latest stable release simply run

```sh
pip install openscvx
```

to install OpenSCVx in your python environment.
</details>

<details>
<summary>Nightly</summary>

For the latest development version (nightly), clone the repository and install in editable mode:

```sh
# Clone the repo
git clone https://github.com/haynec/OpenSCvx.git
cd OpenSCvx

# Install in editable/development mode
pip install -e .
```

This will install the code as a package and allow you to make local changes.
</details>

### Dependencies

The main packages are:

- `cvxpy` - is used to formulate and solve the convex subproblems
- `jax` - is used for determining the Jacobians using automatic differentiation, vectorization, and ahead-of-time (AOT) compilation of the dynamics and their Jacobians 
- `numpy` - is used for numerical operations
- `diffrax` - is used for the numerical integration of the dynamics
- `termcolor` - is used for pretty command line output
- `plotly` - is used for all visualizations

These will be installed automatically, but can be installed via conda or pip if you are building from source.

#### GUI Dependencies (Optional)

For interactive 3D plotting and real-time visualization, additional packages are required:

- `pyqtgraph` - is used for interactive 3D plotting and real-time visualization
- `PyQt5` - provides the Qt5 GUI framework for pyqtgraph
- `scipy` - is used for spatial transformations in plotting functions
- `PyOpenGL` - provides OpenGL bindings for Python, required for 3D plotting
- `PyOpenGL_accelerate` - (optional) speeds up PyOpenGL

For local development:

```sh
pip install -e ".[gui]"
```

Or with conda:

```sh
conda env update -f environment.yml
```

The GUI features include:

- Interactive 3D trajectory visualization with `plot_animation_pyqtgraph()`
- SCP iteration animation with `plot_scp_animation_pyqtgraph()`
- Camera view animation with `plot_camera_animation_pyqtgraph()`
- Real-time optimization visualization in examples like `drone_racing_realtime.py`

#### CVXPYGen Dependencies (Optional)

For code generation and faster solver performance, CVXPYGen can be installed:

- `cvxpygen` - enables code generation for faster solver performance
- `qocogen` - custom solver backend for CVXPYGen (included with cvxpygen extras)

To install with CVXPYGen support:

```sh
pip install openscvx[cvxpygen]
```

Or for both GUI and CVXPYGen:

```sh
pip install openscvx[gui,cvxpygen]
```

CVXPYGen features include:

- Automatic C++ code generation for optimization problems
- Faster solver performance through compiled code for smaller problems
- Support for customized solver backends like QOCOGen

### Local Development

This git repository can be installed using https

```sh
git clone https://github.com/haynec/OpenSCvx.git
```

or ssh

```sh
git clone git@github.com/haynec/OpenSCvx.git
```

Dependencies can then be installed using Conda or Pip

<details>
<summary>Via Conda</summary>

1. Clone the repo using https or ssh
2. Install environment packages (this will take about a minute or two):
   ```sh
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```sh
   conda activate openscvx
   ```
</details>

<details>
<summary>Via pip</summary>

1. Prerequisites
   Python >= 3.9
2. Clone the repo using https or ssh
3. Create virtual environment (called `venv` here) and source it
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install environment packages:
   ```sh
   pip install -r requirements.txt
   ```
   
   Or install with optional dependencies:
   ```sh
   pip install -e ".[gui,cvxpygen]"
   ```
</details>

## Next Steps

- **[Examples](examples.md)**: Explore the comprehensive set of example problems
- **[Basic Problem Setup](Usage/basic_problem_setup.md)**: Learn how to set up your first optimization problem
- **[API Reference](api/)**: Detailed documentation of all classes and functions
- **[Citation](citation.md)**: Information for citing OpenSCvx in your research
