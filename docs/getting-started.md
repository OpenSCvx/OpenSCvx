# **OpenSCvx**

OpenSCvx :rocket: is an [JAX](https://github.com/jax-ml/jax)-based library for nonconvex trajectory planning in Python.

!!! warning
    This repository is still in beta, here there be dragons :dragon:. A few pages are still under development and are denoted by :construction:.

!!! tip

    If you're new to **OpenSCvx**, then this page should tell you everything you need to get started. 

    - Go ahead and follow the installation instructions below.
    - Run through the "Tutorials" to get a feel for how to problems are instantiated. For a more complex problem, check out the "Drone LoS Guidance" example.
    - The "Basic Problem Setup" goes in depth into each neccesary element to setup your problem in detail.
    - The "Advanced Problem Setup" goes into more advanced features parameters and options to fine tune the performance of your problem. 

We provide a simple interface to define the dynamics and constraints of your problem, all in continuous time no need for you to discritize your dynamics and constraints, while keeping the repo light enough that if the aspiring user wishes to delve a bit deeper and implement there own components they can do so with relative ease. 

## Features
- Free Final Time
- Fully-adaptive time dilation
- [Continuous-Time Constraint Satisfaction](https://arxiv.org/pdf/2404.16826)
- Vectorized and Ahead-of-Time ([AOT](https://docs.jax.dev/en/latest/aot.html)) Compiled Multishooting Discretization
- JAX Automatic Differentiation for Jacobians

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
- Faster solver performance through compiled code
- Support for custom solver backends like QOCOGen

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

## Examples

See `examples/params/` folder for several example trajectory optimization problems.
To run a problem simply run any of the examples directly, for example:

```sh
python3 examples/params/brachistochrone.py
```
and adjust the plotting as needed.

Check out the problem definitions inside `examples/params` to see how to define your own problems.

## Citation
Please cite the following works if you use the repository,
```
@ARTICLE{hayner2025los,
        author={Hayner, Christopher R. and Carson III, John M. and Açıkmeşe, Behçet and Leung, Karen},
        journal={IEEE Robotics and Automation Letters}, 
        title={Continuous-Time Line-of-Sight Constrained Trajectory Planning for 6-Degree of Freedom Systems}, 
        year={2025},
        volume={},
        number={},
        pages={1-8},
        keywords={Robot sensing systems;Vectors;Vehicle dynamics;Line-of-sight propagation;Trajectory planning;Trajectory optimization;Quadrotors;Nonlinear dynamical systems;Heuristic algorithms;Convergence;Constrained Motion Planning;Optimization and Optimal Control;Aerial Systems: Perception and Autonomy},
        doi={10.1109/LRA.2025.3545299}}
```

```
@misc{elango2024ctscvx,
      title={Successive Convexification for Trajectory Optimization with Continuous-Time Constraint Satisfaction}, 
      author={Purnanand Elango and Dayou Luo and Abhinav G. Kamath and Samet Uzun and Taewan Kim and Behçet Açıkmeşe},
      year={2024},
      eprint={2404.16826},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2404.16826}, 
}
```

```
@misc{chari2025qoco,
  title = {QOCO: A Quadratic Objective Conic Optimizer with Custom Solver Generation},
  author = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  year = {2025},
  eprint = {2503.12658},
  archiveprefix = {arXiv},
  primaryclass = {math.OC},
}
```