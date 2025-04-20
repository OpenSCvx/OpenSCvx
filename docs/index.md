# **OpenSCvx**

OpenSCvx :rocket: is an [JAX](https://github.com/jax-ml/jax)-based library for nonconvex trajectory planning in Python. 

!!! tip

    If you're new to **OpenSCvx**, then this page should tell you everything you need to get started. 

    - Go ahead and follow the installation instructions below.
    - Run through the "Tutorials" to get a feel for how to problems are instantiated. For a more complex problem, check out the "Drone LoS Guidance" example.
    - The "Basic API" goes in depth into each neccesary element to setup your problem in detail.
    - The "Advanced API" goes into more advanced features parameters and options to fine tune the performance of your problem. 


The aim of this work is hopefully to make the lives of a few graduate students out there easier as this was hell to implement when I was just getting started. 

We provide a simple interface to define the dynamics and constraints of your problem, all in continuous time no need for you to descritize your dynamics and constraints, while keeping the repo light enough that if the aspiring user wishes to delve a bit deeper and implement there own components they can do so with relative ease. 

## Features
- Free Final Time
- Fully-adaptive time dilation
- [Continuous-Time Constraint Satisfaction](https://arxiv.org/pdf/2404.16826)
- Vectorized and Ahead-of-Time ([AOT](https://docs.jax.dev/en/latest/aot.html)) Compiled Multishooting Discretization
- JAX Automatic Differentiation for Jacobians

## Installation
The main packages are:

- ```cvxpy``` - is used to formulate and solve the convex subproblems
- ```jax``` - is used for determining the Jacobians using automatic differentiation, vectorization, and ahead-of-time (AOT) compilation of the dynamics and their Jacobians 
- ```numpy``` - is used for numerical operations
- ```diffrax``` - is used for the numerical integration of the dynamics
- ```termcolor``` - is used for pretty command line output
- ```plotly``` - is used for all visualizations


These can be installed via conda or pip.

<details>
<summary>Via Conda (Recommended) </summary>

1. Clone the repo 
```sh
git clone https://github.com/haynec/OpenSCvx.git
```

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

<summary>Via Pip</summary>

0. Prerequisites 
```
Python >= 3.9
```

1. Clone the repo
```sh 
git clone https://github.com/haynec/OpenSCvx.git
```

2. Install environment packages:
```sh
pip install -r requirements.txt
```
</details>


## Examples
See `examples/` folder for several example trajectory optimization problems. To run a problem simply run `examples/main.py` with:

```bash
python3 -m examples.main
```
To change which example is run by `main` simply replace the `params` import line:

```python
# other imports
from examples.params.dr_vp import problem
# rest of code
```

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