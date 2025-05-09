import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class DiscretizationConfig:
    dis_type: str = "FOH"
    custom_integrator: bool = True
    solver: str = "Tsit5"
    args: Dict = field(default_factory=dict)
    atol: float = 1e-3
    rtol: float = 1e-6

    """
    Configuration class for discretization settings.

    This class defines the parameters required for discretizing system dynamics.

    Main arguments:
    These are the arguments most commonly used day-to-day.

    Args:
        dis_type (str): The type of discretization to use (e.g., "FOH" for First-Order Hold). Defaults to "FOH".
        custom_integrator (bool): This enables our custom fixed-step RK45 algorthim. This tends to be faster then Diffrax but unless your going for speed, its reccomended to stick with Diffrax for robustness and other solver options. Defaults to False.
        solver (str): Not used if custom_integrator is enabled. Any choice of solver in Diffrax is valid, please refer here, [How to Choose a Solver](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/). Defaults to "Tsit5".

    Other arguments:
    These arguments are less frequently used, and for most purposes you shouldn't need to understand these.

    Args:
        args (Dict): Additional arguments to pass to the solver which can be found [here](https://docs.kidger.site/diffrax/api/diffeqsolve/). Defaults to an empty dictionary.
        atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
        rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
    """


@dataclass
class DevConfig:
    profiling: bool = False
    debug: bool = False
    printing: bool = True

    """
    Configuration class for development settings.

    This class defines the parameters used for development and debugging purposes.

    Main arguments:
    These are the arguments most commonly used day-to-day.

    Args:
        profiling (bool): Whether to enable profiling for performance analysis. Defaults to False.
        debug (bool): Disables all precompilation so you can place breakpoints and inspect values. Defaults to False.
    """


@dataclass
class ConvexSolverConfig:
    solver: str = "QOCO"
    solver_args: dict = field(default_factory=lambda: {"abstol": 1e-6, "reltol": 1e-9})
    cvxpygen: bool = False

    """
    Configuration class for convex solver settings.

    This class defines the parameters required for configuring a convex solver.

    These are the arguments most commonly used day-to-day. Generally I have found [QOCO](https://qoco-org.github.io/qoco/index.html) to be the most performant of the CVXPY solvers for these types of problems (I do have a bias as the author is from my group) and can handle up to SOCP's. 
    [CLARABEL](https://clarabel.org/stable/) is also a great option with feasibility checking and can handle a few more problem types.
    [CVXPYGen](https://github.com/cvxgrp/cvxpygen) is also great if your problem isn't too large and allows. I have found qocogen to be the most performant of the CVXPYGen solvers. 

    Args:
        solver (str):  The name of the CVXPY solver to use. A list of options can be found [here](https://www.cvxpy.org/tutorial/solvers/index.html). Defaults to "QOCO".
        solver_args (dict): Ensure you are using the correct arguments for your solver as they are not all common. Additional arguments to configure the solver, such as tolerances. 
                            Defaults to {"abstol": 1e-6, "reltol": 1e-9}.
        cvxpygen (bool): Whether to enable CVXPY code generation for the solver. Defaults to False.
    """


@dataclass
class PropagationConfig:
    inter_sample: int = 30
    dt: float = 0.1
    solver: str = "Dopri8"
    args: Dict = field(default_factory=dict)
    atol: float = 1e-3
    rtol: float = 1e-6

    """
    Configuration class for propagation settings.

    This class defines the parameters required for propagating the nonlinear system dynamics using the optimal control sequence.

    Main arguments:
    These are the arguments most commonly used day-to-day.
    
    Args:
        dt (float): The time step for propagation. Defaults to 0.1.
        inter_sample (int): How dense the propagation within multishot discretization should be.

    Other arguments:
    The solver should likley not to be changed as it is a high accuracy 8th order runga kutta method.
    
    Args:
        solver (str): The numerical solver to use for propagation (e.g., "Dopri8"). Defaults to "Dopri8".
        args (Dict): Additional arguments to pass to the solver. Defaults to an empty dictionary.
        atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
        rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
    """


@dataclass
class SimConfig:
    x_bar: np.ndarray
    u_bar: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray
    max_state: np.ndarray
    min_state: np.ndarray
    max_control: np.ndarray
    min_control: np.ndarray
    total_time: float
    idx_x_true: slice
    idx_u_true: slice
    idx_t: slice
    idx_y: slice
    idx_s: slice
    ctcs_node_intervals: list = None
    constraints_ctcs: List[callable] = field(
        default_factory=list
    )  # TODO (norrisg): clean this up, consider moving to dedicated `constraints` dataclass
    constraints_nodal: List[callable] = field(default_factory=list)
    n_states: int = None
    n_controls: int = None
    S_x: np.ndarray = None
    inv_S_x: np.ndarray = None
    c_x: np.ndarray = None
    S_u: np.ndarray = None
    inv_S_u: np.ndarray = None
    c_u: np.ndarray = None

    def __post_init__(self):
        self.n_states = len(self.max_state)
        self.n_controls = len(self.max_control)

        assert (
            len(self.initial_state.value) == self.n_states - (self.idx_y.stop - self.idx_y.start)
        ), f"Initial state must have {self.n_states - (self.idx_y.stop - self.idx_y.start)} elements"
        assert (
            len(self.final_state.value) == self.n_states - (self.idx_y.stop - self.idx_y.start)
        ), f"Final state must have {self.n_states - (self.idx_y.stop - self.idx_y.start)} elements"
        assert (
            self.max_state.shape[0] == self.n_states
        ), f"Max state must have {self.n_states} elements"
        assert (
            self.min_state.shape[0] == self.n_states
        ), f"Min state must have {self.n_states} elements"
        assert (
            self.max_control.shape[0] == self.n_controls
        ), f"Max control must have {self.n_controls} elements"
        assert (
            self.min_control.shape[0] == self.n_controls
        ), f"Min control must have {self.n_controls} elements"

        if self.S_x is None or self.c_x is None:
            self.S_x, self.c_x = get_affine_scaling_matrices(
                self.n_states, self.min_state, self.max_state
            )
            # Use the fact that S_x is diagonal to compute the inverse
            self.inv_S_x = np.diag(1 / np.diag(self.S_x))
        if self.S_u is None or self.c_u is None:
            self.S_u, self.c_u = get_affine_scaling_matrices(
                self.n_controls, self.min_control, self.max_control
            )
            self.inv_S_u = np.diag(1 / np.diag(self.S_u))


@dataclass
class ScpConfig:
    n: int = None
    k_max: int = 200
    w_tr: float = 1e0
    lam_vc: float = 1e0
    ep_tr: float = 1e-4
    ep_vb: float = 1e-4
    ep_vc: float = 1e-8
    lam_cost: float = 0.0
    lam_vb: float = 0.0
    uniform_time_grid: bool = False
    cost_drop: int = -1
    cost_relax: float = 1.0
    w_tr_adapt: float = 1.0
    w_tr_max: float = None
    w_tr_max_scaling_factor: float = None

    """
    Configuration class for Sequential Convex Programming (SCP).

    This class defines the parameters used to configure the SCP solver. You will very likely need to modify
    the weights for your problem. Please refer to my guide [here](https://haynec.github.io/openscvx/hyperparameter_tuning) for more information.

    Attributes:
        n (int): The number of discretization nodes. Defaults to `None`.
        k_max (int): The maximum number of SCP iterations. Defaults to 200.
        w_tr (float): The trust region weight. Defaults to 1.0.
        lam_vc (float): The penalty weight for virtual control. Defaults to 1.0.
        ep_tr (float): The trust region convergence tolerance. Defaults to 1e-4.
        ep_vb (float): The boundary constraint convergence tolerance. Defaults to 1e-4.
        ep_vc (float): The virtual constraint convergence tolerance. Defaults to 1e-8.
        lam_cost (float): The weight for original cost. Defaults to 0.0.
        lam_vb (float): The weight for virtual buffer. This is only used if there are nonconvex nodal constraints present. Defaults to 0.0.
        uniform_time_grid (bool): Whether to use a uniform time grid. TODO haynec add a link to the time dilation page. Defaults to `False`.
        cost_drop (int): The number of iterations to allow for cost stagnation before termination. Defaults to -1 (disabled).
        cost_relax (float): The relaxation factor for cost reduction. Defaults to 1.0.
        w_tr_adapt (float): The adaptation factor for the trust region weight. Defaults to 1.0.
        w_tr_max (float): The maximum allowable trust region weight. Defaults to `None`.
        w_tr_max_scaling_factor (float): The scaling factor for the maximum trust region weight. Defaults to `None`.
    """

    def __post_init__(self):
        keys_to_scale = ["w_tr", "lam_vc", "lam_cost", "lam_vb"]
        scale = max(getattr(self, key) for key in keys_to_scale)
        for key in keys_to_scale:
            setattr(self, key, getattr(self, key) / scale)

        if self.w_tr_max_scaling_factor is not None and self.w_tr_max is None:
            self.w_tr_max = self.w_tr_max_scaling_factor * self.w_tr


@dataclass
class Config:
    sim: SimConfig
    scp: ScpConfig
    cvx: ConvexSolverConfig
    dis: DiscretizationConfig
    prp: PropagationConfig
    dev: DevConfig

    def __post_init__(self):
        pass