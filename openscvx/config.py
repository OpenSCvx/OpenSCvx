import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from openscvx.backend.state import State
from openscvx.backend.control import Control


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class DiscretizationConfig:

    def __init__(
        self,
        dis_type: str = "FOH",
        custom_integrator: bool = False,
        solver: str = "Tsit5",
        args: Dict = None,
        atol: float = 1e-3,
        rtol: float = 1e-6,
    ):
        """
        Configuration class for discretization settings.

        This class defines the parameters required for discretizing system dynamics.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            dis_type (str): The type of discretization to use (e.g., "FOH" for First-Order Hold). Defaults to "FOH".
            custom_integrator (bool): This enables our custom fixed-step RK45 algorithm. This tends to be faster than Diffrax but unless you're going for speed, it's recommended to stick with Diffrax for robustness and other solver options. Defaults to False.
            solver (str): Not used if custom_integrator is enabled. Any choice of solver in Diffrax is valid, please refer here, [How to Choose a Solver](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/). Defaults to "Tsit5".

        Other arguments:
        These arguments are less frequently used, and for most purposes you shouldn't need to understand these.

        Args:
            args (Dict): Additional arguments to pass to the solver which can be found [here](https://docs.kidger.site/diffrax/api/diffeqsolve/). Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """
        self.dis_type = dis_type
        self.custom_integrator = custom_integrator
        self.solver = solver
        self.args = args if args is not None else {}
        self.atol = atol
        self.rtol = rtol

@dataclass
class DevConfig:

    def __init__(self, profiling: bool = False, debug: bool = False, printing: bool = True):
        """
        Configuration class for development settings.

        This class defines the parameters used for development and debugging purposes.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            profiling (bool): Whether to enable profiling for performance analysis. Defaults to False.
            debug (bool): Disables all precompilation so you can place breakpoints and inspect values. Defaults to False.
            printing (bool): Whether to enable printing during development. Defaults to True.
        """
        self.profiling = profiling
        self.debug = debug
        self.printing = printing

@dataclass
class ConvexSolverConfig:
    def __init__(
        self,
        solver: str = "QOCO",
        solver_args: dict = {"abstol": 1e-6, "reltol": 1e-9, "enforce_dpp": True},
        cvxpygen: bool = False,
        cvxpygen_override: bool = False,
    ):
        """
        Configuration class for convex solver settings.

        This class defines the parameters required for configuring a convex solver.

        These are the arguments most commonly used day-to-day. Generally I have found [QOCO](https://qoco-org.github.io/qoco/index.html)
        to be the most performant of the CVXPY solvers for these types of problems (I do have a bias as the author is from my group)
        and can handle up to SOCP's. [CLARABEL](https://clarabel.org/stable/) is also a great option with feasibility checking and
        can handle a few more problem types. [CVXPYGen](https://github.com/cvxgrp/cvxpygen) is also great if your problem isn't too large.
        I have found qocogen to be the most performant of the CVXPYGen solvers.

        Args:
            solver (str): The name of the CVXPY solver to use. A list of options can be found
                        [here](https://www.cvxpy.org/tutorial/solvers/index.html). Defaults to "QOCO".
            solver_args (dict, optional): Ensure you are using the correct arguments for your solver as they are not all common.
                                        Additional arguments to configure the solver, such as tolerances.
                                        Defaults to {"abstol": 1e-6, "reltol": 1e-9}.
            cvxpygen (bool): Whether to enable CVXPY code generation for the solver. Defaults to False.
        """
        self.solver = solver
        self.solver_args = solver_args if solver_args is not None else {"abstol": 1e-6, "reltol": 1e-9}
        self.cvxpygen = cvxpygen
        self.cvxpygen_override = cvxpygen_override


@dataclass
class PropagationConfig:
    def __init__(
        self,
        inter_sample: int = 30,
        dt: float = 0.01,
        solver: str = "Dopri8",
        max_tau_len: int = 1000,
        args: Optional[Dict] = None,
        atol: float = 1e-3,
        rtol: float = 1e-6,
    ):
        """
        Configuration class for propagation settings.

        This class defines the parameters required for propagating the nonlinear system dynamics
        using the optimal control sequence.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Other arguments:
        The solver should likely not be changed as it is a high accuracy 8th-order Runge-Kutta method.

        Args:
            inter_sample (int): How dense the propagation within multishot discretization should be. Defaults to 30.
            dt (float): The time step for propagation. Defaults to 0.1.
            solver (str): The numerical solver to use for propagation (e.g., "Dopri8"). Defaults to "Dopri8".
            max_tau_len (int): The maximum length of the time vector for propagation. Defaults to 1000.
            args (Dict, optional): Additional arguments to pass to the solver. Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """

        self.inter_sample = inter_sample
        self.dt = dt
        self.solver = solver
        self.max_tau_len = max_tau_len
        self.args = args if args is not None else {}
        self.atol = atol
        self.rtol = rtol

@dataclass(init=False)
class SimConfig:
    # No class-level field declarations
    
    def __init__(
        self,
        x: State,
        x_prop: State,
        u: Control,
        total_time: float,
        idx_x_true: slice,
        idx_x_true_prop: slice,
        idx_u_true: slice,
        idx_t: slice,
        idx_y: slice,
        idx_y_prop: slice,
        idx_s: slice,
        save_compiled: bool = True,
        ctcs_node_intervals: Optional[list] = None,
        constraints_ctcs: Optional[List[Callable]] = None,
        constraints_nodal: Optional[List[Callable]] = None,
        n_states: Optional[int] = None,
        n_states_prop: Optional[int] = None,
        n_controls: Optional[int] = None,
        scaling_x_overrides: Optional[list] = None,
        scaling_u_overrides: Optional[list] = None,
    ):
        """
        Configuration class for simulation settings.

        This class defines the parameters required for simulating a trajectory optimization problem.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            x (State): State object, must have .min and .max attributes for bounds.
            x_prop (State): Propagation state object, must have .min and .max attributes for bounds.
            u (Control): Control object, must have .min and .max attributes for bounds.
            total_time (float): The total simulation time.
            idx_x_true (slice): Slice for true state indices.
            idx_x_true_prop (slice): Slice for true propagation state indices.
            idx_u_true (slice): Slice for true control indices.
            idx_t (slice): Slice for time index.
            idx_y (slice): Slice for constraint violation indices.
            idx_y_prop (slice): Slice for propagation constraint violation indices.
            idx_s (slice): Slice for time dilation index.
            save_compiled (bool): If True, save and reuse compiled solver functions. Defaults to True.
            ctcs_node_intervals (list, optional): Node intervals for CTCS constraints.
            constraints_ctcs (list, optional): List of CTCS constraints.
            constraints_nodal (list, optional): List of nodal constraints.
            n_states (int, optional): The number of state variables. Defaults to `None` (inferred from x.max).
            n_states_prop (int, optional): The number of propagation state variables. Defaults to `None` (inferred from x_prop.max).
            n_controls (int, optional): The number of control variables. Defaults to `None` (inferred from u.max).
            scaling_x_overrides (list, optional): List of (upper_bound, lower_bound, idx) for custom state scaling. Each can be scalar or array, idx can be int, list, or slice.
            scaling_u_overrides (list, optional): List of (upper_bound, lower_bound, idx) for custom control scaling. Each can be scalar or array, idx can be int, list, or slice.

        Note:
            You can specify custom scaling for specific states/controls using scaling_x_overrides and scaling_u_overrides. Any indices not covered by overrides will use the default min/max bounds.
        """
        # Assign all arguments to self
        self.x = x
        self.x_prop = x_prop
        self.u = u
        self.total_time = total_time
        self.idx_x_true = idx_x_true
        self.idx_x_true_prop = idx_x_true_prop
        self.idx_u_true = idx_u_true
        self.idx_t = idx_t
        self.idx_y = idx_y
        self.idx_y_prop = idx_y_prop
        self.idx_s = idx_s
        self.save_compiled = save_compiled
        self.ctcs_node_intervals = ctcs_node_intervals
        self.constraints_ctcs = constraints_ctcs if constraints_ctcs is not None else []
        self.constraints_nodal = constraints_nodal if constraints_nodal is not None else []
        self.n_states = n_states
        self.n_states_prop = n_states_prop
        self.n_controls = n_controls
        self.scaling_x_overrides = scaling_x_overrides
        self.scaling_u_overrides = scaling_u_overrides

        # Then call post init logic
        self.__post_init__()

    def __post_init__(self):
        self.n_states = len(self.x.max)
        self.n_controls = len(self.u.max)

        # Helper to apply overrides
        def apply_overrides(size, overrides, min_arr, max_arr):
            upper = np.array(max_arr, dtype=float)
            lower = np.array(min_arr, dtype=float)
            if overrides is not None:
                for ub, lb, idx in overrides:
                    if isinstance(idx, int):
                        idxs = [idx]
                    elif isinstance(idx, slice):
                        idxs = list(range(*idx.indices(size)))
                    else:
                        idxs = list(idx)
                    if np.isscalar(ub):
                        ub_vals = [ub] * len(idxs)
                    else:
                        ub_vals = ub
                    if np.isscalar(lb):
                        lb_vals = [lb] * len(idxs)
                    else:
                        lb_vals = lb
                    for i, uval, lval in zip(idxs, ub_vals, lb_vals):
                        upper[i] = uval
                        lower[i] = lval
            return upper, lower

        # State scaling
        min_x = np.array(self.x.min)
        max_x = np.array(self.x.max)
        upper_x, lower_x = apply_overrides(self.n_states, self.scaling_x_overrides, min_x, max_x)
        S_x, c_x = get_affine_scaling_matrices(self.n_states, lower_x, upper_x)
        self.S_x = S_x
        self.c_x = c_x
        self.inv_S_x = np.diag(1 / np.diag(self.S_x))

        # Control scaling
        min_u = np.array(self.u.min)
        max_u = np.array(self.u.max)
        upper_u, lower_u = apply_overrides(self.n_controls, self.scaling_u_overrides, min_u, max_u)
        S_u, c_u = get_affine_scaling_matrices(self.n_controls, lower_u, upper_u)
        self.S_u = S_u
        self.c_u = c_u
        self.inv_S_u = np.diag(1 / np.diag(self.S_u))



@dataclass
class ScpConfig:

    def __init__(
        self,
        n: Optional[int] = None,
        k_max: int = 200,
        w_tr: float = 1.0,
        lam_vc: float = 1.0,
        ep_tr: float = 1e-4,
        ep_vb: float = 1e-4,
        ep_vc: float = 1e-8,
        lam_cost: float = 0.0,
        lam_vb: float = 0.0,
        uniform_time_grid: bool = False,
        cost_drop: int = -1,
        cost_relax: float = 1.0,
        w_tr_adapt: float = 1.0,
        w_tr_max: Optional[float] = None,
        w_tr_max_scaling_factor: Optional[float] = None,
    ):
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
            uniform_time_grid (bool): Whether to use a uniform time grid. Defaults to `False`.
            cost_drop (int): The number of iterations to allow for cost stagnation before termination. Defaults to -1 (disabled).
            cost_relax (float): The relaxation factor for cost reduction. Defaults to 1.0.
            w_tr_adapt (float): The adaptation factor for the trust region weight. Defaults to 1.0.
            w_tr_max (float): The maximum allowable trust region weight. Defaults to `None`.
            w_tr_max_scaling_factor (float): The scaling factor for the maximum trust region weight. Defaults to `None`.
        """
        self.n = n
        self.k_max = k_max
        self.w_tr = w_tr
        self.lam_vc = lam_vc
        self.ep_tr = ep_tr
        self.ep_vb = ep_vb
        self.ep_vc = ep_vc
        self.lam_cost = lam_cost
        self.lam_vb = lam_vb
        self.uniform_time_grid = uniform_time_grid
        self.cost_drop = cost_drop
        self.cost_relax = cost_relax
        self.w_tr_adapt = w_tr_adapt
        self.w_tr_max = w_tr_max
        self.w_tr_max_scaling_factor = w_tr_max_scaling_factor

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