import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from openscvx.dynamics import Dynamics


def get_affine_scaling_matrices(n, minimum, maximum):
    S = np.diag(np.maximum(np.ones(n), abs(minimum - maximum) / 2))
    c = (maximum + minimum) / 2
    return S, c


@dataclass
class DiscretizationConfig:
    def __init__(self, 
                 dis_type: str = "FOH", 
                 custom_integrator: bool = True, 
                 solver: str = "Tsit5", 
                 args: Dict = None, 
                 atol: float = 1e-3, 
                 rtol: float = 1e-6):
        """
        Configuration class for discretization settings.

        This class defines the parameters required for discretizing system dynamics.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            dis_type (str): The type of discretization to use (e.g., "FOH" for First-Order Hold). Defaults to "FOH".
            custom_integrator (bool): Whether to use a custom integrator for discretization. Defaults to True.
            solver (str): The numerical solver to use for integration (e.g., "Tsit5"). Defaults to "Tsit5".

        Other arguments:
        These arguments are less frequently used, and for most purposes you shouldn't need to understand these.

        Args:
            args (Dict): Additional arguments to pass to the solver. Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """"""
        Configuration class for discretization settings.

        This class defines the parameters required for discretizing system dynamics.

        Main arguments:
        These are the arguments most commonly used day-to-day.

            dis_type (str): The type of discretization to use (e.g., "FOH" for First-Order Hold). Defaults to "FOH".
            custom_integrator (bool): Whether to use a custom integrator for discretization. Defaults to True.
            solver (str): The numerical solver to use for integration (e.g., "Tsit5"). Defaults to "Tsit5".

        Other arguments:
        These arguments are less frequently used, and for most purposes you shouldn't need to understand these.

            args (Dict): Additional arguments to pass to the solver. Defaults to an empty dictionary.
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
    def __init__(self, 
                 profiling: bool = False, 
                 debug: bool = False,
                 debug_printing: bool = True):
        """
        Configuration class for development settings.

        This class defines the parameters used for development and debugging purposes.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            profiling (bool): Whether to enable profiling for performance analysis. Defaults to False.
            debug (bool): Whether to enable debug mode for additional logging and error checks. Defaults to False.
        """
        self.profiling = profiling
        self.debug = debug    
        self.debug_printing = debug_printing


@dataclass
class ConvexSolverConfig:
    def __init__(self, 
                 solver: str = "QOCO", 
                 solver_args: dict = None, 
                 cvxpygen: bool = False):
        """
        Configuration class for convex solver settings.

        This class defines the parameters required for configuring a convex solver.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            solver (str): The name of the convex solver to use (e.g., "QOCO"). Defaults to "QOCO".
            solver_args (dict): Additional arguments to configure the solver, such as tolerances. 
                                Defaults to {"abstol": 1e-6, "reltol": 1e-9}.
            cvxpygen (bool): Whether to enable CVXPY code generation for the solver. Defaults to False.
        """
        self.solver = solver
        self.solver_args = solver_args if solver_args is not None else {"abstol": 1e-6, "reltol": 1e-9}
        self.cvxpygen = cvxpygen


@dataclass
class PropagationConfig:
    def __init__(self, 
                 inter_sample: int = 30, 
                 dt: float = 0.1, 
                 solver: str = "Dopri8", 
                 args: Dict = None, 
                 atol: float = 1e-3, 
                 rtol: float = 1e-6):
        """
        Configuration class for propagation settings.

        This class defines the parameters required for propagating the system dynamics.

        Main arguments:
        These are the arguments most commonly used day-to-day.
        
        Args:
            inter_sample (int): The number of intermediate samples between control updates. Defaults to 30.
            dt (float): The time step for propagation. Defaults to 0.1.
            solver (str): The numerical solver to use for propagation (e.g., "Dopri8"). Defaults to "Dopri8".

        Other arguments:
        These arguments are less frequently used, and for most purposes you shouldn't need to understand these.
        
        Args:
            args (Dict): Additional arguments to pass to the solver. Defaults to an empty dictionary.
            atol (float): Absolute tolerance for the solver. Defaults to 1e-3.
            rtol (float): Relative tolerance for the solver. Defaults to 1e-6.
        """
        self.inter_sample = inter_sample
        self.dt = dt
        self.solver = solver
        self.args = args if args is not None else {}
        self.atol = atol
        self.rtol = rtol

@dataclass
class SimConfig:
    def __init__(self,
                 x_bar: np.ndarray,
                 u_bar: np.ndarray,
                 initial_state: np.ndarray,
                 final_state: np.ndarray,
                 max_state: np.ndarray,
                 min_state: np.ndarray,
                 max_control: np.ndarray,
                 min_control: np.ndarray,
                 total_time: float,
                 n_states: int = None,
                 n_controls: int = None,
                 S_x: np.ndarray = None,
                 inv_S_x: np.ndarray = None,
                 c_x: np.ndarray = None,
                 S_u: np.ndarray = None,
                 inv_S_u: np.ndarray = None,
                 c_u: np.ndarray = None):
        """
        Configuration class for simulation settings.

        This class defines the parameters required for simulating a trajectory optimization problem.

        Main arguments:
        These are the arguments most commonly used day-to-day.

        Args:
            x_bar (np.ndarray): The nominal state trajectory.
            u_bar (np.ndarray): The nominal control trajectory.
            initial_state (np.ndarray): The initial state of the system.
            final_state (np.ndarray): The final state of the system.
            max_state (np.ndarray): The maximum allowable state values.
            min_state (np.ndarray): The minimum allowable state values.
            max_control (np.ndarray): The maximum allowable control values.
            min_control (np.ndarray): The minimum allowable control values.
            total_time (float): The total simulation time.

        Other arguments:
        These arguments are less frequently used, and for most purposes you shouldn't need to understand these. All of these are optional.

        Args:
            n_states (int, optional): The number of state variables. Defaults to `None`.
            n_controls (int, optional): The number of control variables. Defaults to `None`.
            S_x (np.ndarray, optional): State scaling matrix. Defaults to `None`.
            inv_S_x (np.ndarray, optional): Inverse of the state scaling matrix. Defaults to `None`.
            c_x (np.ndarray, optional): State offset vector. Defaults to `None`.
            S_u (np.ndarray, optional): Control scaling matrix. Defaults to `None`.
            inv_S_u (np.ndarray, optional): Inverse of the control scaling matrix. Defaults to `None`.
            c_u (np.ndarray, optional): Control offset vector. Defaults to `None`.
        """


        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        
 
    def __post_init__(self):
        self.n_states = len(self.max_state)
        self.n_controls = len(self.max_control)

        assert (
            len(self.initial_state.value) == self.n_states - 1
        ), f"Initial state must have {self.n_states - 1} elements"
        assert (
            len(self.final_state.value) == self.n_states - 1
        ), f"Final state must have {self.n_states - 1} elements"
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
    def __init__(self, n: int = None,
                 k_max: int = 200,
                 w_tr: float = 1e0,
                 lam_vc: float = 1e0,
                 ep_tr: float = 1e-4,
                 ep_vb: float = 1e-4,
                 ep_vc: float = 1e-8,
                 lam_cost: float = 0.0,
                 lam_vb: float = 0.0,
                 uniform_time_grid: bool = False,
                 cost_drop: int = -1,
                 cost_relax: float = 1.0,
                 w_tr_adapt: float = 1.0,
                 w_tr_max: float = None,
                 w_tr_max_scaling_factor: float = None):
            """
            Configuration class for Sequential Convex Programming (SCP).

            This class defines the parameters used to configure the SCP solver for trajectory optimization problems.

            Attributes:
                n (int): The number of decision variables. Defaults to `None`.
                k_max (int): The maximum number of SCP iterations. Defaults to 200.
                w_tr (float): The trust region weight. Defaults to 1.0.
                lam_vc (float): The penalty weight for virtual constraints. Defaults to 1.0.
                ep_tr (float): The trust region convergence tolerance. Defaults to 1e-4.
                ep_vb (float): The boundary constraint convergence tolerance. Defaults to 1e-4.
                ep_vc (float): The virtual constraint convergence tolerance. Defaults to 1e-8.
                lam_cost (float): The weight for cost relaxation. Defaults to 0.0.
                lam_vb (float): The weight for boundary constraint relaxation. Defaults to 0.0.
                uniform_time_grid (bool): Whether to use a uniform time grid. Defaults to `False`.
                cost_drop (int): The number of iterations to allow for cost stagnation before termination. Defaults to -1 (disabled).
                cost_relax (float): The relaxation factor for cost reduction. Defaults to 1.0.
                w_tr_adapt (float): The adaptation factor for the trust region weight. Defaults to 1.0.
                w_tr_max (float): The maximum allowable trust region weight. Defaults to `None`.
                w_tr_max_scaling_factor (float): The scaling factor for the maximum trust region weight. Defaults to `None`.
            """
            for key, value in locals().items():
                if key != "self":
                    setattr(self, key, value)

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
    dyn: Dynamics
    cvx: ConvexSolverConfig
    dis: DiscretizationConfig
    prp: PropagationConfig
    dev: DevConfig

    def __post_init__(self):
        pass