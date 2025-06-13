import jax.numpy as jnp
from typing import List, Union, Optional
import queue
import threading
import time
from pathlib import Path
from copy import deepcopy

import cvxpy as cp
import jax
from jax import export, ShapeDtypeStruct
from functools import partial
import numpy as np

from openscvx.config import (
    ScpConfig,
    SimConfig,
    ConvexSolverConfig,
    DiscretizationConfig,
    PropagationConfig,
    DevConfig,
    Config,
)
from openscvx.dynamics import Dynamics
from openscvx.augmentation.dynamics_augmentation import build_augmented_dynamics
from openscvx.augmentation.ctcs import sort_ctcs_constraints
from openscvx.constraints.violation import get_g_funcs, CTCSViolation
from openscvx.discretization import get_discretization_solver
from openscvx.propagation import get_propagation_solver
from openscvx.constraints.ctcs import CTCSConstraint
from openscvx.constraints.nodal import NodalConstraint
from openscvx.ptr import PTR_init, PTR_main
from openscvx.post_processing import propagate_trajectory_results
from openscvx.ocp import OptimalControlProblem
from openscvx import io
from openscvx.utils import stable_function_hash
from openscvx.backend.state import State, Free
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter



# TODO: (norrisg) Decide whether to have constraints`, `cost`, alongside `dynamics`, ` etc.
class TrajOptProblem:
    def __init__(
        self,
        dynamics: Dynamics,
        constraints: List[Union[CTCSConstraint, NodalConstraint]],
        x: State,
        u: Control,
        N: int,
        idx_time: int,
        params: dict = {},
        dynamics_prop: callable = None,
        x_prop: State = None,
        scp: Optional[ScpConfig] = None,
        dis: Optional[DiscretizationConfig] = None,
        prp: Optional[PropagationConfig] = None,
        sim: Optional[SimConfig] = None,
        dev: Optional[DevConfig] = None,
        cvx: Optional[ConvexSolverConfig] = None,
        licq_min=0.0,
        licq_max=1e-4,
        time_dilation_factor_min=0.3,
        time_dilation_factor_max=3.0,
    ):
        """
        The primary class in charge of compiling and exporting the solvers 


        Args:
            dynamics (Dynamics): Dynamics function decorated with @dynamics
            constraints (List[Union[CTCSConstraint, NodalConstraint]]): List of constraints decorated with @ctcs or @nodal
            idx_time (int): Index of the time variable in the state vector
            N (int): Number of segments in the trajectory
            time_init (float): Initial time for the trajectory
            x_guess (jnp.ndarray): Initial guess for the state trajectory
            u_guess (jnp.ndarray): Initial guess for the control trajectory
            initial_state (BoundaryConstraint): Initial state constraint
            final_state (BoundaryConstraint): Final state constraint
            x_max (jnp.ndarray): Upper bound on the state variables
            x_min (jnp.ndarray): Lower bound on the state variables
            u_max (jnp.ndarray): Upper bound on the control variables
            u_min (jnp.ndarray): Lower bound on the control variables
            dynamics_prop: Propagation dynamics function decorated with @dynamics
            initial_state_prop: Propagation initial state constraint
            scp: SCP configuration object
            dis: Discretization configuration object
            prp: Propagation configuration object
            sim: Simulation configuration object
            dev: Development configuration object
            cvx: Convex solver configuration object

        Returns:
            None
        """

        self.params = params

        if dynamics_prop is None:
            dynamics_prop = dynamics
        
        if x_prop is None:
            x_prop = deepcopy(x)

        # TODO (norrisg) move this into some augmentation function, if we want to make this be executed after the init (i.e. within problem.initialize) need to rethink how problem is defined
        constraints_ctcs = []
        constraints_nodal = []
        for constraint in constraints:
            if isinstance(constraint, CTCSConstraint):
                constraints_ctcs.append(
                    constraint
                )
            elif isinstance(constraint, NodalConstraint):
                constraints_nodal.append(
                    constraint
                )
            else:
                raise ValueError(
                    f"Unknown constraint type: {type(constraint)}, All constraints must be decorated with @ctcs or @nodal"
                )

        constraints_ctcs, node_intervals, num_augmented_states = sort_ctcs_constraints(constraints_ctcs, N)

        # Index tracking
        idx_x_true = slice(0, x.shape[0])
        idx_x_true_prop = slice(0, x_prop.shape[0])
        idx_u_true = slice(0, u.shape[0])
        idx_constraint_violation = slice(
            idx_x_true.stop, idx_x_true.stop + num_augmented_states
        )
        idx_constraint_violation_prop = slice(
            idx_x_true_prop.stop, idx_x_true_prop.stop + num_augmented_states
        )

        idx_time_dilation = slice(idx_u_true.stop, idx_u_true.stop + 1)

        # check that idx_time is in the correct range
        assert idx_time >= 0 and idx_time < len(
            x.max
        ), "idx_time must be in the range of the state vector and non-negative"
        idx_time = slice(idx_time, idx_time + 1)

        # Create a new state object for the augmented states
        if num_augmented_states != 0:
            y = State(name="y", shape=(num_augmented_states,))
            y.initial = np.zeros((num_augmented_states,))
            y.final = np.array([Free(0)] * num_augmented_states)
            y.guess = np.zeros((N, num_augmented_states,))
            y.min = np.zeros((num_augmented_states,))
            y.max = licq_max * np.ones((num_augmented_states,))
            
            x.append(y, augmented=True)
            x_prop.append(y, augmented=True)

        s = Control(name="s", shape=(1,))
        s.min = np.array([time_dilation_factor_min * x.final[idx_time][0]])
        s.max = np.array([time_dilation_factor_max * x.final[idx_time][0]])
        s.guess = np.ones((N, 1)) * x.final[idx_time][0]

        
        u.append(s, augmented=True)

        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x=x,
                x_prop=x_prop,
                u=u,
                total_time=x.initial[idx_time][0],
                n_states=x.initial.shape[0],
                n_states_prop=x_prop.initial.shape[0],
                idx_x_true=idx_x_true,
                idx_x_true_prop=idx_x_true_prop,
                idx_u_true=idx_u_true,
                idx_t=idx_time,
                idx_y=idx_constraint_violation,
                idx_y_prop=idx_constraint_violation_prop,
                idx_s=idx_time_dilation,
                ctcs_node_intervals=node_intervals,
            )

        if scp is None:
            scp = ScpConfig(
                n=N,
                k_max=200,
                w_tr=1e1,  # Weight on the Trust Reigon
                lam_cost=1e1,  # Weight on the Nonlinear Cost
                lam_vc=1e2,  # Weight on the Virtual Control Objective
                lam_vb=0e0,  # Weight on the Virtual Buffer Objective (only for penalized nodal constraints)
                ep_tr=1e-4,  # Trust Region Tolerance
                ep_vb=1e-4,  # Virtual Control Tolerance
                ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
                cost_drop=4,  # SCP iteration to relax minimal final time objective
                cost_relax=0.5,  # Minimal Time Relaxation Factor
                w_tr_adapt=1.2,  # Trust Region Adaptation Factor
                w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
            )
        else:
            assert (
                self.settings.scp.n == N
            ), "Number of segments must be the same as in the config"

        if dev is None:
            dev = DevConfig()
        if cvx is None:
            cvx = ConvexSolverConfig()
        if prp is None:
            prp = PropagationConfig()

        sim.constraints_ctcs = constraints_ctcs
        sim.constraints_nodal = constraints_nodal

        ctcs_violation_funcs = get_g_funcs(constraints_ctcs)
        self.dynamics_augmented = build_augmented_dynamics(dynamics, ctcs_violation_funcs, idx_x_true, idx_u_true)
        self.dynamics_augmented_prop = build_augmented_dynamics(dynamics_prop, ctcs_violation_funcs, idx_x_true_prop, idx_u_true)

        self.settings = Config(
            sim=sim,
            scp=scp,
            dis=dis,
            dev=dev,
            cvx=cvx,
            prp=prp,
        )
        
        self.optimal_control_problem: cp.Problem = None
        self.discretization_solver: callable = None
        self.cpg_solve = None

        # set up emitter & thread only if printing is enabled
        if self.settings.dev.printing:
            self.print_queue      = queue.Queue()
            self.emitter_function = lambda data: self.print_queue.put(data)
            self.print_thread     = threading.Thread(
                target=io.intermediate,
                args=(self.print_queue, self.settings),
                daemon=True,
            )
            self.print_thread.start()
        else:
            # no-op emitter; nothing ever gets queued or printed
            self.emitter_function = lambda data: None


        self.timing_init = None
        self.timing_solve = None
        self.timing_post = None

    def initialize(self):
        io.intro()

        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Ensure parameter sizes and normalization are correct
        self.settings.scp.__post_init__()
        self.settings.sim.__post_init__()

        # Compile dynamics and jacobians
        self.dynamics_augmented.f = jax.vmap(self.dynamics_augmented.f, in_axes=(0, 0, 0, *(None,) * len(self.params)))
        self.dynamics_augmented.A = jax.vmap(self.dynamics_augmented.A, in_axes=(0, 0, 0, *(None,) * len(self.params)))
        self.dynamics_augmented.B = jax.vmap(self.dynamics_augmented.B, in_axes=(0, 0, 0, *(None,) * len(self.params)))
  
        self.dynamics_augmented_prop.f = jax.vmap(self.dynamics_augmented_prop.f, in_axes=(0, 0, 0, *(None,) * len(self.params)))

        for constraint in self.settings.sim.constraints_nodal:
            if not constraint.convex:
                # TODO: (haynec) switch to AOT instead of JIT
                constraint.g = jax.jit(constraint.g)
                constraint.grad_g_x = jax.jit(constraint.grad_g_x)
                constraint.grad_g_u = jax.jit(constraint.grad_g_u)

        # Generate solvers and optimal control problem
        self.discretization_solver = get_discretization_solver(self.dynamics_augmented, self.settings, self.params)
        self.propagation_solver = get_propagation_solver(self.dynamics_augmented_prop.f, self.settings, self.params)
        self.optimal_control_problem = OptimalControlProblem(self.settings)

        # Collect all relevant functions
        functions_to_hash = [self.dynamics_augmented.f, self.dynamics_augmented_prop.f]
        for constraint in self.settings.sim.constraints_nodal:
            functions_to_hash.append(constraint.func)
        for constraint in self.settings.sim.constraints_ctcs:
            functions_to_hash.append(constraint.func)

        # Get unique source-based hash
        function_hash = stable_function_hash(functions_to_hash)

        solver_dir = Path(".tmp")
        solver_dir.mkdir(parents=True, exist_ok=True)
        dis_solver_file = solver_dir / f"compiled_discretization_solver_{function_hash}.jax"
        prop_solver_file = solver_dir / f"compiled_propagation_solver_{function_hash}.jax"


        # Compile the solvers
        if not self.settings.dev.debug:
            # Check if the compiled file already exists 
            try:
                with open(dis_solver_file, "rb") as f:
                    serial_dis = f.read()
                # Load the compiled code
                self.discretization_solver = export.deserialize(serial_dis)
            except FileNotFoundError:
                # Extract parameter values and names in order
                param_values = [param.value for _, param in self.params.items()]
                
                self.discretization_solver = export.export(jax.jit(self.discretization_solver))(
                    np.ones((self.settings.scp.n, self.settings.sim.n_states)),
                    np.ones((self.settings.scp.n, self.settings.sim.n_controls)),
                    *param_values
                )
                # Serialize and Save the compiled code in a temp directory
                with open(dis_solver_file, "wb") as f:
                    f.write(self.discretization_solver.serialize())

        # Compile the discretization solver and save it
        dtau = 1.0 / (self.settings.scp.n - 1) 
        dt_max = self.settings.sim.u.max[self.settings.sim.idx_s][0] * dtau

        self.settings.prp.max_tau_len = int(dt_max / self.settings.prp.dt) + 2

        # Check if the compiled file already exists 
        try:
            with open(prop_solver_file, "rb") as f:
                serial_prop = f.read()
            # Load the compiled code
            self.propagation_solver = export.deserialize(serial_prop)
        except FileNotFoundError:
            # Extract parameter values and names in order
            param_values = [param.value for _, param in self.params.items()]

            propagation_solver = export.export(jax.jit(self.propagation_solver))(
                np.ones((self.settings.sim.n_states_prop)),                # x_0
                (0.0, 0.0),                                                # time span
                np.ones((1, self.settings.sim.n_controls)),                # controls_current
                np.ones((1, self.settings.sim.n_controls)),                # controls_next
                np.ones((1, 1)),                                           # tau_0
                np.ones((1, 1)).astype("int"),                             # segment index
                0,                                                         # idx_s_stop
                np.ones((self.settings.prp.max_tau_len,)),                 # save_time (tau_cur_padded)
                np.ones((self.settings.prp.max_tau_len,), dtype=bool),     # mask_padded (boolean mask)
                *param_values,                                             # additional parameters
            )

            # Serialize and Save the compiled code in a temp directory
            self.propagation_solver = propagation_solver

            with open(prop_solver_file, "wb") as f:
                f.write(self.propagation_solver.serialize())

        # Initialize the PTR loop
        self.cpg_solve = PTR_init(
            self.params,
            self.optimal_control_problem,
            self.discretization_solver,
            self.settings,
        )

        t_f_while = time.time()
        self.timing_init = t_f_while - t_0_while
        print("Total Initialization Time: ", self.timing_init)

        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_initialize.prof")

    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.settings.scp.__post_init__()
        self.settings.sim.__post_init__()

        if self.optimal_control_problem is None or self.discretization_solver is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve()"
            )

        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Print top header for solver results
        io.header()

        result = PTR_main(
            self.params,
            self.settings,
            self.optimal_control_problem,
            self.discretization_solver,
            self.cpg_solve,
            self.emitter_function,
        )

        t_f_while = time.time()
        self.timing_solve = t_f_while - t_0_while

        while self.print_queue.qsize() > 0:
            time.sleep(0.1)

        # Print bottom footer for solver results as well as total computation time
        io.footer(self.timing_solve)

        # Disable the profiler
        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_solve.prof")

        return result

    def post_process(self, result):
        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_post = time.time()
        result = propagate_trajectory_results(self.params, self.settings, result, self.propagation_solver)
        t_f_post = time.time()

        self.timing_post = t_f_post - t_0_post
        print("Total Post Processing Time: ", self.timing_post)

        # Disable the profiler
        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_postprocess.prof")
        return result
