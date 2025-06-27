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
from openscvx.ptr import PTR_init, PTR_subproblem, format_result
from openscvx.post_processing import propagate_trajectory_results
from openscvx.ocp import OptimalControlProblem
from openscvx import io
from openscvx.utils import stable_function_hash
from openscvx.backend.state import State, Free
from openscvx.backend.control import Control
from openscvx.backend.parameter import Parameter
from openscvx.results import OptimizationResults



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

        # SCP state variables
        self.scp_k = 0
        self.scp_J_tr = 1e2
        self.scp_J_vb = 1e2
        self.scp_J_vc = 1e2
        self.scp_trajs = []
        self.scp_controls = []
        self.scp_V_multi_shoot_traj = []

    def initialize(self):
        io.intro()

        # Print problem summary
        io.print_problem_summary(self.settings)

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
        function_hash = stable_function_hash(
            functions_to_hash,
            n_discretization_nodes=self.settings.scp.n,
            dt=self.settings.prp.dt,
            total_time=self.settings.sim.total_time,
            state_max=self.settings.sim.x.max,
            state_min=self.settings.sim.x.min,
            control_max=self.settings.sim.u.max,
            control_min=self.settings.sim.u.min
        )

        solver_dir = Path(".tmp")
        solver_dir.mkdir(parents=True, exist_ok=True)
        dis_solver_file = solver_dir / f"compiled_discretization_solver_{function_hash}.jax"
        prop_solver_file = solver_dir / f"compiled_propagation_solver_{function_hash}.jax"


        # Compile the solvers
        if not self.settings.dev.debug:
            if self.settings.sim.save_compiled:
                # Check if the compiled file already exists 
                try:
                    with open(dis_solver_file, "rb") as f:
                        serial_dis = f.read()
                    # Load the compiled code
                    self.discretization_solver = export.deserialize(serial_dis)
                    print("✓ Loaded existing discretization solver")
                except FileNotFoundError:
                    print("Compiling discretization solver...")
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
                    print("✓ Discretization solver compiled and saved")
            else:
                print("Compiling discretization solver (not saving/loading from disk)...")
                param_values = [param.value for _, param in self.params.items()]
                self.discretization_solver = export.export(jax.jit(self.discretization_solver))(
                    np.ones((self.settings.scp.n, self.settings.sim.n_states)),
                    np.ones((self.settings.scp.n, self.settings.sim.n_controls)),
                    *param_values
                )

        # Compile the discretization solver and save it
        dtau = 1.0 / (self.settings.scp.n - 1) 
        dt_max = self.settings.sim.u.max[self.settings.sim.idx_s][0] * dtau

        self.settings.prp.max_tau_len = int(dt_max / self.settings.prp.dt) + 2

        # Check if the compiled file already exists 
        if self.settings.sim.save_compiled:
            try:
                with open(prop_solver_file, "rb") as f:
                    serial_prop = f.read()
                # Load the compiled code
                self.propagation_solver = export.deserialize(serial_prop)
                print("✓ Loaded existing propagation solver")
            except FileNotFoundError:
                print("Compiling propagation solver...")
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
                print("✓ Propagation solver compiled and saved")
        else:
            print("Compiling propagation solver (not saving/loading from disk)...")
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
            self.propagation_solver = propagation_solver

        # Initialize the PTR loop
        print("Initializing the SCvx Subproblem Solver...")
        self.cpg_solve = PTR_init(
            self.params,
            self.optimal_control_problem,
            self.discretization_solver,
            self.settings,
        )
        print("✓ SCvx Subproblem Solver initialized")

        # Reset SCP state
        self.scp_k = 1
        self.scp_J_tr = 1e2
        self.scp_J_vb = 1e2
        self.scp_J_vc = 1e2
        self.scp_trajs = [self.settings.sim.x.guess]
        self.scp_controls = [self.settings.sim.u.guess]
        self.scp_V_multi_shoot_traj = []

        t_f_while = time.time()
        self.timing_init = t_f_while - t_0_while
        print("Total Initialization Time: ", self.timing_init)

        # Robust priming call for propagation_solver.call (no debug prints)
        try:
            x_0 = np.ones(self.settings.sim.x_prop.initial.shape, dtype=self.settings.sim.x_prop.initial.dtype)
            tau_grid = (0.0, 1.0)
            controls_current = np.ones((1, self.settings.sim.u.shape[0]), dtype=self.settings.sim.u.guess.dtype)
            controls_next = np.ones((1, self.settings.sim.u.shape[0]), dtype=self.settings.sim.u.guess.dtype)
            tau_init = np.array([[0.0]], dtype=np.float64)
            node = np.array([[0]], dtype=np.int64)
            idx_s_stop = self.settings.sim.idx_s.stop
            save_time = np.ones((self.settings.prp.max_tau_len,), dtype=np.float64)
            mask_padded = np.ones((self.settings.prp.max_tau_len,), dtype=bool)
            param_values = [np.ones_like(param.value) if hasattr(param.value, 'shape') else float(param.value) for _, param in self.params.items()]
            self.propagation_solver.call(
                x_0, tau_grid, controls_current, controls_next, tau_init, node, idx_s_stop, save_time, mask_padded, *param_values
            )
        except Exception as e:
            print(f"[Initialization] Priming propagation_solver.call failed: {e}")

        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_initialize.prof")

    def step(self):
        """Performs a single SCP iteration.
        
        This method is designed for real-time plotting and interactive optimization.
        It performs one complete SCP iteration including subproblem solving,
        state updates, and progress emission for real-time visualization.
        
        Returns:
            dict: Dictionary containing convergence status and current state
        """
        x = self.settings.sim.x
        u = self.settings.sim.u

        # Run the subproblem
        x_sol, u_sol, cost, J_total, J_vb_vec, J_vc_vec, J_tr_vec, prob_stat, V_multi_shoot, subprop_time, dis_time = PTR_subproblem(
            self.params.items(),
            self.cpg_solve,
            x,
            u,
            self.discretization_solver,
            self.optimal_control_problem,
            self.settings,
        )

        # Update state
        self.scp_V_multi_shoot_traj.append(V_multi_shoot)
        x.guess = x_sol
        u.guess = u_sol
        self.scp_trajs.append(x.guess)
        self.scp_controls.append(u.guess)

        self.scp_J_tr = np.sum(np.array(J_tr_vec))
        self.scp_J_vb = np.sum(np.array(J_vb_vec))
        self.scp_J_vc = np.sum(np.array(J_vc_vec))

        # Update weights
        self.settings.scp.w_tr = min(self.settings.scp.w_tr * self.settings.scp.w_tr_adapt, self.settings.scp.w_tr_max)
        if self.scp_k > self.settings.scp.cost_drop:
            self.settings.scp.lam_cost = self.settings.scp.lam_cost * self.settings.scp.cost_relax

        # Emit data
        self.emitter_function(
            {
                "iter": self.scp_k,
                "dis_time": dis_time * 1000.0,
                "subprop_time": subprop_time * 1000.0,
                "J_total": J_total,
                "J_tr": self.scp_J_tr,
                "J_vb": self.scp_J_vb,
                "J_vc": self.scp_J_vc,
                "cost": cost[-1],
                "prob_stat": prob_stat,
            }
        )

        # Increment counter
        self.scp_k += 1

        # Create a result dictionary for this step
        return {
            "converged": (self.scp_J_tr < self.settings.scp.ep_tr) and \
                         (self.scp_J_vb < self.settings.scp.ep_vb) and \
                         (self.scp_J_vc < self.settings.scp.ep_vc),
            "u": u,
            "x": x,
            "V_multi_shoot": V_multi_shoot
        }

    def solve(self, max_iters: Optional[int] = None, continuous: bool = False) -> OptimizationResults:
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
        
        k_max = max_iters if max_iters is not None else self.settings.scp.k_max
        
        while self.scp_k <= k_max:
            result = self.step()
            if result["converged"] and not continuous:
                break

        t_f_while = time.time()
        self.timing_solve = t_f_while - t_0_while

        while self.print_queue.qsize() > 0:
            time.sleep(0.1)

        # Print bottom footer for solver results as well as total computation time
        io.footer()

        # Disable the profiler
        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_solve.prof")

        return format_result(self, self.scp_k <= k_max)

    def post_process(self, result: OptimizationResults) -> OptimizationResults:
        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_post = time.time()
        result = propagate_trajectory_results(self.params, self.settings, result, self.propagation_solver)
        t_f_post = time.time()

        self.timing_post = t_f_post - t_0_post
        
        # Print results summary
        io.print_results_summary(result, self.timing_post, self.timing_init, self.timing_solve)

        # Disable the profiler
        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_postprocess.prof")
        return result
