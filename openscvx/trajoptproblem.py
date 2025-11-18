import os
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union

import jax

os.environ["EQX_ON_ERROR"] = "nan"

from openscvx import io
from openscvx.caching import (
    get_solver_cache_paths,
    load_or_compile_discretization_solver,
    load_or_compile_propagation_solver,
    prime_propagation_solver,
)
from openscvx.config import (
    Config,
    ConvexSolverConfig,
    DevConfig,
    DiscretizationConfig,
    PropagationConfig,
    ScpConfig,
    SimConfig,
)
from openscvx.discretization import get_discretization_solver
from openscvx.ocp import (
    OptimalControlProblem,
    create_cvxpy_variables,
    lower_convex_constraints,
)
from openscvx.post_processing import propagate_trajectory_results
from openscvx.propagation import get_propagation_solver
from openscvx.ptr import PTR_init, PTR_step, format_result
from openscvx.results import OptimizationResults
from openscvx.symbolic.builder import preprocess_symbolic_problem
from openscvx.symbolic.expr import CTCS, Constraint
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.lower import lower_symbolic_expressions
from openscvx.time import Time

if TYPE_CHECKING:
    import cvxpy as cp


class _ParameterDict(dict):
    """Dictionary that syncs to both internal _parameters dict and CVXPy parameters.

    This allows users to naturally update parameters like:
        problem.parameters["obs_radius"] = 2.0

    Changes automatically propagate to:
    1. Internal _parameters dict (plain dict for JAX)
    2. CVXPy parameters (for optimization)
    """

    def __init__(self, problem, internal_dict, *args, **kwargs):
        self._problem = problem
        self._internal_dict = internal_dict  # Reference to plain dict for JAX
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Sync to internal dict for JAX
        self._internal_dict[key] = value
        # Sync to CVXPy if it exists
        if self._problem.cvxpy_params is not None and key in self._problem.cvxpy_params:
            self._problem.cvxpy_params[key].value = value

    def update(self, other=None, **kwargs):
        """Update multiple parameters and sync to internal dict and CVXPy."""
        if other is not None:
            if hasattr(other, "items"):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value


class TrajOptProblem:
    def __init__(
        self,
        dynamics: dict,
        constraints: List[Union[Constraint, CTCS]],
        states: List[State],
        controls: List[Control],
        N: int,
        time: Time,
        dynamics_prop: Optional[dict] = None,
        states_prop: Optional[List[State]] = None,
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
            dynamics (dict): Dictionary mapping state names to their dynamics expressions.
                Each key should be a state name, and each value should be an Expr
                representing the derivative of that state.
            constraints (List[Union[CTCSConstraint, NodalConstraint]]):
                List of constraints decorated with @ctcs or @nodal
            states (List[State]): List of State objects representing the state variables.
                May optionally include a State named "time" (see time parameter below).
            controls (List[Control]): List of Control objects representing the control variables
            N (int): Number of segments in the trajectory
            time (Time): Time configuration object with initial, final, min, max.
                Required. If including a "time" state in states, the Time object will be ignored
                and time properties should be set on the time State object instead.
            dynamics_prop (dict, optional): Dictionary mapping EXTRA state names to their
                dynamics expressions for propagation. Only specify additional states beyond
                optimization states (e.g., {"distance": speed}). Do NOT duplicate optimization
                state dynamics here.
            states_prop (List[State], optional): List of EXTRA State objects for propagation only.
                Only specify additional states beyond optimization states. Used with dynamics_prop.
            scp: SCP configuration object
            dis: Discretization configuration object
            prp: Propagation configuration object
            sim: Simulation configuration object
            dev: Development configuration object
            cvx: Convex solver configuration object
            licq_min: Minimum LICQ constraint value
            licq_max: Maximum LICQ constraint value
            time_dilation_factor_min: Minimum time dilation factor
            time_dilation_factor_max: Maximum time dilation factor

        Returns:
            None

        Note:
            There are two approaches for handling time:
            1. Auto-create (simple): Don't include "time" in states, provide Time object
            2. User-provided (for time-dependent constraints): Include "time" State in states and
               in dynamics dict, don't provide Time object
        """

        # Step 1: Symbolic Preprocessing & Augmentation
        (
            dynamics_aug,
            x_aug,
            u_aug,
            constraints_ctcs,
            constraints_nodal,
            constraints_nodal_convex,
            parameters,
            node_intervals,
            dynamics_prop_aug,
            x_prop_aug,
            u_prop_aug,
        ) = preprocess_symbolic_problem(
            dynamics=dynamics,
            constraints=constraints,
            states=states,
            controls=controls,
            N=N,
            time=time,
            licq_min=licq_min,
            licq_max=licq_max,
            time_dilation_factor_min=time_dilation_factor_min,
            time_dilation_factor_max=time_dilation_factor_max,
            dynamics_prop_extra=dynamics_prop,
            states_prop_extra=states_prop,
        )

        # Step 2: Lower to JAX
        # TODO: Move CVXPy lowering here after SCP weights become CVXPy Parameters
        # Currently CVXPy lowering happens in initialize() because some weights
        # (lam_vc, lam_vb) are baked into OCP cost at creation time, and users
        # currently modify these between __init__ and initialize(). Once all weights
        # are CVXPy Parameters, CVXPy lowering can happen here alongside JAX lowering.
        (
            dynamics_augmented,
            lowered_constraints_nodal,
            constraints_nodal_convex,
            x_unified,
            u_unified,
            dynamics_augmented_prop,
            x_prop_unified,
        ) = lower_symbolic_expressions(
            dynamics_aug=dynamics_aug,
            states_aug=x_aug,
            controls_aug=u_aug,
            constraints_nodal=constraints_nodal,
            constraints_nodal_convex=constraints_nodal_convex,
            parameters=parameters,
            dynamics_prop=dynamics_prop_aug,
            states_prop=x_prop_aug,
            controls_prop=u_prop_aug,
        )

        # Step 3: Store Processed Components

        # Store state and control lists (includes user-defined + augmented)
        self.states = x_aug
        self.controls = u_aug

        # Store propagation states (includes extra propagation-only states)
        self.states_prop = x_prop_aug

        # Store unified objects for easy access
        self.x_unified = x_unified
        self.u_unified = u_unified

        # Store parameters in two forms:
        # 1. _parameters: plain dict for JAX functions
        # 2. _parameter_wrapper: wrapper dict for user access that auto-syncs
        self._parameters = parameters  # Plain dict for JAX
        self._parameter_wrapper = _ParameterDict(self, self._parameters, parameters)
        self.cvxpy_params = None  # Will be set during initialize()

        # Store dynamics objects
        self.dynamics_augmented = dynamics_augmented
        self.dynamics_augmented_prop = dynamics_augmented_prop

        # ==================== STEP 4: Setup SCP Configuration ====================

        # All indices are now stored in the unified objects themselves!
        # SimConfig will access them via properties
        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x=x_unified,
                x_prop=x_prop_unified,
                u=u_unified,
                total_time=x_unified.initial[x_unified.time_slice][0],
                n_states=x_unified.initial.shape[0],
                n_states_prop=x_prop_unified.initial.shape[0],
                ctcs_node_intervals=node_intervals,
            )

        if scp is None:
            scp = ScpConfig(
                n=N,
                w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
            )
        else:
            assert scp.n == N, "Number of segments must be the same as in the config"

        if dev is None:
            dev = DevConfig()
        if cvx is None:
            cvx = ConvexSolverConfig()
        if prp is None:
            prp = PropagationConfig()

        # Store constraints in SimConfig
        sim.constraints_ctcs = constraints_ctcs
        sim.constraints_nodal = lowered_constraints_nodal
        sim.constraints_nodal_convex = constraints_nodal_convex

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
            self.print_queue = queue.Queue()
            self.emitter_function = lambda data: self.print_queue.put(data)
            self.print_thread = threading.Thread(
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

    @property
    def parameters(self):
        """Get the parameters dictionary.

        The returned dictionary automatically syncs to CVXPy when modified:
            problem.parameters["obs_radius"] = 2.0  # Auto-syncs to CVXPy
            problem.parameters.update({"gate_0_center": center})  # Also syncs

        Returns:
            _ParameterDict: Special dict that syncs to CVXPy on assignment
        """
        return self._parameter_wrapper

    @parameters.setter
    def parameters(self, new_params: dict):
        """Replace the entire parameters dictionary and sync to CVXPy.

        Args:
            new_params: New parameters dictionary
        """
        self._parameters = dict(new_params)  # Create new plain dict
        self._parameter_wrapper = _ParameterDict(self, self._parameters, new_params)
        self._sync_parameters()

    def _sync_parameters(self):
        """Sync all parameter values to CVXPy parameters."""
        if self.cvxpy_params is not None:
            for name, value in self._parameter_wrapper.items():
                if name in self.cvxpy_params:
                    self.cvxpy_params[name].value = value

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
        self.dynamics_augmented.f = jax.vmap(self.dynamics_augmented.f, in_axes=(0, 0, 0, None))
        self.dynamics_augmented.A = jax.vmap(self.dynamics_augmented.A, in_axes=(0, 0, 0, None))
        self.dynamics_augmented.B = jax.vmap(self.dynamics_augmented.B, in_axes=(0, 0, 0, None))

        self.dynamics_augmented_prop.f = jax.vmap(
            self.dynamics_augmented_prop.f, in_axes=(0, 0, 0, None)
        )

        for constraint in self.settings.sim.constraints_nodal:
            # TODO: (haynec) switch to AOT instead of JIT
            constraint.func = jax.jit(constraint.func)
            constraint.grad_g_x = jax.jit(constraint.grad_g_x)
            constraint.grad_g_u = jax.jit(constraint.grad_g_u)

        # Generate solvers and optimal control problem
        self.discretization_solver = get_discretization_solver(
            self.dynamics_augmented, self.settings, self.parameters
        )
        self.propagation_solver = get_propagation_solver(
            self.dynamics_augmented_prop.f, self.settings, self.parameters
        )

        # TODO: Move this CVXPy lowering section to lower_symbolic_expressions()
        # This should happen in __init__ alongside JAX lowering for architectural consistency.
        # Blocked until all SCP weights become CVXPy Parameters (currently lam_vc and lam_vb
        # are baked into OCP at creation time). See docs/trajoptproblem_preprocessing_analysis.md
        # "CVXPy Lowering: Current Approach vs. Alternatives" section for details.

        # Phase 1: Create CVXPy variables
        ocp_vars = create_cvxpy_variables(self.settings)

        # Phase 2: Lower convex constraints to CVXPy
        lowered_convex_constraints, self.cvxpy_params = lower_convex_constraints(
            self.settings.sim.constraints_nodal_convex, ocp_vars, self._parameters
        )

        # Store lowered constraints back in settings for Phase 3
        self.settings.sim.constraints_nodal_convex = lowered_convex_constraints

        # Phase 3: Build complete optimal control problem
        self.optimal_control_problem = OptimalControlProblem(self.settings, ocp_vars)

        # Collect all relevant functions
        functions_to_hash = [self.dynamics_augmented.f, self.dynamics_augmented_prop.f]
        for constraint in self.settings.sim.constraints_nodal:
            functions_to_hash.append(constraint.func)
        # Note: CTCS constraints are already included in dynamics_augmented.f,
        # so we don't need to add them separately

        # Get cache file paths
        dis_solver_file, prop_solver_file = get_solver_cache_paths(
            functions_to_hash,
            n_discretization_nodes=self.settings.scp.n,
            dt=self.settings.prp.dt,
            total_time=self.settings.sim.total_time,
            state_max=self.settings.sim.x.max,
            state_min=self.settings.sim.x.min,
            control_max=self.settings.sim.u.max,
            control_min=self.settings.sim.u.min,
        )

        # Compile the discretization solver
        self.discretization_solver = load_or_compile_discretization_solver(
            self.discretization_solver,
            dis_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.scp.n,
            self.settings.sim.n_states,
            self.settings.sim.n_controls,
            save_compiled=self.settings.sim.save_compiled,
            debug=self.settings.dev.debug,
        )

        # Setup propagation solver parameters
        dtau = 1.0 / (self.settings.scp.n - 1)
        dt_max = self.settings.sim.u.max[self.settings.sim.time_dilation_slice][0] * dtau
        self.settings.prp.max_tau_len = int(dt_max / self.settings.prp.dt) + 2

        # Compile the propagation solver
        self.propagation_solver = load_or_compile_propagation_solver(
            self.propagation_solver,
            prop_solver_file,
            self._parameters,  # Plain dict for JAX
            self.settings.sim.n_states_prop,
            self.settings.sim.n_controls,
            self.settings.prp.max_tau_len,
            save_compiled=self.settings.sim.save_compiled,
        )

        # Initialize the PTR loop
        print("Initializing the SCvx Subproblem Solver...")
        self.cpg_solve = PTR_init(
            self._parameters,  # Plain dict for JAX/CVXPy
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

        # Prime the propagation solver
        prime_propagation_solver(self.propagation_solver, self._parameters, self.settings)

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
        result = PTR_step(
            self._parameters,  # Plain dict for JAX/CVXPy
            self.settings,
            self.optimal_control_problem,
            self.discretization_solver,
            self.cpg_solve,
            self.emitter_function,
            self.scp_k,
            self.scp_J_tr,
            self.scp_J_vb,
            self.scp_J_vc,
            self.scp_trajs,
            self.scp_controls,
            self.scp_V_multi_shoot_traj,
        )

        # Update instance state from result
        self.scp_k = result["scp_k"]
        self.scp_J_tr = result["scp_J_tr"]
        self.scp_J_vb = result["scp_J_vb"]
        self.scp_J_vc = result["scp_J_vc"]

        return result

    def solve(
        self, max_iters: Optional[int] = None, continuous: bool = False
    ) -> OptimizationResults:
        # Sync parameters before solving
        self._sync_parameters()

        # Ensure parameter sizes and normalization are correct
        self.settings.scp.__post_init__()
        self.settings.sim.__post_init__()

        if self.optimal_control_problem is None or self.discretization_solver is None:
            raise ValueError("Problem has not been initialized. Call initialize() before solve()")

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
        result = propagate_trajectory_results(
            self._parameters, self.settings, result, self.propagation_solver
        )
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
