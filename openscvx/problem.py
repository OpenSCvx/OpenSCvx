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
from openscvx.lowered import LoweredProblem, ParameterDict
from openscvx.lowered.dynamics import Dynamics
from openscvx.ocp import OptimalControlProblem
from openscvx.post_processing import propagate_trajectory_results
from openscvx.propagation import get_propagation_solver
from openscvx.ptr import PTR_init, PTR_step, format_result
from openscvx.results import OptimizationResults
from openscvx.solver_state import SolverState
from openscvx.symbolic.builder import preprocess_symbolic_problem
from openscvx.symbolic.constraint_set import ConstraintSet
from openscvx.symbolic.expr import CTCS, Constraint
from openscvx.symbolic.expr.control import Control
from openscvx.symbolic.expr.state import State
from openscvx.symbolic.lower import lower_symbolic_problem
from openscvx.symbolic.problem import SymbolicProblem
from openscvx.time import Time

if TYPE_CHECKING:
    import cvxpy as cp


class Problem:
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

        # Symbolic Preprocessing & Augmentation
        self.symbolic: SymbolicProblem = preprocess_symbolic_problem(
            dynamics=dynamics,
            constraints=ConstraintSet(unsorted=list(constraints)),
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

        # Lower to JAX and CVXPy
        self._lowered: LoweredProblem = lower_symbolic_problem(self.symbolic)

        # Store parameters in two forms:
        self._parameters = self.symbolic.parameters  # Plain dict for JAX functions
        # Wrapper dict for user access that auto-syncs
        self._parameter_wrapper = ParameterDict(self, self._parameters, self.symbolic.parameters)

        # Setup SCP Configuration
        self.settings = Config(
            sim=SimConfig(
                x=self._lowered.x_unified,
                x_prop=self._lowered.x_prop_unified,
                u=self._lowered.u_unified,
                total_time=self._lowered.x_unified.initial[self._lowered.x_unified.time_slice][0],
                n_states=self._lowered.x_unified.initial.shape[0],
                n_states_prop=self._lowered.x_prop_unified.initial.shape[0],
                ctcs_node_intervals=self.symbolic.node_intervals,
            ),
            scp=ScpConfig(
                n=N,
                w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
            ),
            dis=DiscretizationConfig(),
            dev=DevConfig(),
            cvx=ConvexSolverConfig(),
            prp=PropagationConfig(),
        )

        # OCP construction happens in initialize() so users can modify
        # settings (like uniform_time_grid) between __init__ and initialize()
        self.optimal_control_problem: cp.Problem = None
        self.discretization_solver: callable = None
        self.cpg_solve = None

        # Set up emitter & thread only if printing is enabled
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

        # Compiled dynamics (vmapped versions, set in initialize())
        self._compiled: Optional[Dynamics] = None
        self._compiled_prop: Optional[Dynamics] = None

        # Solver state (created fresh for each solve)
        self._state: Optional[SolverState] = None

    @property
    def parameters(self):
        """Get the parameters dictionary.

        The returned dictionary automatically syncs to CVXPy when modified:
            problem.parameters["obs_radius"] = 2.0  # Auto-syncs to CVXPy
            problem.parameters.update({"gate_0_center": center})  # Also syncs

        Returns:
            ParameterDict: Special dict that syncs to CVXPy on assignment
        """
        return self._parameter_wrapper

    @parameters.setter
    def parameters(self, new_params: dict):
        """Replace the entire parameters dictionary and sync to CVXPy.

        Args:
            new_params: New parameters dictionary
        """
        self._parameters = dict(new_params)  # Create new plain dict
        self._parameter_wrapper = ParameterDict(self, self._parameters, new_params)
        self._sync_parameters()

    def _sync_parameters(self):
        """Sync all parameter values to CVXPy parameters."""
        if self._lowered.cvxpy_params is not None:
            for name, value in self._parameter_wrapper.items():
                if name in self._lowered.cvxpy_params:
                    self._lowered.cvxpy_params[name].value = value

    @property
    def state(self) -> Optional[SolverState]:
        """Access the current solver state.

        The solver state contains all mutable state from the SCP iterations,
        including current guesses, costs, weights, and history.

        Returns:
            SolverState if initialized, None otherwise

        Example:
            >>> problem.initialize()
            >>> problem.step()
            >>> print(f"Iteration {problem.state.k}, J_tr={problem.state.J_tr}")
        """
        return self._state

    @property
    def lowered(self) -> LoweredProblem:
        """Access the lowered problem containing JAX/CVXPy objects.

        Returns:
            LoweredProblem with dynamics, constraints, unified interfaces, and CVXPy vars
        """
        return self._lowered

    @property
    def x_unified(self):
        """Unified state interface (delegates to lowered.x_unified)."""
        return self._lowered.x_unified

    @property
    def u_unified(self):
        """Unified control interface (delegates to lowered.u_unified)."""
        return self._lowered.u_unified

    def initialize(self):
        io.intro()

        # Print problem summary
        io.print_problem_summary(self.settings, self._lowered)

        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Ensure parameter sizes and normalization are correct
        self.settings.scp.__post_init__()
        self.settings.sim.__post_init__()

        # Create compiled (vmapped) dynamics as new instances
        # This preserves the original un-vmapped versions in _lowered
        self._compiled = Dynamics(
            f=jax.vmap(self._lowered.dynamics.f, in_axes=(0, 0, 0, None)),
            A=jax.vmap(self._lowered.dynamics.A, in_axes=(0, 0, 0, None)),
            B=jax.vmap(self._lowered.dynamics.B, in_axes=(0, 0, 0, None)),
        )

        self._compiled_prop = Dynamics(
            f=jax.vmap(self._lowered.dynamics_prop.f, in_axes=(0, 0, 0, None)),
        )

        for constraint in self._lowered.jax_constraints.nodal:
            # TODO: (haynec) switch to AOT instead of JIT
            constraint.func = jax.jit(constraint.func)
            constraint.grad_g_x = jax.jit(constraint.grad_g_x)
            constraint.grad_g_u = jax.jit(constraint.grad_g_u)

        # JIT compile cross-node constraints
        for constraint in self._lowered.jax_constraints.cross_node:
            constraint.func = jax.jit(constraint.func)
            constraint.grad_g_X = jax.jit(constraint.grad_g_X)
            constraint.grad_g_U = jax.jit(constraint.grad_g_U)

        # Generate solvers using compiled (vmapped) dynamics
        self.discretization_solver = get_discretization_solver(
            self._compiled, self.settings, self.parameters
        )
        self.propagation_solver = get_propagation_solver(
            self._compiled_prop.f, self.settings, self.parameters
        )

        # Build optimal control problem using LoweredProblem
        self.optimal_control_problem = OptimalControlProblem(self.settings, self._lowered)

        # Get cache file paths using symbolic AST hashing
        # This is more stable than hashing lowered JAX code
        dis_solver_file, prop_solver_file = get_solver_cache_paths(
            self.symbolic,
            dt=self.settings.prp.dt,
            total_time=self.settings.sim.total_time,
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
            self._lowered.jax_constraints,
        )
        print("✓ SCvx Subproblem Solver initialized")

        # Create fresh solver state
        self._state = SolverState.from_settings(self.settings)

        t_f_while = time.time()
        self.timing_init = t_f_while - t_0_while
        print("Total Initialization Time: ", self.timing_init)

        # Prime the propagation solver
        prime_propagation_solver(self.propagation_solver, self._parameters, self.settings)

        if self.settings.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_initialize.prof")

    def reset(self):
        """Reset solver state to run the same problem again.

        This method creates a fresh SolverState from the current settings,
        allowing you to re-run the optimization with the same initial conditions.
        The compiled dynamics and optimal control problem are preserved.

        Raises:
            ValueError: If initialize() has not been called yet.

        Example:
            >>> problem.initialize()
            >>> result1 = problem.solve()
            >>> problem.reset()  # Reset to initial state
            >>> result2 = problem.solve()  # Run again from scratch
        """
        if self._compiled is None:
            raise ValueError("Problem has not been initialized. Call initialize() first")

        # Create fresh solver state from settings
        self._state = SolverState.from_settings(self.settings)

        # Reset timing
        self.timing_solve = None
        self.timing_post = None

    def step(self) -> dict:
        """Performs a single SCP iteration.

        This method is designed for real-time plotting and interactive optimization.
        It performs one complete SCP iteration including subproblem solving,
        state updates, and progress emission for real-time visualization.

        Returns:
            dict: Dictionary containing convergence status and current state
        """
        if self._state is None:
            raise ValueError("Problem has not been initialized. Call initialize() first")

        converged = PTR_step(
            self._parameters,  # Plain dict for JAX/CVXPy
            self.settings,
            self._state,
            self.optimal_control_problem,
            self.discretization_solver,
            self.cpg_solve,
            self.emitter_function,
            self._lowered.jax_constraints,
        )

        # Return dict matching original API
        return {
            "converged": converged,
            "scp_k": self._state.k,
            "scp_J_tr": self._state.J_tr,
            "scp_J_vb": self._state.J_vb,
            "scp_J_vc": self._state.J_vc,
        }

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

        if self._state is None:
            raise ValueError("Solver state not initialized. Call initialize() before solve()")

        # Sync state weights with (re-)normalized settings
        self._state.w_tr = self.settings.scp.w_tr
        self._state.lam_cost = self.settings.scp.lam_cost
        self._state.lam_vc = self.settings.scp.lam_vc
        self._state.lam_vb = self.settings.scp.lam_vb

        # Enable the profiler
        if self.settings.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Print top header for solver results
        io.header()

        k_max = max_iters if max_iters is not None else self.settings.scp.k_max

        while self._state.k <= k_max:
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

        # Sync final solution back to settings for post_processing compatibility
        # TODO: (norrisg) This is hacky and not idempotent!
        # Should instead update post processing to handle SolverState directly
        # Could then save a `self._solution: SolverState` attribute to hold the final state and pass]
        # that into the post processing pipeline 
        self.settings.sim.x.guess = self._state.x_guess
        self.settings.sim.u.guess = self._state.u_guess

        return format_result(self, self._state.k <= k_max)

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
