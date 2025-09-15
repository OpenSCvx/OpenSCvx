import os
import queue
import threading
import time
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Union

import jax
import numpy as np
from jax import jacfwd

os.environ["EQX_ON_ERROR"] = "nan"

from openscvx import io
from openscvx.backend.augmentation import (
    augment_dynamics_with_ctcs,
    decompose_vector_nodal_constraints,
    separate_constraints,
    sort_ctcs_constraints,
)
from openscvx.backend.canonicalizer import canonicalize
from openscvx.backend.control import Control
from openscvx.backend.expr import CTCS, Constraint, Expr
from openscvx.backend.lower import lower_to_jax
from openscvx.backend.preprocessing import (
    collect_and_assign_slices,
    validate_and_normalize_constraint_nodes,
    validate_constraints_at_root,
    validate_dynamics_dimension,
    validate_shapes,
    validate_variable_names,
)
from openscvx.backend.state import State
from openscvx.backend.unified import UnifiedControl, UnifiedState, unify_controls, unify_states
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
from openscvx.constraints.lowered import LoweredNodalConstraint
from openscvx.discretization import get_discretization_solver
from openscvx.dynamics import Dynamics
from openscvx.dynamics import dynamics as to_dynamics
from openscvx.ocp import OptimalControlProblem
from openscvx.post_processing import propagate_trajectory_results
from openscvx.propagation import get_propagation_solver
from openscvx.ptr import PTR_init, PTR_subproblem, format_result
from openscvx.results import OptimizationResults

if TYPE_CHECKING:
    import cvxpy as cp


# TODO: (norrisg) Decide whether to have constraints`, `cost`, alongside `dynamics`, ` etc.
class TrajOptProblem:
    def __init__(
        self,
        dynamics: Expr,
        constraints: List[Union[Constraint, CTCS]],
        x: State,
        u: Control,
        N: int,
        idx_time: int,
        params: Optional[dict] = None,
        dynamics_prop: Optional[callable] = None,
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
            constraints (List[Union[CTCSConstraint, NodalConstraint]]):
                List of constraints decorated with @ctcs or @nodal
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

        # Validate expressions
        all_exprs = [dynamics] + constraints
        validate_variable_names(all_exprs)
        collect_and_assign_slices(all_exprs)
        validate_shapes(all_exprs)
        validate_constraints_at_root(constraints)
        validate_and_normalize_constraint_nodes(constraints, N)
        validate_dynamics_dimension(dynamics, x)

        # Canonicalize all expressions after validation
        dynamics = canonicalize(dynamics)
        constraints = [canonicalize(expr) for expr in constraints]

        # Sort and separate constraints first
        constraints_ctcs, constraints_nodal = separate_constraints(constraints, N)

        # Decompose vector-valued nodal constraints into scalar constraints
        # This is necessary for nonconvex nodal constraints that get lowered to JAX
        constraints_nodal = decompose_vector_nodal_constraints(constraints_nodal)

        # Sort CTCS constraints by their idx to get node_intervals
        constraints_ctcs, node_intervals, num_augmented_states = sort_ctcs_constraints(
            constraints_ctcs
        )

        # Augment dynamics, states, and controls with CTCS constraints, time dilation
        dynamics_aug, x_aug, u_aug = augment_dynamics_with_ctcs(
            dynamics,
            [x],
            [u],
            constraints_ctcs,
            N,
            idx_time,
            licq_min=licq_min,
            licq_max=licq_max,
            time_dilation_factor_min=time_dilation_factor_min,
            time_dilation_factor_max=time_dilation_factor_max,
        )

        # TODO: (norrisg) this is somewhat of a hack; using x_aug, u_aug as leaf-node expressions to
        # assign slices, should probably move into the augmentation functions themselves
        collect_and_assign_slices(all_exprs + x_aug + u_aug)

        # TODO: (norrisg) allow non-ctcs constraints
        dyn_fn = lower_to_jax(dynamics_aug)
        constraints_nodal_fns = lower_to_jax(constraints_nodal)

        dynamics_fn = to_dynamics(dyn_fn)

        x_unified: UnifiedState = unify_states(x_aug)
        u_unified: UnifiedControl = unify_controls(u_aug)

        if params is None:
            params = {}
        self.params = params

        if dynamics_prop is None:
            dynamics_prop = dynamics_fn

        if x_prop is None:
            x_prop = deepcopy(x_unified)

        # Index tracking
        # TODO: (norrisg) use the `_slice` attribute of the State, Control
        idx_x_true = slice(0, x_unified.true.shape[0])
        idx_x_true_prop = slice(0, x_prop.shape[0])
        idx_u_true = slice(0, u_unified.true.shape[0])
        idx_constraint_violation = slice(idx_x_true.stop, idx_x_true.stop + num_augmented_states)
        idx_constraint_violation_prop = slice(
            idx_x_true_prop.stop, idx_x_true_prop.stop + num_augmented_states
        )

        # Time dilation index for reference
        idx_time_dilation = slice(idx_u_true.stop, idx_u_true.stop + 1)

        # check that idx_time is in the correct range
        assert idx_time >= 0 and idx_time < len(x_unified.max), (
            "idx_time must be in the range of the state vector and non-negative"
        )
        idx_time = slice(idx_time, idx_time + 1)

        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x=x_unified,
                x_prop=x_prop,
                u=u_unified,
                total_time=x_unified.initial[idx_time][0],
                n_states=x_unified.initial.shape[0],
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
            assert self.settings.scp.n == N, "Number of segments must be the same as in the config"

        if dev is None:
            dev = DevConfig()
        if cvx is None:
            cvx = ConvexSolverConfig()
        if prp is None:
            prp = PropagationConfig()

        # Create LoweredConstraint objects with Jacobians computed automatically
        lowered_constraints_nodal = []
        for i, fn in enumerate(constraints_nodal_fns):
            # Apply vectorization to handle (N, n_x) and (N, n_u) inputs
            # The lowered functions have signature (x, u, node, **kwargs), so we need to handle node
            # parameter, node is broadcast (same for all),
            constraint = LoweredNodalConstraint(
                func=jax.vmap(fn, in_axes=(0, 0, None)),
                grad_g_x=jax.vmap(jacfwd(fn, argnums=0), in_axes=(0, 0, None)),
                grad_g_u=jax.vmap(jacfwd(fn, argnums=1), in_axes=(0, 0, None)),
                nodes=constraints_nodal[i].nodes,
            )
            lowered_constraints_nodal.append(constraint)

        sim.constraints_ctcs = []
        sim.constraints_nodal = lowered_constraints_nodal

        # Create dynamics objects from the symbolic augmented dynamics
        self.dynamics_augmented = Dynamics(
            f=dyn_fn,
            A=jacfwd(dyn_fn, argnums=0),
            B=jacfwd(dyn_fn, argnums=1),
        )
        # For propagation, use the same augmented dynamics function
        # (since CTCS augmentation applies to both discretization and propagation)
        self.dynamics_augmented_prop = Dynamics(
            f=dyn_fn,
            A=jacfwd(dyn_fn, argnums=0),
            B=jacfwd(dyn_fn, argnums=1),
        )

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
        self.dynamics_augmented.f = jax.vmap(
            self.dynamics_augmented.f, in_axes=(0, 0, 0, *(None,) * len(self.params))
        )
        self.dynamics_augmented.A = jax.vmap(
            self.dynamics_augmented.A, in_axes=(0, 0, 0, *(None,) * len(self.params))
        )
        self.dynamics_augmented.B = jax.vmap(
            self.dynamics_augmented.B, in_axes=(0, 0, 0, *(None,) * len(self.params))
        )

        self.dynamics_augmented_prop.f = jax.vmap(
            self.dynamics_augmented_prop.f, in_axes=(0, 0, 0, *(None,) * len(self.params))
        )

        for constraint in self.settings.sim.constraints_nodal:
            # TODO: (haynec) switch to AOT instead of JIT
            constraint.func = jax.jit(constraint.func)
            constraint.grad_g_x = jax.jit(constraint.grad_g_x)
            constraint.grad_g_u = jax.jit(constraint.grad_g_u)

        # Generate solvers and optimal control problem
        self.discretization_solver = get_discretization_solver(
            self.dynamics_augmented, self.settings, self.params
        )
        self.propagation_solver = get_propagation_solver(
            self.dynamics_augmented_prop.f, self.settings, self.params
        )
        self.optimal_control_problem = OptimalControlProblem(self.settings)

        # Collect all relevant functions
        functions_to_hash = [self.dynamics_augmented.f, self.dynamics_augmented_prop.f]
        for constraint in self.settings.sim.constraints_nodal:
            functions_to_hash.append(constraint.func)
        for constraint in self.settings.sim.constraints_ctcs:
            functions_to_hash.append(constraint.func)

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
            self.params,
            self.settings.scp.n,
            self.settings.sim.n_states,
            self.settings.sim.n_controls,
            save_compiled=self.settings.sim.save_compiled,
            debug=self.settings.dev.debug,
        )

        # Setup propagation solver parameters
        dtau = 1.0 / (self.settings.scp.n - 1)
        dt_max = self.settings.sim.u.max[self.settings.sim.idx_s][0] * dtau
        self.settings.prp.max_tau_len = int(dt_max / self.settings.prp.dt) + 2

        # Compile the propagation solver
        self.propagation_solver = load_or_compile_propagation_solver(
            self.propagation_solver,
            prop_solver_file,
            self.params,
            self.settings.sim.n_states_prop,
            self.settings.sim.n_controls,
            self.settings.prp.max_tau_len,
            save_compiled=self.settings.sim.save_compiled,
        )

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

        # Prime the propagation solver
        prime_propagation_solver(self.propagation_solver, self.params, self.settings)

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
        (
            x_sol,
            u_sol,
            cost,
            J_total,
            J_vb_vec,
            J_vc_vec,
            J_tr_vec,
            prob_stat,
            V_multi_shoot,
            subprop_time,
            dis_time,
        ) = PTR_subproblem(
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
        self.settings.scp.w_tr = min(
            self.settings.scp.w_tr * self.settings.scp.w_tr_adapt, self.settings.scp.w_tr_max
        )
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
            "converged": (
                (self.scp_J_tr < self.settings.scp.ep_tr)
                and (self.scp_J_vb < self.settings.scp.ep_vb)
                and (self.scp_J_vc < self.settings.scp.ep_vc)
            ),
            "u": u,
            "x": x,
            "V_multi_shoot": V_multi_shoot,
        }

    def solve(
        self, max_iters: Optional[int] = None, continuous: bool = False
    ) -> OptimizationResults:
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
            self.params, self.settings, result, self.propagation_solver
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
