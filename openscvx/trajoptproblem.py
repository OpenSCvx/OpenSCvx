import jax.numpy as jnp
from typing import List, Union, Optional
import queue
import threading
import time

import cvxpy as cp
import jax
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
from openscvx.constraints.boundary import BoundaryConstraint, boundary
from openscvx.constraints.ctcs import CTCSConstraint
from openscvx.constraints.nodal import NodalConstraint
from openscvx.ptr import PTR_init, PTR_main
from openscvx.post_processing import propagate_trajectory_results
from openscvx.ocp import OptimalControlProblem
from openscvx import io


# TODO: (norrisg) Decide whether to have constraints`, `cost`, alongside `dynamics`, ` etc.
class TrajOptProblem:
    def __init__(
        self,
        dynamics: Dynamics,
        constraints: List[Union[CTCSConstraint, NodalConstraint]],
        idx_time: int,
        N: int,
        time_init: float,
        x_guess: jnp.ndarray,
        u_guess: jnp.ndarray,
        initial_state: BoundaryConstraint,
        final_state: BoundaryConstraint,
        x_max: jnp.ndarray,
        x_min: jnp.ndarray,
        u_max: jnp.ndarray,
        u_min: jnp.ndarray,
        dynamics_prop: callable = None,
        initial_state_prop: BoundaryConstraint = None,
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
        if dynamics_prop is None:
            dynamics_prop = dynamics
        
        if initial_state_prop is None:
            initial_state_prop = initial_state

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
        idx_x_true = slice(0, len(initial_state.value))
        idx_x_true_prop = slice(0, len(initial_state_prop.value))
        idx_u_true = slice(0, len(u_max))
        idx_constraint_violation = slice(
            idx_x_true.stop, idx_x_true.stop + num_augmented_states
        )
        idx_constraint_violation_prop = slice(
            idx_x_true_prop.stop, idx_x_true_prop.stop + num_augmented_states
        )

        idx_time_dilation = slice(idx_u_true.stop, idx_u_true.stop + 1)

        # check that idx_time is in the correct range
        assert idx_time >= 0 and idx_time < len(
            x_max
        ), "idx_time must be in the range of the state vector and non-negative"
        idx_time = slice(idx_time, idx_time + 1)

        x_min_augmented = np.hstack([x_min, np.repeat(licq_min, num_augmented_states)])
        x_max_augmented = np.hstack([x_max, np.repeat(licq_max, num_augmented_states)])

        u_min_augmented = np.hstack([u_min, time_dilation_factor_min * time_init])
        u_max_augmented = np.hstack([u_max, time_dilation_factor_max * time_init])

        x_bar_augmented = np.hstack([x_guess, np.full((x_guess.shape[0], num_augmented_states), 0)])
        u_bar_augmented = np.hstack(
            [u_guess, np.full((u_guess.shape[0], 1), time_init)]
        )

        initial_state_prop_values = np.hstack([initial_state_prop.value, np.repeat(licq_min, num_augmented_states)])
        initial_state_prop_types = np.hstack([initial_state_prop.type, ["Fix"] * num_augmented_states])
        initial_state_prop = boundary(initial_state_prop_values)
        initial_state_prop.types = initial_state_prop_types

        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x_bar=x_bar_augmented,
                u_bar=u_bar_augmented,
                initial_state=initial_state,
                initial_state_prop=initial_state_prop,
                final_state=final_state,
                max_state=x_max_augmented,
                min_state=x_min_augmented,
                max_control=u_max_augmented,
                min_control=u_min_augmented,
                total_time=time_init,
                n_states=len(initial_state.value),
                n_states_prop=len(initial_state_prop.value),
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
                self.scp.n == N
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

        self.params = Config(
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
        if self.params.dev.printing:
            self.print_queue      = queue.Queue()
            self.emitter_function = lambda data: self.print_queue.put(data)
            self.print_thread     = threading.Thread(
                target=io.intermediate,
                args=(self.print_queue, self.params),
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
        if self.params.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        # Compile dynamics and jacobians
        self.dynamics_augmented.f = jax.vmap(self.dynamics_augmented.f)
        self.dynamics_augmented.A = jax.jit(jax.vmap(self.dynamics_augmented.A, in_axes=(0, 0, 0)))
        self.dynamics_augmented.B = jax.jit(jax.vmap(self.dynamics_augmented.B, in_axes=(0, 0, 0)))


        self.dynamics_augmented_prop.f = jax.vmap(self.dynamics_augmented_prop.f)

        for constraint in self.params.sim.constraints_nodal:
            if not constraint.convex:
                # TODO: (haynec) switch to AOT instead of JIT
                constraint.g = jax.jit(constraint.g)
                constraint.grad_g_x = jax.jit(constraint.grad_g_x)
                constraint.grad_g_u = jax.jit(constraint.grad_g_u)

        # Generate solvers and optimal control problem
        self.discretization_solver = get_discretization_solver(self.dynamics_augmented, self.params)
        self.propagation_solver = get_propagation_solver(self.dynamics_augmented_prop.f, self.params)
        self.optimal_control_problem = OptimalControlProblem(self.params)

        # Initialize the PTR loop
        self.cpg_solve = PTR_init(
            self.optimal_control_problem,
            self.discretization_solver,
            self.params,
        )

        # Compile the solvers
        if not self.params.dev.debug:
            self.discretization_solver = (
                jax.jit(self.discretization_solver)
                .lower(
                    np.ones((self.params.scp.n, self.params.sim.n_states)),
                    np.ones((self.params.scp.n, self.params.sim.n_controls)),
                )
                .compile()
            )

        self.propagation_solver = (
            jax.jit(self.propagation_solver)
            .lower(
                np.ones((self.params.sim.n_states_prop)),
                (0.0, 0.0),
                np.ones((1, self.params.sim.n_controls)),
                np.ones((1, self.params.sim.n_controls)),
                np.ones((1, 1)),
                np.ones((1, 1)).astype("int"),
                0,
            )
            .compile()
        )

        t_f_while = time.time()
        self.timing_init = t_f_while - t_0_while
        print("Total Initialization Time: ", self.timing_init)

        if self.params.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_initialize.prof")

    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        if self.optimal_control_problem is None or self.discretization_solver is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve()"
            )

        # Enable the profiler
        if self.params.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_while = time.time()
        # Print top header for solver results
        io.header()

        result = PTR_main(
            self.params,
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
        if self.params.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_solve.prof")

        return result

    def post_process(self, result):
        # Enable the profiler
        if self.params.dev.profiling:
            import cProfile

            pr = cProfile.Profile()
            pr.enable()

        t_0_post = time.time()
        result = propagate_trajectory_results(self.params, result, self.propagation_solver)
        t_f_post = time.time()

        self.timing_post = t_f_post - t_0_post
        print("Total Post Processing Time: ", self.timing_post)

        # Disable the profiler
        if self.params.dev.profiling:
            pr.disable()
            # Save results so it can be viusualized with snakeviz
            pr.dump_stats("profiling_postprocess.prof")
        return result
