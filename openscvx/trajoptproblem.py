import jax.numpy as jnp
from typing import List

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
from openscvx.dynamics import get_augmented_dynamics, get_jacobians
from openscvx.constraints.ctcs import get_g_func
from openscvx.discretization import ExactDis, Diffrax_Prop
from openscvx.constraints.boundary import BoundaryConstraint
from openscvx.ptr import PTR_init, PTR_main, PTR_post

# TODO: (norrisg) Decide whether to have constraints`, `cost`, alongside `dynamics`, ` etc.
class TrajOptProblem:
    def __init__(
        self,
        dynamics: callable,
        constraints: List[callable],
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
        scp: ScpConfig = None,
        dis: DiscretizationConfig = None,
        prp: PropagationConfig = None,
        sim: SimConfig = None,
        dev: DevConfig = None,
        cvx: ConvexSolverConfig = None,
        ctcs_augmentation_min=0.0,
        ctcs_augmentation_max=1e-4,
        time_dilation_factor_min=0.3,
        time_dilation_factor_max=3.0,
    ):
        """Initializes the TrajOptProblem class.

        This constructor sets up a trajectory optimization problem by defining the system dynamics, 
        constraints, and various configurations for solving the problem.

        Args:
            dynamics (callable): The system dynamics function, which defines the evolution of the state.
            constraints (List[callable]): A list of constraint functions to be satisfied during optimization.
            N (int): The number of discretization points for the trajectory.
            time_init (float): The initial time for the trajectory.
            x_guess (jnp.ndarray): Initial guess for the state trajectory.
            u_guess (jnp.ndarray): Initial guess for the control trajectory.
            initial_state (BoundaryConstraint): Boundary constraint for the initial state of the system.
            final_state (BoundaryConstraint): Boundary constraint for the final state of the system.
            x_max (jnp.ndarray): Maximum bounds for the state variables.
            x_min (jnp.ndarray): Minimum bounds for the state variables.
            u_max (jnp.ndarray): Maximum bounds for the control variables.
            u_min (jnp.ndarray): Minimum bounds for the control variables.
            scp (ScpConfig, optional): Configuration for the sequential convex programming solver.
            dis (DiscretizationConfig, optional): Configuration for the discretization method.
            prp (PropagationConfig, optional): Configuration for the propagation method.
            sim (SimConfig, optional): Configuration for the simulation settings.
            dev (DevConfig, optional): Configuration for device-specific settings.
            cvx (ConvexSolverConfig, optional): Configuration for the convex solver.

        Raises:
            ValueError: If any of the input parameters are invalid or inconsistent.
        !!! note

            It is possible to have `t1 < t0`, in which case integration proceeds backwards
            in time.
        """

        
        x_min_augmented = np.hstack([x_min, ctcs_augmentation_min])
        x_max_augmented = np.hstack([x_max, ctcs_augmentation_max])

        u_min_augmented = np.hstack([u_min, time_dilation_factor_min * time_init])
        u_max_augmented = np.hstack([u_max, time_dilation_factor_max * time_init])

        x_bar_augmented = np.hstack([x_guess, np.full((x_guess.shape[0], 1), 0)])
        u_bar_augmented = np.hstack(
            [u_guess, np.full((u_guess.shape[0], 1), time_init)]
        )

        if dis is None:
            dis = DiscretizationConfig()

        if sim is None:
            sim = SimConfig(
                x_bar=x_bar_augmented,
                u_bar=u_bar_augmented,
                initial_state=initial_state,
                final_state=final_state,
                max_state=x_max_augmented,
                min_state=x_min_augmented,
                max_control=u_max_augmented,
                min_control=u_min_augmented,
                total_time=time_init,
                n_states=len(x_max),
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

        for constraint in constraints:
            if constraint.constraint_type == "ctcs":
                sim.constraints_ctcs.append(
                    lambda x, u, func=constraint: jnp.sum(func.penalty(func(x, u)))
                )
            elif constraint.constraint_type == "nodal":
                sim.constraints_nodal.append(constraint)
            else:
                raise ValueError(
                    f"Unknown constraint type: {constraint.constraint_type}, All constraints must be decorated with @ctcs or @nodal"
                )

        g_func = get_g_func(sim.constraints_ctcs)
        self.dynamics_augmented = get_augmented_dynamics(dynamics, g_func)
        self.A_uncompiled, self.B_uncompiled = get_jacobians(self.dynamics_augmented)

        self.params = Config(
            sim=sim,
            scp=scp,
            dis=dis,
            dev=dev,
            cvx=cvx,
            prp=prp,
        )

        self.ocp: cp.Problem = None
        self.dynamics_discretized: ExactDis = None
        self.cpg_solve = None

    def compile(self):
        # TODO: (norrisg) Could consider using dataclass just to hold dynamics and jacobians
        # TODO: (norrisg) Consider writing the compiled versions into the same variables?
        # Otherwise if have a dataclass could have 2 instances, one for compied and one for uncompiled
        self.state_dot = jax.vmap(self.dynamics_augmented)
        self.A = jax.jit(jax.vmap(self.A_uncompiled, in_axes=(0, 0)))
        self.B = jax.jit(jax.vmap(self.B_uncompiled, in_axes=(0, 0)))

    def initialize(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        self.compile()

        self.ocp, self.dynamics_discretized, self.cpg_solve = PTR_init(
            self.state_dot, self.A, self.B, self.params
        )

        # Extract the number of states and controls from the parameters
        n_x = self.params.sim.n_states
        n_u = self.params.sim.n_controls

        # Define indices for slicing the augmented state vector
        self.i0 = 0
        self.i1 = n_x
        self.i2 = self.i1 + n_x * n_x
        self.i3 = self.i2 + n_x * n_u
        self.i4 = self.i3 + n_x * n_u
        self.i5 = self.i4 + n_x

        if not self.params.dev.debug:
            self.dynamics_discretized.calculate_discretization = jax.jit(
                self.dynamics_discretized.calculate_discretization
            ).lower(
                np.ones((self.params.scp.n, self.params.sim.n_states)),
                np.ones((self.params.scp.n, self.params.sim.n_controls)),
            ).compile()

        diff_prop = Diffrax_Prop(self.state_dot, self.A, self.B, self.params)
        self.params.prp.integrator = jax.jit(diff_prop.solve_ivp).lower(
            np.ones((self.params.sim.n_states)),
            (0.0, 0.0),
            np.ones((1, self.params.sim.n_controls)), 
            np.ones((1, self.params.sim.n_controls)), 
            np.ones((1,1)), 
            0
        ).compile()


        # _ = self.dynamics_discretized.simulate_nonlinear_time(np.ones((self.params.sim.n_states)), np.zeros((10, self.params.sim.n_controls)), np.linspace(0,1, 100), np.linspace(0,1, 10))
        
    def solve(self):
        # Ensure parameter sizes and normalization are correct
        self.params.scp.__post_init__()
        self.params.sim.__post_init__()

        if self.ocp is None or self.dynamics_discretized is None:
            raise ValueError(
                "Problem has not been initialized. Call initialize() before solve()"
            )

        return PTR_main(
            self.params, self.ocp, self.dynamics_discretized, self.cpg_solve
        )

    def post_process(self, result):
        return PTR_post(self.params, result, self.dynamics_discretized)
