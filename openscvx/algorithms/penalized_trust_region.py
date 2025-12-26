"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
import warnings
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import numpy.linalg as la

from openscvx.config import Config

from .autotuning import update_scp_weights
from .base import Algorithm, AlgorithmState

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints

warnings.filterwarnings("ignore")


class PenalizedTrustRegion(Algorithm):
    """Penalized Trust Region (PTR) successive convexification algorithm.

    PTR is the default SCvx algorithm that uses trust region methods with
    penalty-based constraint handling. It includes adaptive parameter tuning
    and virtual control relaxation for handling infeasible subproblems.

    Example:
        Using PTR with a Problem::

            from openscvx.algorithms import PenalizedTrustRegion

            problem = Problem(dynamics, constraints, states, controls, N, time)
            problem.initialize()
            result = problem.solve()
    """

    def initialize(
        self,
        params: dict,
        ocp: cp.Problem,
        discretization_solver: callable,
        settings: Config,
        jax_constraints: "LoweredJaxConstraints",
        solve_ocp: callable,
    ) -> None:
        """Initialize PTR algorithm.

        Stores solver callable and performs a warm-start solve to
        initialize DPP and JAX jacobians.

        Args:
            params: Problem parameters dictionary
            ocp: CVXPy optimal control problem
            discretization_solver: Compiled discretization solver
            settings: Configuration object
            jax_constraints: JIT-compiled constraint functions
            solve_ocp: Callable that solves the OCP
        """
        self._solve_ocp = solve_ocp

        if "x_init" in ocp.param_dict:
            ocp.param_dict["x_init"].value = settings.sim.x.initial

        if "x_term" in ocp.param_dict:
            ocp.param_dict["x_term"].value = settings.sim.x.final

        # Create temporary state for initialization solve
        init_state = AlgorithmState.from_settings(settings)

        # Solve a dumb problem to initialize DPP and JAX jacobians
        _ = self._subproblem(
            params.items(),
            init_state,
            discretization_solver,
            ocp,
            settings,
            jax_constraints,
        )

    def step(
        self,
        params: dict,
        settings: Config,
        state: AlgorithmState,
        ocp: cp.Problem,
        discretization_solver: callable,
        emitter_function: callable,
        jax_constraints: "LoweredJaxConstraints",
    ) -> bool:
        """Execute one PTR iteration.

        Solves the convex subproblem, updates state in place, and checks
        convergence based on trust region, virtual buffer, and virtual
        control costs.

        Args:
            params: Problem parameters dictionary
            settings: Configuration object
            state: Mutable solver state (modified in place)
            ocp: CVXPy optimal control problem
            discretization_solver: Compiled discretization solver
            emitter_function: Callback for iteration progress
            jax_constraints: JIT-compiled constraint functions

        Returns:
            True if J_tr, J_vb, and J_vc are all below their thresholds.
        """
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
            vc_mat,
            tr_mat,
        ) = self._subproblem(
            params.items(),
            state,
            discretization_solver,
            ocp,
            settings,
            jax_constraints,
        )

        # Update state in place by appending to history
        # The x_guess/u_guess properties will automatically return the latest entry
        state.V_history.append(V_multi_shoot)
        state.X.append(x_sol)
        state.U.append(u_sol)
        state.VC_history.append(vc_mat)
        state.TR_history.append(tr_mat)

        state.J_tr = np.sum(np.array(J_tr_vec))
        state.J_vb = np.sum(np.array(J_vb_vec))
        state.J_vc = np.sum(np.array(J_vc_vec))

        # Update weights in state
        update_scp_weights(state, settings, state.k)

        # Emit data
        emitter_function(
            {
                "iter": state.k,
                "dis_time": dis_time * 1000.0,
                "subprop_time": subprop_time * 1000.0,
                "J_total": J_total,
                "J_tr": state.J_tr,
                "J_vb": state.J_vb,
                "J_vc": state.J_vc,
                "cost": cost[-1],
                "prob_stat": prob_stat,
            }
        )

        # Increment iteration counter
        state.k += 1

        # Return convergence status
        return (
            (state.J_tr < settings.scp.ep_tr)
            and (state.J_vb < settings.scp.ep_vb)
            and (state.J_vc < settings.scp.ep_vc)
        )

    def _subproblem(
        self,
        params,
        state: AlgorithmState,
        discretization_solver,
        ocp: cp.Problem,
        settings: Config,
        jax_constraints: "LoweredJaxConstraints",
    ):
        """Solve a single convex subproblem.

        Args:
            params: Problem parameters (as items iterator)
            state: Current solver state
            discretization_solver: Compiled discretization solver
            ocp: CVXPy optimal control problem
            settings: Configuration object
            jax_constraints: JIT-compiled constraint functions

        Returns:
            Tuple containing solution data, costs, and timing information.
        """
        ocp.param_dict["x_bar"].value = state.x
        ocp.param_dict["u_bar"].value = state.u

        # Convert parameters to dictionary
        param_dict = dict(params)

        t0 = time.time()
        A_bar, B_bar, C_bar, x_prop, V_multi_shoot = discretization_solver.call(
            state.x, state.u.astype(float), param_dict
        )

        ocp.param_dict["A_d"].value = A_bar.__array__()
        ocp.param_dict["B_d"].value = B_bar.__array__()
        ocp.param_dict["C_d"].value = C_bar.__array__()
        ocp.param_dict["x_prop"].value = x_prop.__array__()
        dis_time = time.time() - t0

        # Update nodal constraint linearization parameters
        # TODO: (norrisg) investigate why we are passing `0` for the node here
        if jax_constraints.nodal:
            for g_id, constraint in enumerate(jax_constraints.nodal):
                ocp.param_dict["g_" + str(g_id)].value = np.asarray(
                    constraint.func(state.x, state.u, 0, param_dict)
                )
                ocp.param_dict["grad_g_x_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_x(state.x, state.u, 0, param_dict)
                )
                ocp.param_dict["grad_g_u_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_u(state.x, state.u, 0, param_dict)
                )

        # Update cross-node constraint linearization parameters
        if jax_constraints.cross_node:
            for g_id, constraint in enumerate(jax_constraints.cross_node):
                # Cross-node constraints take (X, U, params) not (x, u, node, params)
                ocp.param_dict["g_cross_" + str(g_id)].value = np.asarray(
                    constraint.func(state.x, state.u, param_dict)
                )
                ocp.param_dict["grad_g_X_cross_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_X(state.x, state.u, param_dict)
                )
                ocp.param_dict["grad_g_U_cross_" + str(g_id)].value = np.asarray(
                    constraint.grad_g_U(state.x, state.u, param_dict)
                )

        # Convex constraints are already lowered and handled in the OCP, no action needed here

        # Initialize lam_vc as matrix if it's still a scalar in state
        if isinstance(state.lam_vc, (int, float)):
            # Convert scalar to matrix: (N-1, n_states)
            state.lam_vc = np.ones((settings.scp.n - 1, settings.sim.n_states)) * state.lam_vc

        # Update CVXPy parameters from state
        ocp.param_dict["w_tr"].value = state.w_tr
        ocp.param_dict["lam_cost"].value = state.lam_cost
        ocp.param_dict["lam_vc"].value = state.lam_vc
        ocp.param_dict["lam_vb"].value = state.lam_vb

        t0 = time.time()
        self._solve_ocp()
        subprop_time = time.time() - t0

        x_new_guess = (
            settings.sim.S_x @ ocp.var_dict["x"].value.T + np.expand_dims(settings.sim.c_x, axis=1)
        ).T
        u_new_guess = (
            settings.sim.S_u @ ocp.var_dict["u"].value.T + np.expand_dims(settings.sim.c_u, axis=1)
        ).T

        # Calculate costs from boundary conditions using utility function
        # Note: The original code only considered final_type, but the utility handles both
        # Here we maintain backward compatibility by only using final_type
        costs = [0]
        for i, bc_type in enumerate(settings.sim.x.final_type):
            if bc_type == "Minimize":
                costs += x_new_guess[:, i]
            elif bc_type == "Maximize":
                costs -= x_new_guess[:, i]

        # Create the block diagonal matrix using jax.numpy.block
        inv_block_diag = np.block(
            [
                [
                    settings.sim.inv_S_x,
                    np.zeros((settings.sim.inv_S_x.shape[0], settings.sim.inv_S_u.shape[1])),
                ],
                [
                    np.zeros((settings.sim.inv_S_u.shape[0], settings.sim.inv_S_x.shape[1])),
                    settings.sim.inv_S_u,
                ],
            ]
        )

        # Calculate J_tr_vec using the JAX-compatible block diagonal matrix
        tr_mat = inv_block_diag @ np.hstack((x_new_guess - state.x, u_new_guess - state.u)).T
        J_tr_vec = la.norm(tr_mat, axis=0) ** 2
        vc_mat = np.abs(ocp.var_dict["nu"].value)
        J_vc_vec = np.sum(vc_mat, axis=1)

        id_ncvx = 0
        J_vb_vec = 0
        if jax_constraints.nodal:
            for constraint in jax_constraints.nodal:
                J_vb_vec += np.maximum(0, ocp.var_dict["nu_vb_" + str(id_ncvx)].value)
                id_ncvx += 1

        # Add cross-node constraint violations
        id_cross = 0
        if jax_constraints.cross_node:
            for constraint in jax_constraints.cross_node:
                J_vb_vec += np.maximum(0, ocp.var_dict["nu_vb_cross_" + str(id_cross)].value)
                id_cross += 1

        # Convex constraints are already handled in the OCP, no processing needed here
        return (
            x_new_guess,
            u_new_guess,
            costs,
            ocp.value,
            J_vb_vec,
            J_vc_vec,
            J_tr_vec,
            ocp.status,
            V_multi_shoot,
            subprop_time,
            dis_time,
            vc_mat,
            abs(tr_mat),
        )
