"""Penalized Trust Region (PTR) successive convexification algorithm.

This module implements the PTR algorithm for solving non-convex trajectory
optimization problems through iterative convex approximation.
"""

import time
import warnings
from typing import TYPE_CHECKING, List

import cvxpy as cp
import numpy as np
import numpy.linalg as la

from openscvx.config import Config

from .autotuning import update_scp_weights
from .base import Algorithm, AlgorithmState

if TYPE_CHECKING:
    from openscvx.lowered import LoweredJaxConstraints
    from openscvx.solvers import ConvexSolver

warnings.filterwarnings("ignore")


def _set_param(prob: cp.Problem, name: str, value: np.ndarray) -> None:
    """Set a CVXPY parameter with helpful error messages on failure.

    Args:
        prob: The CVXPY problem containing the parameter.
        name: The parameter name in prob.param_dict.
        value: The value to assign.

    Raises:
        ValueError: If the value is not real, with diagnostic information.
    """
    try:
        prob.param_dict[name].value = value
    except ValueError as e:
        if "must be real" in str(e):
            arr = np.asarray(value)
            nan_mask = ~np.isfinite(arr)
            nan_indices = np.argwhere(nan_mask)

            # Build list of "index -> value" strings
            index_value_strs = [
                f"  {tuple(int(i) for i in idx)} -> {arr[tuple(idx)]}" for idx in nan_indices[:20]
            ]
            if len(nan_indices) > 20:
                index_value_strs.append(f"  ... and {len(nan_indices) - 20} more")

            arr_str = np.array2string(arr, threshold=200, edgeitems=3, max_line_width=120)
            msg = (
                f"Parameter '{name}' with shape {arr.shape} contains {len(nan_indices)} non-real"
                " value(s):\n" + "\n".join(index_value_strs) + f"\n\n{name} = {arr_str}"
            )
            raise ValueError(msg) from e
        raise


class PenalizedTrustRegion(Algorithm):
    """Penalized Trust Region (PTR) successive convexification algorithm.

    PTR solves non-convex trajectory optimization problems through iterative
    convex approximation. Each subproblem balances competing cost terms:

    - **Trust region penalty**: Discourages large deviations from the previous
      iterate, keeping the solution within the region where linearization is valid.
    - **Virtual control**: Relaxes dynamics constraints, penalized to drive
      defects toward zero as the algorithm converges.
    - **Virtual buffer**: Relaxes non-convex constraints, similarly penalized
      to enforce feasibility at convergence.
    - **Problem objective and other terms**: The user-defined cost (e.g., minimum
      fuel, minimum time) and any additional penalty terms.

    The interplay between these terms guides the optimization: the trust region
    anchors the solution near the linearization point while virtual terms allow
    temporary constraint violations that shrink over iterations.

    Example:
        Using PTR with a Problem::

            from openscvx.algorithms import PenalizedTrustRegion

            problem = Problem(dynamics, constraints, states, controls, N, time)
            problem.initialize()
            result = problem.solve()
    """

    def __init__(self):
        """Initialize PTR with unset infrastructure.

        Call initialize() before step() to set up compiled components.
        """
        self._solver: "ConvexSolver" = None
        self._discretization_solver: callable = None
        self._jax_constraints: "LoweredJaxConstraints" = None
        self._emitter: callable = None

    def initialize(
        self,
        solver: "ConvexSolver",
        discretization_solver: callable,
        jax_constraints: "LoweredJaxConstraints",
        emitter: callable,
        params: dict,
        settings: Config,
    ) -> None:
        """Initialize PTR algorithm.

        Stores compiled infrastructure and performs a warm-start solve to
        initialize DPP and JAX jacobians.

        Args:
            solver: Convex subproblem solver (e.g., CVXPySolver)
            discretization_solver: Compiled discretization solver
            jax_constraints: JIT-compiled constraint functions
            emitter: Callback for emitting iteration progress
            params: Problem parameters dictionary (for warm-start)
            settings: Configuration object (for warm-start)
        """
        # Store immutable infrastructure
        self._solver = solver
        self._discretization_solver = discretization_solver
        self._jax_constraints = jax_constraints
        self._emitter = emitter

        # Access the underlying problem for parameter initialization
        ocp = self._solver.problem

        if "x_init" in ocp.param_dict:
            _set_param(ocp, "x_init", settings.sim.x.initial)

        if "x_term" in ocp.param_dict:
            _set_param(ocp, "x_term", settings.sim.x.final)

        # Create temporary state for initialization solve
        init_state = AlgorithmState.from_settings(settings)

        # Solve a dumb problem to initialize DPP and JAX jacobians
        _ = self._subproblem(params, init_state, settings)

    def step(
        self,
        state: AlgorithmState,
        params: dict,
        settings: Config,
    ) -> bool:
        """Execute one PTR iteration.

        Solves the convex subproblem, updates state in place, and checks
        convergence based on trust region, virtual buffer, and virtual
        control costs.

        Args:
            state: Mutable solver state (modified in place)
            params: Problem parameters dictionary (may change between steps)
            settings: Configuration object (may change between steps)

        Returns:
            True if J_tr, J_vb, and J_vc are all below their thresholds.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._solver is None:
            raise RuntimeError(
                "PenalizedTrustRegion.step() called before initialize(). "
                "Call initialize() first to set up compiled infrastructure."
            )

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
        ) = self._subproblem(params, state, settings)

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
        self._emitter(
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
        params: dict,
        state: AlgorithmState,
        settings: Config,
    ):
        """Solve a single convex subproblem.

        Uses stored infrastructure (solver, discretization_solver, jax_constraints)
        with per-step params and settings.

        Args:
            params: Problem parameters dictionary
            state: Current solver state
            settings: Configuration object

        Returns:
            Tuple containing solution data, costs, and timing information.
        """
        # Get the underlying CVXPy problem from the solver
        ocp = self._solver.problem

        _set_param(ocp, "x_bar", state.x)
        _set_param(ocp, "u_bar", state.u)

        param_dict = params

        t0 = time.time()
        A_bar, B_bar, C_bar, x_prop, V_multi_shoot = self._discretization_solver.call(
            state.x, state.u.astype(float), param_dict
        )

        _set_param(ocp, "A_d", A_bar.__array__())
        _set_param(ocp, "B_d", B_bar.__array__())
        _set_param(ocp, "C_d", C_bar.__array__())
        _set_param(ocp, "x_prop", x_prop.__array__())
        dis_time = time.time() - t0

        # Update nodal constraint linearization parameters
        # TODO: (norrisg) investigate why we are passing `0` for the node here
        if self._jax_constraints.nodal:
            for g_id, constraint in enumerate(self._jax_constraints.nodal):
                _set_param(
                    ocp,
                    f"g_{g_id}",
                    np.asarray(constraint.func(state.x, state.u, 0, param_dict)),
                )
                _set_param(
                    ocp,
                    f"grad_g_x_{g_id}",
                    np.asarray(constraint.grad_g_x(state.x, state.u, 0, param_dict)),
                )
                _set_param(
                    ocp,
                    f"grad_g_u_{g_id}",
                    np.asarray(constraint.grad_g_u(state.x, state.u, 0, param_dict)),
                )

        # Update cross-node constraint linearization parameters
        if self._jax_constraints.cross_node:
            for g_id, constraint in enumerate(self._jax_constraints.cross_node):
                # Cross-node constraints take (X, U, params) not (x, u, node, params)
                _set_param(
                    ocp,
                    f"g_cross_{g_id}",
                    np.asarray(constraint.func(state.x, state.u, param_dict)),
                )
                _set_param(
                    ocp,
                    f"grad_g_X_cross_{g_id}",
                    np.asarray(constraint.grad_g_X(state.x, state.u, param_dict)),
                )
                _set_param(
                    ocp,
                    f"grad_g_U_cross_{g_id}",
                    np.asarray(constraint.grad_g_U(state.x, state.u, param_dict)),
                )

        # Convex constraints are already lowered and handled in the OCP, no action needed here

        # Initialize lam_vc as matrix if it's still a scalar in state
        if isinstance(state.lam_vc, (int, float)):
            # Convert scalar to matrix: (N-1, n_states)
            state.lam_vc = np.ones((settings.scp.n - 1, settings.sim.n_states)) * state.lam_vc

        # Update CVXPy parameters from state
        _set_param(ocp, "w_tr", state.w_tr)
        _set_param(ocp, "lam_cost", state.lam_cost)
        _set_param(ocp, "lam_vc", state.lam_vc)
        _set_param(ocp, "lam_vb", state.lam_vb)

        t0 = time.time()
        self._solver.solve(state, params, settings)
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
        if self._jax_constraints.nodal:
            for constraint in self._jax_constraints.nodal:
                J_vb_vec += np.maximum(0, ocp.var_dict["nu_vb_" + str(id_ncvx)].value)
                id_ncvx += 1

        # Add cross-node constraint violations
        id_cross = 0
        if self._jax_constraints.cross_node:
            for constraint in self._jax_constraints.cross_node:
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

    def citation(self) -> List[str]:
        """Return BibTeX citations for the PTR algorithm.

        Returns:
            List containing the BibTeX entry for the PTR paper.
        """
        return [
            r"""@article{drusvyatskiy2018error,
  title={Error bounds, quadratic growth, and linear convergence of proximal methods},
  author={Drusvyatskiy, Dmitriy and Lewis, Adrian S},
  journal={Mathematics of operations research},
  volume={43},
  number={3},
  pages={919--948},
  year={2018},
  publisher={INFORMS}
}""",
            r"""@article{szmuk2020successive,
  title={Successive convexification for real-time six-degree-of-freedom powered descent guidance
    with state-triggered constraints},
  author={Szmuk, Michael and Reynolds, Taylor P and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal={Journal of Guidance, Control, and Dynamics},
  volume={43},
  number={8},
  pages={1399--1413},
  year={2020},
  publisher={American Institute of Aeronautics and Astronautics}
}""",
            r"""@article{reynolds2020dual,
  title={Dual quaternion-based powered descent guidance with state-triggered constraints},
  author={Reynolds, Taylor P and Szmuk, Michael and Malyuta, Danylo and Mesbahi, Mehran and
    A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et and Carson III, John M},
  journal={Journal of Guidance, Control, and Dynamics},
  volume={43},
  number={9},
  pages={1584--1599},
  year={2020},
  publisher={American Institute of Aeronautics and Astronautics}
}""",
        ]
