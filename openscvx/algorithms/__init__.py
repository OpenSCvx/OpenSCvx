"""Successive convexification algorithms for trajectory optimization.

This module provides implementations of SCvX (Successive Convexification) algorithms
for solving non-convex trajectory optimization problems through iterative convex
approximation.

Current Implementations:
    PTR (Penalized Trust Region): The default SCvX algorithm using trust region
        methods with penalty-based constraint handling. Includes adaptive parameter
        tuning and virtual control relaxation.

Core Components:
    SolverState: Mutable state container for algorithm iterations, tracking
        trajectory evolution, weights, and convergence metrics.

    OptimizationResults: Results container with solution trajectory, costs,
        constraint satisfaction, and solver diagnostics.

    update_scp_weights: Adaptive parameter tuning for trust region weights
        and penalty parameters during optimization.

Algorithm Functions:
    PTR_init: Initialize solver state for PTR algorithm
    PTR_step: Execute one iteration of PTR algorithm
    format_result: Convert solver state to OptimizationResults

Planned Architecture (ABC-based):
    A base class will be introduced to enable pluggable algorithm implementations.
    Future algorithms will implement the SCvXAlgorithm interface:

    .. code-block:: python

        # algorithms/base.py (planned):
        class SCvXAlgorithm(ABC):
            @abstractmethod
            def initialize(self, lowered: LoweredProblem) -> SolverState:
                '''Initialize solver state from a lowered problem.'''
                ...

            @abstractmethod
            def step(self, state: SolverState, solver: ConvexSolver) -> SolverState:
                '''Execute one iteration of the algorithm.'''
                ...

    This will enable users to implement custom SCvX variants such as:
    - GuSTO (Guaranteed Sequential Trajectory Optimization)
    - Vanilla SCvX
    - ALTRO (Augmented Lagrangian Trajectory Optimization)
    - Custom research algorithms
"""

from .optimization_results import OptimizationResults
from .ptr import PTR_init, PTR_step, format_result
from .solver_state import SolverState

__all__ = [
    # Core state and results
    "SolverState",
    "OptimizationResults",
    # PTR algorithm
    "PTR_init",
    "PTR_step",
    "format_result",
]
