# API
!!! Warning
    This page is still under development :construction:.

<!-- ## Dynamics -->

::: openscvx.dynamics.Dynamics
    options:
      show_source: false
      show_root_heading: true

## Constraints

Constraints in openscvx are created using symbolic expressions with comparison operators (`==`, `<=`, `>=`). By default, constraints are enforced at discrete nodes along the trajectory (nodal constraints). The symbolic expression system provides two specialized constraint wrappers for precise control over when and how constraints are enforced.

### Basic Constraints

All basic constraints are automatically enforced at all discrete nodes unless wrapped with `.at()` or `.over()`.

#### Equality

::: openscvx.symbolic.expr.constraint.Equality
    options:
      show_source: false
      show_root_heading: true

#### Inequality

::: openscvx.symbolic.expr.constraint.Inequality
    options:
      show_source: false
      show_root_heading: true

### Specialized Constraint Wrappers

#### NodalConstraint

NodalConstraint allows selective enforcement of constraints at specific time points (nodes) in a discretized trajectory. Created using the `.at()` method on constraints. **Note:** Bare constraints without `.at()` or `.over()` are automatically converted to NodalConstraints applied at all nodes.

::: openscvx.symbolic.expr.constraint.NodalConstraint
    options:
      show_source: false
      show_root_heading: true

#### CTCS (Continuous-Time Constraint Satisfaction)

CTCS guarantees strict constraint satisfaction throughout the entire continuous trajectory, not just at discrete nodes. It works by augmenting the state vector with additional states whose dynamics integrate constraint violation penalties. Created using the `.over()` method on constraints.

::: openscvx.symbolic.expr.constraint.CTCS
    options:
      show_source: false
      show_root_heading: true

## Integrators

### RK45Integrator

::: openscvx.integrators.solve_ivp_rk45
    options:
      show_source: false
      show_root_heading: true

::: openscvx.integrators.rk45_step
    options:
      show_source: false
      show_root_heading: true

### Diffrax Integrators

::: openscvx.integrators.solve_ivp_diffrax
    options:
      show_source: false
      show_root_heading: true

::: openscvx.integrators.solve_ivp_diffrax_prop
    options:
      show_source: false
      show_root_heading: true

## TrajOptProblem

::: openscvx.trajoptproblem.TrajOptProblem.__init__
    options:
      show_source: false
      show_root_heading: true


### ScpConfig

::: openscvx.config.ScpConfig.__init__
    options:
      show_source: false
      show_root_heading: true

### DiscretizationConfig

::: openscvx.config.DiscretizationConfig.__init__
    options:
      show_source: false
      show_root_heading: true

### PropagationConfig

::: openscvx.config.PropagationConfig.__init__
    options:
      show_source: false
      show_root_heading: true

### SimConfig

::: openscvx.config.SimConfig.__init__
    options:
      show_source: false
      show_root_heading: true

### ConvexSolverConfig

::: openscvx.config.ConvexSolverConfig.__init__
    options:
      show_source: false
      show_root_heading: true

### DevConfig

::: openscvx.config.DevConfig.__init__
    options:
      show_source: false
      show_root_heading: true