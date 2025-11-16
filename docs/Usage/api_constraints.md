# Constraints

Constraints in openscvx are created using symbolic expressions with comparison operators (`==`, `<=`, `>=`). By default, constraints are enforced at discrete nodes along the trajectory (nodal constraints). The symbolic expression system provides two specialized constraint wrappers for precise control over when and how constraints are enforced.

## Basic Constraints

All basic constraints are automatically enforced at all discrete nodes unless wrapped with `.at()` or `.over()`.

### Equality

::: openscvx.symbolic.expr.constraint.Equality
    options:
      show_source: false
      show_root_heading: true

### Inequality

::: openscvx.symbolic.expr.constraint.Inequality
    options:
      show_source: false
      show_root_heading: true

## Specialized Constraint Wrappers

### NodalConstraint

NodalConstraint allows selective enforcement of constraints at specific time points (nodes) in a discretized trajectory. Created using the `.at()` method on constraints. **Note:** Bare constraints without `.at()` or `.over()` are automatically converted to NodalConstraints applied at all nodes.

::: openscvx.symbolic.expr.constraint.NodalConstraint
    options:
      show_source: false
      show_root_heading: true

### CTCS (Continuous-Time Constraint Satisfaction)

CTCS guarantees strict constraint satisfaction throughout the entire continuous trajectory, not just at discrete nodes. It works by augmenting the state vector with additional states whose dynamics integrate constraint violation penalties. Created using the `.over()` method on constraints.

::: openscvx.symbolic.expr.constraint.CTCS
    options:
      show_source: false
      show_root_heading: true 