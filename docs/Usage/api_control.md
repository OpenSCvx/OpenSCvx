# Control

Control represents control input variables (actuator commands) in a trajectory optimization problem. Unlike State variables which evolve according to dynamics, Controls are direct decision variables that the optimizer can freely adjust (within specified bounds) at each time step to influence the system dynamics.

Controls support min/max bounds to enforce actuator limits and initial trajectory guesses to help the optimizer converge.

::: openscvx.symbolic.expr.control.Control
    options:
      show_source: false
      show_root_heading: true 