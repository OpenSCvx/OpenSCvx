import jax
import jax.numpy as jnp
import diffrax as dfx

SOLVER_MAP = {
    "Tsit5": dfx.Tsit5,
    "Euler": dfx.Euler,
    "Heun": dfx.Heun,
    "Midpoint": dfx.Midpoint,
    "Ralston": dfx.Ralston,
    "Dopri5": dfx.Dopri5,
    "Dopri8": dfx.Dopri8,
    "Bosh3": dfx.Bosh3,
    "ReversibleHeun": dfx.ReversibleHeun,
    "ImplicitEuler": dfx.ImplicitEuler,
    "KenCarp3": dfx.KenCarp3,
    "KenCarp4": dfx.KenCarp4,
    "KenCarp5": dfx.KenCarp5,
}

# fmt: off
def rk45_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + h/4, y + h*k1/4, *args)
    k3 = f(t + 3*h/8, y + 3*h*k1/32 + 9*h*k2/32, *args)
    k4 = f(t + 12*h/13, y + 1932*h*k1/2197 - 7200*h*k2/2197 + 7296*h*k3/2197, *args)
    k5 = f(t + h, y + 439*h*k1/216 - 8*h*k2 + 3680*h*k3/513 - 845*h*k4/4104, *args)
    y_next = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
    return y_next
# fmt: on


def solve_ivp_rk45(f, tau_grid, y_0, args, debug, t_eval=None, num_steps=50):
    if t_eval is None:
        t_eval = jnp.linspace(tau_grid[0], tau_grid[1], num_steps)

    h = (tau_grid[1] - tau_grid[0]) / (len(t_eval) - 1)
    V_result = jnp.zeros((len(t_eval), len(y_0)))
    V_result = V_result.at[0].set(y_0)

    if debug:
        for i in range(1, len(t_eval)):
            t = tau_grid[0] + i * h
            y_next = rk45_step(f, t, V_result[i - 1], h, *args)
            V_result = V_result.at[i].set(y_next)
    else:

        def body_fun(i, val):
            t, y, V_result = val
            y_next = rk45_step(f, t, y, h, *args)
            V_result = V_result.at[i].set(y_next)
            return (t + h, y_next, V_result)

        _, _, V_result = jax.lax.fori_loop(
            1, len(t_eval), body_fun, (tau_grid[0], y_0, V_result)
        )

    return V_result


def solve_ivp_diffrax(
    dVdt,
    tau_grid,
    V0,
    args,
    t_eval=None,
    num_steps=50,
    solver_name="Tsit5",
    rtol=1e-3,
    atol=1e-6,
    extra_kwargs=None,
):
    t_eval = jnp.linspace(tau_grid[0], tau_grid[1], num_steps)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: dVdt(t, y, *args))
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_grid[0],
        t1=tau_grid[1],
        dt0=(tau_grid[1] - tau_grid[0]) / (len(t_eval) - 1),
        y0=V0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(ts=t_eval),
        **(extra_kwargs or {}),
    )

    return solution.ys
