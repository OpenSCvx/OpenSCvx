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


def solve_ivp_rk45(
    f,
    tau_final: float,
    y_0,
    args,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    is_not_compiled: bool = False,
):
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    h = (tau_final - tau_0) / (len(substeps) - 1)
    solution = jnp.zeros((len(substeps), len(y_0)))
    solution = solution.at[0].set(y_0)

    if is_not_compiled:
        for i in range(1, len(substeps)):
            t = tau_0 + i * h
            solution = solution.at[i].set(rk45_step(f, t, solution[i - 1], h, *args))
    else:

        def body_fun(i, val):
            t, y, V_result = val
            y_next = rk45_step(f, t, y, h, *args)
            V_result = V_result.at[i].set(y_next)
            return (t + h, y_next, V_result)

        _, _, solution = jax.lax.fori_loop(
            1, len(substeps), body_fun, (tau_0, y_0, solution)
        )

    return solution


def solve_ivp_diffrax(
    f,
    tau_final,
    y_0,
    args,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    solver_name="Dopri8",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    extra_kwargs=None,
):
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=(tau_final - tau_0) / (len(substeps) - 1),
        y0=y_0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(ts=substeps),
        **(extra_kwargs or {}),
    )

    return solution.ys


# TODO: (norrisg) this function is basically identical to `solve_ivp_diffrax`, could combine, but requires returning solution and getting `.ys` wherever the `solve_ivp_diffrax` is called
def solve_ivp_diffrax_prop(
    f,
    tau_final,
    y_0,
    args,
    tau_0: float = 0.0,
    num_substeps: int = 50,
    solver_name="Dopri8",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    extra_kwargs=None,
):
    substeps = jnp.linspace(tau_0, tau_final, num_substeps)

    solver_class = SOLVER_MAP.get(solver_name)
    if solver_class is None:
        raise ValueError(f"Unknown solver: {solver_name}")
    solver = solver_class()

    term = dfx.ODETerm(lambda t, y, args: f(t, y, *args))
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
    solution = dfx.diffeqsolve(
        term,
        solver=solver,
        t0=tau_0,
        t1=tau_final,
        dt0=(tau_final - tau_0) / (len(substeps) - 1),
        y0=y_0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(dense=True, ts=substeps),
        **(extra_kwargs or {}),
    )

    return solution
