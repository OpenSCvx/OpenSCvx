# test_integrators.py

import numpy as np
import jax.numpy as jnp
import pytest

from openscvx.integrators import solve_ivp_rk45, solve_ivp_diffrax, solve_ivp_diffrax_prop

def decay(t, y):
    return -y

@pytest.mark.parametrize("num_steps", [11, 21])
def test_solve_ivp_rk45_decay(num_steps):
    """RK45 fixed-step should approximate exp(-t)."""
    t0, t1 = 0.0, 1.0
    times = jnp.linspace(t0, t1, num_steps)
    y0 = jnp.array([1.0])
    sol = solve_ivp_rk45(decay, t1, y0, args=(), is_not_compiled=False, num_substeps=num_steps)
    sol_np = np.array(sol[:, 0])
    expected = np.exp(-np.array(times))
    # allow ~1% relative error
    np.testing.assert_allclose(sol_np, expected, rtol=1e-2, atol=1e-3)

@pytest.mark.parametrize("solver_name", ["Tsit5", "Dopri5", "Dopri8", "Heun"])
@pytest.mark.parametrize("num_steps", [11, 21])
def test_solve_ivp_diffrax_decay(solver_name, num_steps):
    """Diffrax adaptive solver should approximate exp(-t)."""
    t0, t1 = 0.0, 1.0
    times = jnp.linspace(t0, t1, num_steps)
    y0 = jnp.array([1.0])
    sol = solve_ivp_diffrax(
        decay,
        t1,
        y0,
        args=(),
        solver_name=solver_name,
        rtol=1e-3,
        atol=1e-6,
        extra_kwargs=None,
        num_substeps=num_steps,
    )
    sol_np = np.array(sol[:, 0])
    expected = np.exp(-np.array(times))
    # allow slightly larger error for cheap solvers
    tol = 2e-2 if solver_name == "Euler" else 5e-3
    np.testing.assert_allclose(sol_np, expected, rtol=tol, atol=tol)

@pytest.mark.parametrize("solver_name", ["Tsit5", "Dopri5", "Dopri8"])
def test_solve_ivp_diffrax_prop_decay(solver_name):
    # Integrate y' = -y, y(0)=1, from t=0 to t=1
    tau0, tau1 = 0.0, 1.0
    y0 = jnp.array([1.0])
    args = ()  # our f ignores args

    sol = solve_ivp_diffrax_prop(
        f=decay,
        tau_final=tau1,
        y_0=y0,
        args=args,
        tau_0=tau0,
        num_substeps=11,
        solver_name=solver_name,
        rtol=1e-6,
        atol=1e-9,
        extra_kwargs={}
    )

    # Check the discrete solution at the 11 grid points
    ys = np.array(sol.ys[:, 0])
    t_eval = np.linspace(tau0, tau1, 11)
    expected = np.exp(-t_eval)
    np.testing.assert_allclose(ys, expected, rtol=1e-3, atol=1e-6)

    # Check the dense evaluator at t=0.5
    y_half = float(sol.evaluate(0.5)[0])
    assert np.isclose(y_half, np.exp(-0.5), rtol=1e-3, atol=1e-6)
