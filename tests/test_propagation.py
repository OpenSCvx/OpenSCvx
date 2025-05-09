# test_propagation.py

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from openscvx.propagation import prop_aug_dy, s_to_t, t_to_tau, get_propagation_solver


# simple scalar decay: x' = -x
def decay(x, u, node):
    return -x


class Dummy:
    pass


@pytest.mark.parametrize("dis_type,beta_expected", [("ZOH", 0.0), ("FOH", 1.0)])
def test_prop_aug_dy_linear(dis_type, beta_expected):
    """
    prop_aug_dy should compute:
      u = u_cur + beta*(u_next - u_cur)
      return u[:,idx_s] * state_dot(x_batch, u[:,:-1]).squeeze()
    for both ZOH (beta=0) and FOH (beta=(tau-tau_init)*N).
    """
    tau = 0.2
    tau_init = 0.0
    N = 5
    idx_s = 1
    x = np.array([1.0, 2.0])
    u_cur = np.array([[0.5, 3.0]])
    u_next = np.array([[1.5, 5.0]])

    node = 0  # dummy node index

    # compute beta
    if dis_type == "ZOH":
        beta = 0.0
    else:
        beta = (tau - tau_init) * N
    assert pytest.approx(beta) == beta_expected

    # manually compute expected
    u = u_cur + beta * (u_next - u_cur)
    # use a simple state_dot: x + u (with broadcasting)
    expected = u[:, idx_s] * (x + u[:, 0])

    out = prop_aug_dy(
        tau,
        x,
        u_cur,
        u_next,
        tau_init,
        node,
        idx_s,
        lambda x_batch, u_control, node: x_batch + u_control,  # state_dot
        dis_type,
        N,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_s_to_t_basic(dis_type):
    """
    s_to_t should accumulate time steps correctly under both ZOH and FOH.
    """
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 4
    p.dis = Dummy()
    p.dis.dis_type = dis_type

    # build u with slack values [1,2,3,4]
    u = np.stack([[0.0, float(s)] for s in [1, 2, 3, 4]])

    t = s_to_t(u, p)

    # manually reconstruct expected t
    tau = np.linspace(0, 1, p.scp.n)
    expected = [0.0]
    for k in range(1, p.scp.n):
        s_kp = u[k - 1, -1]
        s_k = u[k, -1]
        if dis_type == "ZOH":
            dt = (tau[k] - tau[k - 1]) * s_kp
        else:
            dt = 0.5 * (s_k + s_kp) * (tau[k] - tau[k - 1])
        expected.append(expected[-1] + dt)

    np.testing.assert_allclose(t, expected, rtol=1e-6)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_t_to_tau_constant_slack(dis_type):
    """
    t_to_tau should invert s_to_t back to the original tau grid when slack is constant.
    Also, the interpolated u should exactly match u_nodal in that case.
    """
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 4
    p.dis = Dummy()
    p.dis.dis_type = dis_type

    # constant slack = 2.0, control doesn't matter
    N = p.scp.n
    u_nodal = np.tile(np.array([0.0, 2.0]), (N, 1))

    # get the “nodal” times via s_to_t
    t_nodal = s_to_t(u_nodal, p)

    # invert back
    tau, u_interp = t_to_tau(u_nodal, t_nodal, u_nodal, np.array(t_nodal), p)

    np.testing.assert_allclose(tau, np.linspace(0, 1, N), rtol=1e-6)
    # since slack & control are constant, interpolation must reprodu


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_propagation_solver_decay(dis_type):
    """
    Propagation solver should approximate exp(-t) over [0,1] with ~1% error,
    for both zero-order hold and first-order hold.
    """
    # ——— build dummy params ———
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 5  # N only matters for FOH, but u_cur==u_next here
    p.dis = Dummy()
    p.dis.dis_type = dis_type

    solver = get_propagation_solver(decay, p)

    # initial state, time grid
    V0 = jnp.array([1.0])
    tau_grid = jnp.array([0.0, 1.0])

    # we'll hold control constant and slack=1 so f(x)=−x
    # u arrays must be shape (1, n_u+1)= (1,2): [control, slack]
    u_cur = np.array([[0.0, 1.0]])
    u_next = np.array([[0.0, 1.0]])

    tau_init = float(tau_grid[0])
    idx_s = 1  # slack lives in column 1

    node = 0  # dummy node index

    sol = solver(V0, tau_grid, u_cur, u_next, tau_init, node, idx_s)

    # check discrete output
    ys = np.array(sol.ys[:, 0])
    times = np.linspace(0.0, 1.0, ys.shape[0])
    expected = np.exp(-times)
    # about 1% relative tolerance
    np.testing.assert_allclose(ys, expected, rtol=1e-2, atol=1e-3)

    # check dense evaluator at t=0.5
    y_half = float(sol.evaluate(0.5)[0])
    assert np.isclose(y_half, np.exp(-0.5), rtol=1e-2, atol=1e-3)

    # confirm shape: default num_substeps=50
    assert sol.ys.shape == (50, 1)


@pytest.mark.parametrize("dis_type", ["ZOH", "FOH"])
def test_jit_propagation_solver_compiles(dis_type):
    """
    Ensure that the propagation solver's .ys output can be jitted without errors.
    """
    # — build dummy params —
    p = Dummy()
    p.scp = Dummy()
    p.scp.n = 5
    p.dis = Dummy()
    p.dis.dis_type = dis_type

    solver = get_propagation_solver(decay, p)

    # — dummy inputs for a single integration step —
    V0 = jnp.array([1.0])
    tau_grid = jnp.array([0.0, 1.0])
    u_cur = jnp.array([[0.0, 1.0]])
    u_next = jnp.array([[0.0, 1.0]])
    tau_init = tau_grid[0]
    idx_s = 1

    node = 0  # dummy node index

    # JIT only the ys output (the array of solution states)
    jitted = jax.jit(
        lambda V0, tau_grid, u_cur, u_next, tau_init, node, idx_s: solver(
            V0, tau_grid, u_cur, u_next, tau_init, node, idx_s
        ).ys
    )
    # Lower & compile
    lowered = jitted.lower(V0, tau_grid, u_cur, u_next, tau_init, node, idx_s)
    lowered.compile()
