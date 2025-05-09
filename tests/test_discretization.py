import numpy as np
import jax.numpy as jnp
import jax
import pytest
from openscvx.discretization import get_discretization_solver, dVdt


# --- fixtures for dummy params, state_dot, A, B  ------------------

# dummy parameter namespace
class Dummy: pass

@pytest.fixture
def params():
    p = Dummy()
    p.sim = Dummy();  p.sim.n_states = 2;  p.sim.n_controls = 1
    p.scp = Dummy();  p.scp.n = 5
    p.dis = Dummy()
    p.dis.custom_integrator = True
    p.dis.solver = "Tsit5"
    p.dis.rtol = 1e-3
    p.dis.atol = 1e-6
    p.dis.args = {}
    p.dis.dis_type = "FOH"
    p.dev = Dummy(); p.dev.debug = False
    return p

def state_dot(x, u, node):
    # simple linear: x' = A_true x + B_true u
    return x + u

def A(x, u, node):
    batch = x.shape[0]
    eye = jnp.eye(2)
    return jnp.broadcast_to(eye, (batch, 2, 2))

def B(x, u, node):
    batch = x.shape[0]
    ones = jnp.ones((2,1))
    return jnp.broadcast_to(ones, (batch, 2, 1))

# --- tests ---------------------------------------------------------

def test_discretization_shapes(params):
    # build solver
    solver = get_discretization_solver(state_dot, A, B, params)

    # dummy x,u
    x = jnp.ones((params.scp.n, params.sim.n_states))
    u = jnp.ones((params.scp.n, params.sim.n_controls + 1))  # +1 slack

    A_bar, B_bar, C_bar, z_bar, Vmulti = solver(x, u)

    # expected shapes
    N = params.scp.n
    n_x, n_u = params.sim.n_states, params.sim.n_controls
    assert A_bar.shape == ((N-1), n_x*n_x)
    assert B_bar.shape == ((N-1), n_x*n_u)
    assert C_bar.shape == ((N-1), n_x*n_u)
    assert z_bar.shape == ((N-1), n_x)
    # assert Vmulti.shape == (N, (n_x + n_x*n_x + 2*n_x*n_u + n_x))

def test_jit_dVdt_compiles(params):
    # prepare trivial inputs
    n_x, n_u = params.sim.n_states, params.sim.n_controls
    N = params.scp.n
    aug_dim = n_x + n_x*n_x + 2*n_x*n_u + n_x

    tau    = jnp.array(0.3)
    V_flat = jnp.ones((N-1) * aug_dim)
    u_cur  = jnp.ones((N-1, n_u+1))
    u_next = jnp.ones((N-1, n_u+1))

    # bind out the Python callables & params
    def wrapped(tau_, V_):
        return dVdt(tau_, V_, u_cur, u_next, state_dot, A, B, n_x, n_u, N, params.dis.dis_type)

    # now JIT only over (tau_, V_)
    jitted = jax.jit(wrapped)
    lowered = jitted.lower(tau, V_flat)
    # compile will fail if there’s a trace issue
    lowered.compile()

@pytest.mark.parametrize("integrator", ["custom_integrator", "diffrax"])
def test_jit_discretization_solver_compiles(params, integrator):
    # flip between the two modes
    if integrator == "custom_integrator":
        params.dis.custom_integrator = True
    elif integrator == "diffrax":
        params.dis.custom_integrator = False

    # build the solver (captures only hashable primitives)
    solver = get_discretization_solver(state_dot, A, B, params)

    # dummy x,u (including slack column)
    x = jnp.ones((params.scp.n, params.sim.n_states))
    u = jnp.ones((params.scp.n, params.sim.n_controls + 1))

    # jit & lower & compile
    jitted = jax.jit(solver)
    lowered = jitted.lower(x, u)
    # will raise if there’s any hash or trace error
    lowered.compile()
