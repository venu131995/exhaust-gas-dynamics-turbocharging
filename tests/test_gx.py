import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gx import riemann                                                   # noqa: E402
from gx.engine import EngineSpec, cyl_volume, valve_area, wiebe_rate     # noqa: E402
from gx.gas import GAMMA, R_GAS, hllc, orifice_mass_flow                  # noqa: E402
from gx.network import Link, Network, Pipe, boundary_state, _orifice      # noqa: E402
from gx.sim import Engine, cycle_summary                                  # noqa: E402


def test_exact_riemann_sod_star_state():
    # Toro (2009), Table 4.2, test 1 (gamma = 1.4): p* = 0.30313, u* = 0.92745
    p, u = riemann.star_state(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    assert p == pytest.approx(0.30313, abs=1e-5)
    assert u == pytest.approx(0.92745, abs=1e-5)


def test_hllc_is_consistent():
    rho, u, p = np.array([1.2]), np.array([80.0]), np.array([1.5e5])
    F = hllc(rho, u, p, rho, u, p)
    E = p / (GAMMA - 1) + 0.5 * rho * u * u
    assert np.allclose([F[0], F[1], F[2]], [rho * u, rho * u * u + p, u * (E + p)])


def test_orifice_chokes_and_vanishes():
    A, p0, T0 = 1e-3, 3e5, 800.0
    crit = (2 / (GAMMA + 1)) ** (GAMMA / (GAMMA - 1))
    m_crit = orifice_mass_flow(A, p0, T0, crit * p0)
    assert orifice_mass_flow(A, p0, T0, 0.1 * p0) == pytest.approx(m_crit)     # choked: independent of p_down
    assert orifice_mass_flow(A, p0, T0, p0) == 0.0
    assert _orifice(A, p0, T0, 0.7 * p0) == pytest.approx(float(orifice_mass_flow(A, p0, T0, 0.7 * p0)))


def _closed_tube(n=200):
    net = Network([Pipe('t', 1.0, 0.1, n, f_darcy=0.0, heat_transfer=False)], [], [])
    x = (np.arange(n) + 0.5) / n
    rho = np.where(x < 0.5, 1.0, 0.125)
    p = np.where(x < 0.5, 1.0, 0.1)
    net.q = np.array([rho * net.A, 0 * net.A, p / (GAMMA - 1) * net.A])
    return net, x


def test_shock_tube_close_to_exact_and_conservative():
    net, x = _closed_tube()
    M0, E0 = (net.q[0] * net.dx).sum(), (net.q[2] * net.dx).sum()
    mU = np.zeros((2, 0))
    while net.t < 0.2 - 1e-12:
        mU, _ = net.step(min(net.max_dt(0.8), 0.2 - net.t), mU)
    r, u, p = net.pipe_state(0)
    ex = riemann.sample(x, 0.2, 0.5, (1, 0, 1), (0.125, 0, 0.1), GAMMA)
    assert np.abs(r - ex[0]).mean() < 6e-3
    assert abs((net.q[0] * net.dx).sum() - M0) < 1e-13 * M0
    assert abs((net.q[2] * net.dx).sum() - E0) < 1e-13 * E0


def test_boundary_state_limits():
    rho, p = 1.2, 1.5e5
    # closed valve -> wall: zero velocity
    assert boundary_state(rho, 0.0, p, 5e-3, 0.0, 1e5, 300.0)[1] == 0.0
    # outflow: face mass flow equals the orifice flow from the face stagnation state
    rb, wb, pb = boundary_state(rho, 0.0, p, 5e-3, 2e-3, 1e5, 300.0)
    Tb = pb / (rb * R_GAS)
    T0 = Tb + wb * wb / (2 * R_GAS * GAMMA / (GAMMA - 1))
    p0 = pb * (T0 / Tb) ** (GAMMA / (GAMMA - 1))
    assert wb > 0
    assert rb * wb * 5e-3 == pytest.approx(_orifice(2e-3, p0, T0, 1e5), rel=1e-9)
    # inflow from a higher-pressure reservoir
    assert boundary_state(rho, 0.0, p, 5e-3, 2e-3, 2e5, 300.0)[1] < 0


def test_pipe_between_reservoirs_reaches_uniform_flow():
    net = Network([Pipe('n', 1.0, 0.1, 50, f_darcy=0.0, heat_transfer=False)], [],
                  [Link('pipe-res', (0, 'L'), (2e5, 600.0), tag='in'), Link('pipe-res', (0, 'R'), (1.75e5, 600.0), tag='out')])
    net.init_pipe(0, 1.75e5, 600.0)
    mU = np.zeros((2, 0))
    while net.t < 0.1:
        mU, _ = net.step(min(net.max_dt(0.8), 0.1 - net.t), mU)
    r, u, p = net.pipe_state(0)
    md = r * u * net.A
    assert md.mean() > 1.0 and (md.max() - md.min()) < 1e-6 * md.mean()


def test_engine_geometry_and_combustion():
    sp = EngineSpec()
    assert cyl_volume(sp, 0.0)[0] == pytest.approx(sp.v_clear)
    assert cyl_volume(sp, 180.0)[0] == pytest.approx(sp.v_clear + sp.v_disp)
    assert valve_area(sp.exh, 100.0) == 0.0 and valve_area(sp.exh, 250.0) > 0
    th = np.linspace(-30, 120, 20001)
    burned = np.trapezoid([wiebe_rate(sp, t % 720) for t in th], th)
    # the rate is evaluated up to 1.6 x the nominal duration, so the whole charge burns
    assert burned == pytest.approx(1.0 - math.exp(-sp.wiebe_a * 1.6 ** (sp.wiebe_m + 1)), abs=1e-6)


def test_engine_cycle_conserves_mass_and_energy():
    e = Engine('cp0d', p_boost=2.5e5, fuel_energy=30e3)
    r = e.run_cycle()
    s = cycle_summary(e, r)
    assert abs(s['energy_residual_rel']) < 1e-12
    assert abs(s['mass_residual_rel']) < 1e-12
