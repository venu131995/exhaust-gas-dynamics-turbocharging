"""Verification of the 0D/1D solver. Writes results/verification.json and figures/fig1_verification.png."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gx import riemann                                   # noqa: E402
from gx.engine import Cylinder, EngineSpec, cyl_volume    # noqa: E402
from gx.gas import CV, GAMMA, R_GAS                       # noqa: E402
from gx.network import Link, Network, Pipe, Volume        # noqa: E402
from gx.sim import Engine, Layout, cycle_summary          # noqa: E402

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = {}


def run_to(net, t_end, mU=None, cfl=0.8):
    mU = np.zeros((2, 0)) if mU is None else mU
    led = {}
    while net.t < t_end - 1e-15:
        dt = min(net.max_dt(cfl), t_end - net.t)
        mU, inc = net.step(dt, mU)
        for k, v in inc.items():
            if '_p0T0' not in k:
                led[k] = led.get(k, 0.0) + v
    return mU, led


# ---------------------------------------------------------------- 1. shock tube vs exact Riemann solution
def shock_tube():
    res, sol = {}, None
    for n in (100, 200, 400, 800):
        net = Network([Pipe('t', 1.0, 0.1, n, f_darcy=0.0, heat_transfer=False)], [], [])
        x = (np.arange(n) + 0.5) / n
        rho = np.where(x < 0.5, 1.0, 0.125)
        p = np.where(x < 0.5, 1.0, 0.1)
        net.q = np.array([rho * net.A, 0 * net.A, p / (GAMMA - 1) * net.A])
        M0 = (net.q[0] * net.dx).sum()
        E0 = (net.q[2] * net.dx).sum()
        run_to(net, 0.2)
        r, u, pp = net.pipe_state(0)
        ex = riemann.sample(x, 0.2, 0.5, (1, 0, 1), (0.125, 0, 0.1), GAMMA)
        res[n] = dict(L1_rho=float(np.abs(r - ex[0]).mean()), L1_u=float(np.abs(u - ex[1]).mean()),
                      L1_p=float(np.abs(pp - ex[2]).mean()),
                      mass_change_rel=float(abs((net.q[0] * net.dx).sum() - M0) / M0),
                      energy_change_rel=float(abs((net.q[2] * net.dx).sum() - E0) / E0))
        if n == 400:
            sol = (x, r, u, pp, ex)
    ns = np.array(list(res))
    e = np.array([res[n]['L1_rho'] for n in ns])
    order = float(-np.polyfit(np.log(ns), np.log(e), 1)[0])
    return dict(errors={str(k): v for k, v in res.items()}, order_rho=order), sol


# ---------------------------------------------------------------- 2. smooth pulse: observed order (self-convergence)
def smooth_order():
    sols = {}
    for n in (100, 200, 400, 800):
        net = Network([Pipe('t', 1.0, 0.1, n, f_darcy=0.0, heat_transfer=False)], [], [])
        x = (np.arange(n) + 0.5) / n
        p = 1.0 + 0.02 * np.exp(-((x - 0.5) / 0.06) ** 2)
        rho = p ** (1 / GAMMA)
        net.q = np.array([rho * net.A, 0 * net.A, p / (GAMMA - 1) * net.A])
        run_to(net, 0.25)
        sols[n] = net.pipe_state(0)[2]
    coarse = lambda f, m: f.reshape(-1, m).mean(axis=1)
    e1 = np.abs(sols[100] - coarse(sols[800], 8)).mean()
    e2 = np.abs(sols[200] - coarse(sols[800], 4)).mean()
    e3 = np.abs(sols[400] - coarse(sols[800], 2)).mean()
    # Richardson: errors against the finest solution; order from the first two differences
    d12 = np.abs(sols[100] - coarse(sols[200], 2)).mean()
    d23 = np.abs(coarse(sols[200], 2) - coarse(sols[400], 4)).mean()
    d34 = np.abs(coarse(sols[400], 4) - coarse(sols[800], 8)).mean()
    return dict(diff_100_200=float(d12), diff_200_400=float(d23), diff_400_800=float(d34),
                order=float(math.log2(d23 / d34)), order_coarse=float(math.log2(d12 / d23)))


# ---------------------------------------------------------------- 3. steady subsonic nozzle vs area-Mach relation
def nozzle():
    A0 = math.pi / 4 * 0.1 ** 2
    area = lambda x: A0 * (1.0 - 0.35 * np.sin(np.pi * x) ** 2)
    out = {}
    for n in (50, 100, 200, 400):
        p0, T0, pe = 2.0e5, 600.0, 1.92e5     # subsonic everywhere (choking needs pe/p0 < 0.89)
        net = Network([Pipe('nozzle', 1.0, 0.1, n, f_darcy=0.0, heat_transfer=False, area_fn=area)], [],
                      [Link('pipe-res', (0, 'L'), (p0, T0), tag='in'), Link('pipe-res', (0, 'R'), (pe, T0), tag='out')])
        net.init_pipe(0, pe, T0)
        run_to(net, 0.3)
        rho, u, p = net.pipe_state(0)
        x = (np.arange(n) + 0.5) / n
        mdot = rho * u * net.A
        md = mdot.mean()
        g = GAMMA
        M_num = u / np.sqrt(g * p / rho)

        def mach_from_area(A):
            lo, hi = 1e-6, 1.0
            for _ in range(80):
                M = 0.5 * (lo + hi)
                f = A * p0 / math.sqrt(R_GAS * T0) * math.sqrt(g) * M * (1 + (g - 1) / 2 * M * M) ** (-(g + 1) / (2 * (g - 1)))
                lo, hi = (M, hi) if f < md else (lo, M)
            return 0.5 * (lo + hi)
        M_ex = np.array([mach_from_area(a) for a in net.A])
        out[n] = dict(max_mach_err=float(np.abs(M_num - M_ex).max()), mdot_spread_rel=float((mdot.max() - mdot.min()) / md),
                      throat_mach=float(M_num.max()))
        if n == 400:
            prof = (x, M_num, M_ex)
    ns = np.array(list(out))
    e = np.array([out[n]['max_mach_err'] for n in ns])
    return dict(errors={str(k): v for k, v in out.items()}, order=float(-np.polyfit(np.log(ns), np.log(e), 1)[0])), prof


# ---------------------------------------------------------------- 4. sealed motored cylinder: p V^gamma = const
def motored():
    spec = EngineSpec()
    out = {}
    for dt in (4e-5, 2e-5, 1e-5):
        c = Cylinder(spec, 0.0, 0.0)
        c.htc_on = False
        vol = Volume('cyl', 0.0, V_fn=lambda t, c=c: c.V_fn(t), heat_fn=c.heat_fn)
        net = Network([], [vol], [])
        # start at BDC (180 deg) at 2.5 bar, 320 K, compress through TDC and back
        net.t = 180.0 / math.degrees(spec.omega)
        V0 = c.V_fn(net.t)[0]
        m = 2.5e5 * V0 / (R_GAS * 320.0)
        mU = np.array([[m], [m * CV * 320.0]])
        inv0 = 2.5e5 * V0 ** GAMMA
        dev, W = 0.0, 0.0
        t_end = net.t + 360.0 / math.degrees(spec.omega)
        while net.t < t_end - 1e-15:
            h = min(dt, t_end - net.t)
            mU, inc = net.step(h, mU)
            W += inc.get('piston_work', 0.0)
            V = c.V_fn(net.t)[0]
            T = mU[1, 0] / (mU[0, 0] * CV)
            pc = mU[0, 0] * R_GAS * T / V
            dev = max(dev, abs(pc * V ** GAMMA - inv0) / inv0)
        U_change = mU[1, 0] - m * CV * 320.0
        out[str(dt)] = dict(max_dev_pVgamma=dev, first_law_resid_rel=abs(U_change + W) / (m * CV * 320.0))
    d = [out[k]['max_dev_pVgamma'] for k in out]
    return dict(runs=out, order=float(math.log2(d[1] / d[2])))


# ---------------------------------------------------------------- 5. momentum budget of a closed pipe with friction
def momentum():
    net = Network([Pipe('t', 2.0, 0.08, 200, f_darcy=0.03, heat_transfer=False)], [], [])
    x = (np.arange(200) + 0.5) / 200 * 2.0
    p = np.where(x < 0.7, 3e5, 1e5)
    T = 600.0
    rho = p / (R_GAS * T)
    net.q = np.array([rho * net.A, 0 * net.A, p / (GAMMA - 1) * net.A])
    P0 = (net.q[1] * net.dx).sum()
    M0, E0 = (net.q[0] * net.dx).sum(), (net.q[2] * net.dx).sum()
    _, led = run_to(net, 0.01)
    P1 = (net.q[1] * net.dx).sum()
    forces = led['wall_force'] + led['friction_force'] + led['area_force']
    return dict(momentum_change=float(P1 - P0), impulse_of_forces=float(forces),
                resid_rel=float(abs(P1 - P0 - forces) / max(abs(forces), 1e-30)),
                friction_impulse=float(led['friction_force']),
                mass_change_rel=float(abs((net.q[0] * net.dx).sum() - M0) / M0),
                energy_change_rel=float(abs((net.q[2] * net.dx).sum() - E0) / E0))


# ---------------------------------------------------------------- 6. engine: conservation and grid convergence
def engine_checks():
    out = {}
    for dx in (0.04, 0.02, 0.01):
        e = Engine('pulse', layout=Layout(dx=dx), p_boost=2.5e5, fuel_energy=30e3)
        r = e.converge(n_max=10)
        s = cycle_summary(e, r)
        out[str(dx)] = dict(W_turbine_avail=r['W_turbine_avail'], imep=s['imep_bar'], m_air=r['m_air'],
                            energy_residual_rel=abs(s['energy_residual_rel']), mass_residual_rel=abs(s['mass_residual_rel']),
                            cycles=r['cycles'])
        print('   engine dx', dx, out[str(dx)], flush=True)
    return out


def main():
    print('1 shock tube', flush=True)
    OUT['shock_tube'], sod = shock_tube()
    print(OUT['shock_tube']['order_rho'], flush=True)
    print('2 smooth order', flush=True)
    OUT['smooth_pulse'] = smooth_order()
    print(OUT['smooth_pulse'], flush=True)
    print('3 nozzle', flush=True)
    OUT['nozzle'], noz = nozzle()
    print(OUT['nozzle'], flush=True)
    print('4 motored cylinder', flush=True)
    OUT['motored_cylinder'] = motored()
    print(OUT['motored_cylinder'], flush=True)
    print('5 momentum budget', flush=True)
    OUT['momentum_budget'] = momentum()
    print(OUT['momentum_budget'], flush=True)
    print('6 engine', flush=True)
    OUT['engine_grid'] = engine_checks()
    (ROOT / 'results' / 'verification.json').write_text(json.dumps(OUT, indent=2))

    x, r, u, pp, ex = sod
    xn, Mn, Me = noz
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    ax[0].plot(x, ex[0], 'k', lw=1, label='exact')
    ax[0].plot(x, r, 'C1.', ms=2, label='finite volume, 400 cells')
    ax[0].set(xlabel='x', ylabel='density', title='shock tube, t = 0.2')
    ax[0].legend(frameon=False, fontsize=7)
    ns = [int(k) for k in OUT['shock_tube']['errors']]
    ax[1].loglog(ns, [OUT['shock_tube']['errors'][str(n)]['L1_rho'] for n in ns], 'o-', label='shock tube (L1 density)')
    ax[1].loglog([200, 400, 800], [OUT['smooth_pulse'][k] for k in ('diff_100_200', 'diff_200_400', 'diff_400_800')],
                 's-', label='smooth pulse (successive differences)')
    ax[1].loglog(ns, 0.8 / np.array(ns), 'k:', lw=0.8, label='slope 1')
    ax[1].loglog(ns, 30 / np.array(ns) ** 2, 'k--', lw=0.8, label='slope 2')
    ax[1].set(xlabel='cells', ylabel='error', title='convergence')
    ax[1].legend(frameon=False, fontsize=6)
    ax[2].plot(xn, Me, 'k', lw=1, label='area-Mach relation')
    ax[2].plot(xn, Mn, 'C1.', ms=3, label='finite volume, 200 cells')
    ax[2].set(xlabel='x (m)', ylabel='Mach number', title='steady nozzle')
    ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / 'fig1_verification.png')


if __name__ == '__main__':
    main()
