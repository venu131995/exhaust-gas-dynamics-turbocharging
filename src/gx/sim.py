"""Engine + exhaust system + turbocharger: build networks, run cycles, match the turbocharger.

Exhaust layouts (same engine, same turbine swallowing area):
  'pulse'  : two groups of 3 cylinders, 0.6 m branches -> small junction -> 1.2 m collector -> twin-entry turbine
  'cp'     : six short branches -> one large manifold (constant pressure) -> pipe -> single-entry turbine
  'pulse0d', 'cp0d' : the same layouts with every pipe lumped into its downstream volume (filling-and-emptying)
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .engine import Cylinder, EngineSpec, crank_angle, valve_area
from .gas import CP, GAMMA, R_GAS
from .network import Link, Network, Pipe, Volume

P_AMB, T_AMB = 1.0e5, 298.15
GA, CPA = 1.4, 1005.0            # air on the compressor side


@dataclass(frozen=True)
class Turbo:
    A_turbine: float = 3.0e-3     # m^2 equivalent nozzle area (all entries together)
    p_back: float = 1.05e5        # turbine outlet pressure
    eta_c: float = 0.80
    eta_t_max: float = 0.80
    uc_opt: float = 0.70          # optimum blade-speed ratio
    work_coeff: float = 0.70      # compressor work = work_coeff * U_c^2
    dia_ratio: float = 0.90       # turbine / compressor tip diameter
    eta_mech: float = 0.97
    T_boost: float = 318.15       # after the charge-air cooler
    dp_cooler: float = 3.0e3      # Pa, charge-air cooler pressure drop

    def with_(self, **kw):
        return replace(self, **kw)

    def compressor(self, p_boost):
        pi_c = (p_boost + self.dp_cooler) / P_AMB
        dh_s = CPA * T_AMB * (pi_c ** ((GA - 1) / GA) - 1)
        dh = dh_s / self.eta_c
        U_c = math.sqrt(max(dh, 0.0) / self.work_coeff)
        return pi_c, dh, U_c

    def eta_turbine(self, uc):
        x = uc / self.uc_opt
        return np.clip(self.eta_t_max * (2 * x - x * x), 0.0, None)


@dataclass(frozen=True)
class Layout:
    branch_len: float = 0.6
    branch_dia: float = 0.08
    coll_len: float = 1.2
    coll_dia: float = 0.11
    junction_vol: float = 3.0e-3
    cp_branch_len: float = 0.3
    cp_manifold_vol: float = 0.08
    cp_pipe_len: float = 0.6
    cp_pipe_dia: float = 0.16
    dx: float = 0.02
    T_wall: float = 600.0

    def with_(self, **kw):
        return replace(self, **kw)


def build(system, spec, turbo, layout, p_boost, fuel_energy):
    cyls = [Cylinder(spec, off, fuel_energy) for off in spec.offsets]
    vols = []
    for j, c in enumerate(cyls):
        v = Volume(f'cyl{j}', 0.0, V_fn=c.V_fn, heat_fn=c.heat_fn)
        v.phase_fn = c.phase_fn
        vols.append(v)
    pipes, links = [], []
    res = (p_boost, turbo.T_boost)
    for j, c in enumerate(cyls):
        links.append(Link('vol-res', j, res, area_fn=lambda t, c=c: valve_area(spec.inl, crank_angle(spec, t, c.offset)),
                          tag='intake'))
    n = lambda L: max(4, int(round(L / layout.dx)))
    ex_area = lambda c: (lambda t: valve_area(spec.exh, crank_angle(spec, t, c.offset)))
    groups = [[0, 1, 2], [3, 4, 5]]
    if system in ('pulse', 'pulse0d'):
        for gi, g in enumerate(groups):
            if system == 'pulse':
                jv = len(vols)
                vols.append(Volume(f'junction{gi}', layout.junction_vol))
                for j in g:
                    k = len(pipes)
                    pipes.append(Pipe(f'branch{j}', layout.branch_len, layout.branch_dia, n(layout.branch_len),
                                      T_wall=layout.T_wall))
                    links.append(Link('pipe-vol', (k, 'L'), j, area_fn=ex_area(cyls[j])))
                    links.append(Link('pipe-vol', (k, 'R'), jv))
                k = len(pipes)
                pipes.append(Pipe(f'collector{gi}', layout.coll_len, layout.coll_dia, n(layout.coll_len),
                                  T_wall=layout.T_wall))
                links.append(Link('pipe-vol', (k, 'L'), jv))
                links.append(Link('pipe-sink', (k, 'R'), turbo.p_back, area_fn=lambda t: turbo.A_turbine / 2,
                                  tag='turbine'))
            else:
                V = (layout.junction_vol + 3 * layout.branch_len * math.pi / 4 * layout.branch_dia ** 2
                     + layout.coll_len * math.pi / 4 * layout.coll_dia ** 2)
                jv = len(vols)
                vols.append(Volume(f'group{gi}', V))
                for j in g:
                    links.append(Link('vol-vol', j, jv, area_fn=ex_area(cyls[j])))
                links.append(Link('vol-sink', jv, turbo.p_back, area_fn=lambda t: turbo.A_turbine / 2, tag='turbine'))
    elif system in ('cp', 'cp0d'):
        mv = len(vols)
        if system == 'cp':
            vols.append(Volume('manifold', layout.cp_manifold_vol))
            for j in range(spec.n_cyl):
                k = len(pipes)
                pipes.append(Pipe(f'branch{j}', layout.cp_branch_len, layout.branch_dia, n(layout.cp_branch_len),
                                  T_wall=layout.T_wall))
                links.append(Link('pipe-vol', (k, 'L'), j, area_fn=ex_area(cyls[j])))
                links.append(Link('pipe-vol', (k, 'R'), mv))
            k = len(pipes)
            pipes.append(Pipe('turbine pipe', layout.cp_pipe_len, layout.cp_pipe_dia, n(layout.cp_pipe_len),
                              T_wall=layout.T_wall))
            links.append(Link('pipe-vol', (k, 'L'), mv))
            links.append(Link('pipe-sink', (k, 'R'), turbo.p_back, area_fn=lambda t: turbo.A_turbine,
                              tag='turbine'))
        else:
            V = (layout.cp_manifold_vol + 6 * layout.cp_branch_len * math.pi / 4 * layout.branch_dia ** 2
                 + layout.cp_pipe_len * math.pi / 4 * layout.cp_pipe_dia ** 2)
            vols.append(Volume('manifold', V))
            for j in range(spec.n_cyl):
                links.append(Link('vol-vol', j, mv, area_fn=ex_area(cyls[j])))
            links.append(Link('vol-sink', mv, turbo.p_back, area_fn=lambda t: turbo.A_turbine, tag='turbine'))
    else:
        raise ValueError(system)
    net = Network(pipes, vols, links)
    return net, cyls


class Engine:
    """Runs one exhaust system at fixed boost and fuelling; collects cycle results and the energy ledger."""

    def __init__(self, system, spec=EngineSpec(), turbo=Turbo(), layout=Layout(), p_boost=2.5e5,
                 fuel_energy=30e3):
        self.system, self.spec, self.turbo, self.layout = system, spec, turbo, layout
        self.p_boost, self.fuel_energy = p_boost, fuel_energy
        self.net, self.cyls = build(system, spec, turbo, layout, p_boost, fuel_energy)
        for v in self.net.volumes:
            v.init(p_boost, 700.0)
        for k in range(len(self.net.pipes)):
            self.net.init_pipe(k, 1.2 * turbo.p_back, 700.0)
        self.mU = np.array([[v.m for v in self.net.volumes], [v.U for v in self.net.volumes]])
        self.dt_max = 2e-5

    def run_cycle(self, record=False):
        sp, tb, net = self.spec, self.turbo, self.net
        t0 = net.t
        t_end = t0 + sp.t_cycle
        L = {}
        M0, E0 = net.contents(self.mU)
        W_t = W_t_avail = 0.0
        pi_c, dh_c, U_c = tb.compressor(self.p_boost)
        U_t = tb.dia_ratio * U_c
        trace = []
        prev_th = [crank_angle(sp, t0, c.offset) for c in self.cyls]
        while net.t < t_end - 1e-12:
            dt = min(net.max_dt(0.8) if net.N else self.dt_max, self.dt_max, t_end - net.t)
            self.mU, inc = net.step(dt, self.mU)
            for k, v in inc.items():
                if '_p0T0' in k:
                    continue
                L[k] = L.get(k, 0.0) + v
            # turbine power from the flows leaving at the turbine entries (same RK weights not needed:
            # post-processing only)
            p0T0 = [v for k, v in inc.items() if k.startswith('turbine_p0T0')]
            for (p0, T0, mdot) in p0T0:
                if mdot > 0 and p0 > tb.p_back:
                    dh_s = CP * T0 * (1 - (tb.p_back / p0) ** ((GAMMA - 1) / GAMMA))
                    c_s = math.sqrt(2 * dh_s)
                    W_t_avail += mdot * dh_s * dt
                    W_t += float(tb.eta_turbine(U_t / c_s)) * mdot * dh_s * dt
            # capture IVC states for Woschni
            for j, c in enumerate(self.cyls):
                th = crank_angle(sp, net.t, c.offset)
                if prev_th[j] < (sp.inl[5] % 720) <= th:
                    m, U = self.mU[:, j]
                    V = c.V_fn(net.t)[0]
                    T = U / (m * (CP - R_GAS))
                    c.ivc_state = (m * R_GAS * T / V, T, V)
                prev_th[j] = th
            if record:
                trace.append(self._snapshot())
        M1, E1 = net.contents(self.mU)
        m_air = L.get('intake_mass', 0.0)
        out = dict(ledger=L, dM=M1 - M0, dE=E1 - E0, W_turbine=W_t, W_turbine_avail=W_t_avail,
                   m_air=m_air, W_comp=m_air * dh_c, pi_c=pi_c, U_c=U_c)
        out['energy_residual'] = (E1 - E0) - (L.get('fuel_heat', 0) - L.get('cyl_wall_heat', 0)
                                            - L.get('piston_work', 0) + L.get('pipe_wall_heat', 0)
                                            + L.get('intake_enthalpy', 0) + L.get('turbine_enthalpy', 0))
        out['mass_residual'] = (M1 - M0) - (L.get('intake_mass', 0) + L.get('turbine_mass', 0))
        if record:
            out['trace'] = trace
        return out

    def _snapshot(self):
        net = self.net
        d = dict(t=net.t)
        V = [v.volume(net.t)[0] for v in net.volumes]
        T = self.mU[1] / (self.mU[0] * (CP - R_GAS))
        d['p_vol'] = self.mU[0] * R_GAS * T / np.array(V)
        d['T_vol'] = T
        if net.N:
            rho, u, p = net._prim(net.q)
            ends = [net.offset[k + 1] - 1 for k in range(len(net.pipes))]
            d['p_pipe_end'] = p[ends]
            d['T_pipe_end'] = (p / (rho * R_GAS))[ends]
            d['u_pipe_end'] = u[ends]
        return d

    def converge(self, n_max=12, tol=2e-3):
        """Run cycles until indicated work and air mass change by less than tol (relative)."""
        prev = None
        for i in range(n_max):
            r = self.run_cycle()
            key = np.array([r['ledger'].get('piston_work', 0.0), r['m_air'], r['W_turbine']])
            if prev is not None and np.all(np.abs(key - prev) <= tol * np.abs(key) + 1e-12):
                r['cycles'] = i + 1
                return r
            prev = key
        r['cycles'] = n_max
        return r


def cycle_summary(eng, r):
    sp = eng.spec
    L = r['ledger']
    W_i = L.get('piston_work', 0.0)                   # net work done by the gas on the pistons, per cycle
    q_fuel = L.get('fuel_heat', 0.0)
    vd = sp.v_disp * sp.n_cyl
    m_air = r['m_air']
    lam = m_air / (sp.n_cyl * eng.fuel_energy / 42.7e6 * 14.5)
    pmep = L.get('pumping_work', 0.0) / vd / 1e5
    return dict(system=eng.system, p_boost_bar=eng.p_boost / 1e5, imep_bar=W_i / vd / 1e5, pmep_bar=pmep,
                eta_indicated=W_i / q_fuel if q_fuel else 0.0, lambda_=lam, m_air_kg_per_cycle=m_air,
                W_turbine_J=r['W_turbine'], W_turbine_avail_J=r['W_turbine_avail'], W_comp_J=r['W_comp'],
                turbine_eff_mean=r['W_turbine'] / r['W_turbine_avail'] if r['W_turbine_avail'] else 0.0,
                power_kw=W_i / sp.t_cycle / 1e3, energy_residual_rel=r['energy_residual'] / q_fuel if q_fuel else 0,
                mass_residual_rel=r['mass_residual'] / m_air if m_air else 0, cycles=r.get('cycles'))


def match(system, fuel_energy, spec=EngineSpec(), turbo=Turbo(), layout=Layout(), p0=2.0e5, tol=0.003,
          max_iter=12, verbose=False):
    """Find the boost pressure at which the turbine drives the compressor (eta_mech W_t = W_c)."""
    def resid(pb):
        eng = Engine(system, spec, turbo, layout, pb, fuel_energy)
        r = eng.converge()
        s = cycle_summary(eng, r)
        g = turbo.eta_mech * r['W_turbine'] - r['W_comp']
        if verbose:
            print(f'   {system} pb {pb / 1e5:.3f} bar: Wt*eta {turbo.eta_mech * r["W_turbine"]:.0f} J, '
                  f'Wc {r["W_comp"]:.0f} J, cycles {r["cycles"]}', flush=True)
        return g / max(r['W_comp'], 1.0), eng, r, s

    a, b = p0, p0 * 1.25
    ga, *_ = resid(a)
    gb, eng, r, s = resid(b)
    for _ in range(max_iter):
        if abs(gb) < tol:
            break
        c = b - gb * (b - a) / (gb - ga)
        c = float(np.clip(c, 1.05e5, 6e5))
        a, ga = b, gb
        b = c
        gb, eng, r, s = resid(b)
    s['match_residual'] = gb
    return eng, r, s
