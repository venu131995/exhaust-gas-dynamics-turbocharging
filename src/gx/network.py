"""A 0D/1D gas network: 1-D pipes (finite volume), 0-D volumes and quasi-steady restrictions.

* Pipes: second-order MUSCL reconstruction (minmod) with HLLC fluxes, wall friction and wall heat transfer,
  area-change source; all pipes share one state array.
* Volumes: lumped mass and internal energy (cylinders, junctions, manifolds) with optional prescribed
  volume V(t) and heat sources.
* Links: a pipe end or a volume is joined to a volume, a fixed reservoir or a sink through a restriction of
  effective area A(t), using the compressible orifice equation. The same mass and enthalpy flow leaves one side
  and enters the other, so mass and energy are conserved exactly by construction.
Time integration: Heun (SSP-RK2) for everything at once. Every flow of mass and energy across the system boundary
is accumulated with the same Runge-Kutta weights, so the energy ledger closes to round-off.
"""
from dataclasses import dataclass, field

import math

import numpy as np

from .gas import CP, CV, GAMMA, R_GAS, hllc, mc_limiter, orifice_mass_flow


_PRC = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))
_C1 = 2 * GAMMA / (GAMMA - 1)


def _orifice(A, p0, T0, pd):
    """Scalar version of gas.orifice_mass_flow (pure math, for the boundary root solves)."""
    if A <= 0.0 or pd >= p0:
        return 0.0
    pr = max(pd / p0, _PRC)
    return A * p0 / math.sqrt(R_GAS * T0) * math.sqrt(max(_C1 * (pr ** (2 / GAMMA) - pr ** ((GAMMA + 1) / GAMMA)), 0.0))


def _illinois(f, lo, hi, flo, fhi, tol, it=60):
    """Root of f in [lo, hi] with f(lo), f(hi) of opposite sign (Illinois variant of regula falsi)."""
    side = 0
    x, st = lo, None
    for _ in range(it):
        x = (lo * fhi - hi * flo) / (fhi - flo)
        fx, st = f(x)
        if abs(fx) <= tol or abs(hi - lo) <= 1e-13 * abs(hi):
            break
        if (fx > 0) == (fhi > 0):
            hi, fhi = x, fx
            if side == -1:
                flo *= 0.5
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi *= 0.5
            side = 1
    return st


def boundary_state(rc, wc, pc, Ap, A_eff, pO, TO, sink=False, g=GAMMA, R=R_GAS):
    """Face state at a pipe end joined to a volume/reservoir through a restriction of area A_eff.

    Method of characteristics: the invariant K = w + 2c/(g-1) arriving from the interior (w = velocity towards
    the outside) is kept. Outflow: isentropic from the cell to the face, face mass flow = orifice flow from the face
    stagnation state to the outside pressure. Inflow: the incoming gas has the reservoir's entropy, so the wave
    relation is used in pressure form, w = K - a cc (p/pc)^((g-1)/(2g)); face temperature from the reservoir
    stagnation enthalpy; face mass flow = orifice flow from the reservoir to the face pressure (throttling
    included). Closed (A_eff = 0) or no driving pressure: reflecting wall. Returns (rho, w, p).
    """
    a = 2.0 / (g - 1.0)
    cc = math.sqrt(g * pc / rc)
    K = wc + a * cc
    c0f = K / a                                   # face sound speed at zero velocity
    p0f = pc * (c0f / cc) ** (2 * g / (g - 1))
    wall = (g * p0f / (c0f * c0f), 0.0, p0f)
    if A_eff <= 0.0 or (sink and p0f <= pO) or abs(p0f - pO) < 1e-12 * pO:
        return wall
    scale = A_eff * pO / math.sqrt(R * max(TO if TO else 300.0, 1.0)) + 1e-30
    if p0f > pO:                                  # outflow
        def f(cb):
            pb = pc * (cb / cc) ** (2 * g / (g - 1))
            Tb = cb * cb / (g * R)
            wb = K - a * cb
            rb = pb / (R * Tb)
            T0 = Tb + wb * wb / (2 * CP)
            p0 = pb * (T0 / Tb) ** (g / (g - 1))
            return rb * wb * Ap - _orifice(A_eff, p0, T0, pO), (rb, wb, pb)
        lo, hi = (g - 1) * K / (g + 1), c0f        # sonic face .. zero flow
        flo, slo = f(lo)
        if flo <= 0:                               # the pipe end chokes before the restriction does
            return slo
        fhi, _ = f(hi)
        return _illinois(f, lo, hi, flo, fhi, 1e-12 * scale)
    expo = (g - 1) / (2 * g)

    def gfun(pb):
        wb = K - a * cc * (pb / pc) ** expo
        Tb = max(TO - wb * wb / (2 * CP), 1.0)
        rb = pb / (R * Tb)
        return rb * (-wb) * Ap - _orifice(A_eff, pO, TO, pb), (rb, wb, pb)
    glo, _ = gfun(p0f)
    ghi, _ = gfun(pO)
    rb, wb, pb = _illinois(gfun, p0f, pO, glo, ghi, 1e-12 * scale)
    cb = math.sqrt(g * pb / rb)
    if -wb > cb:                                  # cap at sonic inflow at the pipe end
        wb = -cb
    return (rb, wb, pb)


@dataclass
class Pipe:
    name: str
    length: float
    diameter: float
    n: int
    f_darcy: float = 0.02
    T_wall: float = 600.0
    heat_transfer: bool = True
    area_fn: object = None          # optional A(x) for verification (nozzles)

    def areas(self):
        x_face = np.linspace(0, self.length, self.n + 1)
        x_c = 0.5 * (x_face[1:] + x_face[:-1])
        if self.area_fn is None:
            A = np.pi * self.diameter ** 2 / 4
            return np.full(self.n, A), np.full(self.n + 1, A)
        return self.area_fn(x_c), self.area_fn(x_face)


@dataclass
class Volume:
    name: str
    V: float                         # m^3 (used when V_fn is None)
    V_fn: object = None              # t -> (V, dV/dt)
    heat_fn: object = None           # (vol, t, p, T) -> heat added to the gas, W (combustion - wall loss)
    m: float = 0.0
    U: float = 0.0

    def init(self, p, T):
        V = self.volume(0.0)[0]
        self.m = p * V / (R_GAS * T)
        self.U = self.m * CV * T

    def volume(self, t):
        return (self.V, 0.0) if self.V_fn is None else self.V_fn(t)


@dataclass
class Link:
    """kind: 'pipe-vol' (pipe end <-> volume), 'vol-res' (volume <-> fixed reservoir),
    'pipe-sink' / 'vol-sink' (outflow only, to a back pressure), 'pipe-res' (pipe end <-> reservoir)."""
    kind: str
    a: object                         # pipe index + end ('L'/'R'), or volume index
    b: object = None                  # volume index, or reservoir (p, T), or sink back pressure
    area_fn: object = None            # t -> effective area, m^2
    name: str = ''
    tag: str = ''                     # ledger tag, e.g. 'intake', 'turbine'


class Network:
    def __init__(self, pipes, volumes, links):
        self.pipes, self.volumes, self.links = pipes, volumes, links
        sizes = [p.n for p in pipes]
        self.offset = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        N = int(self.offset[-1])
        self.N = N
        self.dx = np.concatenate([np.full(p.n, p.length / p.n) for p in pipes]) if pipes else np.zeros(0)
        Ac, Af = [], []
        for p in pipes:
            a, f = p.areas()
            Ac.append(a)
            Af.append(f)
        self.A = np.concatenate(Ac) if pipes else np.zeros(0)
        # interior faces (between cells of the same pipe)
        left, right, Aface = [], [], []
        for k, p in enumerate(pipes):
            o = self.offset[k]
            left.extend(range(o, o + p.n - 1))
            right.extend(range(o + 1, o + p.n))
            Aface.extend(Af[k][1:-1])
        self.iL, self.iR, self.Aface = np.array(left, int), np.array(right, int), np.array(Aface)
        self.Aend = {}
        for k, p in enumerate(pipes):
            self.Aend[(k, 'L')] = Af[k][0]
            self.Aend[(k, 'R')] = Af[k][-1]
        self.D = np.concatenate([np.full(p.n, p.diameter) for p in pipes]) if pipes else np.zeros(0)
        self.fD = np.concatenate([np.full(p.n, p.f_darcy) for p in pipes]) if pipes else np.zeros(0)
        self.Tw = np.concatenate([np.full(p.n, p.T_wall) for p in pipes]) if pipes else np.zeros(0)
        self.ht = np.concatenate([np.full(p.n, float(p.heat_transfer)) for p in pipes]) if pipes else np.zeros(0)
        # area-change source uses face areas of each cell
        dA = []
        for k, p in enumerate(pipes):
            dA.extend(np.diff(Af[k]))
        self.dA = np.array(dA)
        # slope limiter neighbours (one-sided at pipe ends)
        self.prev = np.arange(N) - 1
        self.next = np.arange(N) + 1
        for k, p in enumerate(pipes):
            o = self.offset[k]
            self.prev[o] = o
            self.next[o + p.n - 1] = o + p.n - 1
        # pipe ends with no link are closed walls (pressure force only)
        linked = {L.a for L in links if L.kind.startswith('pipe')}
        self.walls = []
        for k, p in enumerate(pipes):
            for end in ('L', 'R'):
                if (k, end) not in linked:
                    i = self.offset[k] if end == 'L' else self.offset[k] + p.n - 1
                    self.walls.append((i, -1.0 if end == 'L' else 1.0, self.Aend[(k, end)]))
        self.q = np.zeros((3, N))
        self.t = 0.0
        self.ledger = {}

    # ------------------------------------------------------------------ setup
    def init_pipe(self, k, p, T, u=0.0):
        o, n = self.offset[k], self.pipes[k].n
        rho = p / (R_GAS * T)
        A = self.A[o:o + n]
        self.q[:, o:o + n] = np.array([rho * A, rho * u * A, (p / (GAMMA - 1) + 0.5 * rho * u * u) * A])

    # ------------------------------------------------------------------ physics
    def _prim(self, q):
        rho = q[0] / self.A
        u = q[1] / q[0]
        p = (GAMMA - 1) * (q[2] / self.A - 0.5 * rho * u * u)
        return rho, u, p

    def rates(self, q, mU, t):
        """Time derivatives of pipe contents (per unit length), volume (m, U), and ledger rates."""
        N = self.N
        dq = np.zeros_like(q)
        led = {}
        if N:
            rho, u, p = self._prim(q)
            if np.any(p <= 0) or np.any(rho <= 0):
                raise FloatingPointError('negative pressure or density in a pipe')
            W = np.array([rho, u, p])
            slope = mc_limiter(W - W[:, self.prev], W[:, self.next] - W)
            WL = W[:, self.iL] + 0.5 * slope[:, self.iL]
            WR = W[:, self.iR] - 0.5 * slope[:, self.iR]
            # fall back to first order where reconstruction is non-physical
            bad = (WL[0] <= 0) | (WL[2] <= 0) | (WR[0] <= 0) | (WR[2] <= 0)
            WL[:, bad] = W[:, self.iL[bad]]
            WR[:, bad] = W[:, self.iR[bad]]
            F = np.array(hllc(WL[0], WL[1], WL[2], WR[0], WR[1], WR[2])) * self.Aface
            np.add.at(dq, (slice(None), self.iL), -F)
            np.add.at(dq, (slice(None), self.iR), F)
            # sources per unit length (dq holds fluxes; divide by dx at the end)
            src_mom = p * self.dA                                        # p dA (integrated over the cell)
            tau = self.fD / 8.0 * rho * u * np.abs(u)                    # wall shear stress
            fric = -tau * np.pi * self.D * self.dx                        # force on gas in the cell
            h = self.ht * self.fD / 8.0 * rho * np.abs(u) * CP * 0.71 ** (-2.0 / 3.0)   # Reynolds-Colburn analogy
            Tgas = p / (rho * R_GAS)
            qwall = h * np.pi * self.D * self.dx * (self.Tw - Tgas)       # W into the gas
            dq[1] += src_mom + fric
            dq[2] += qwall
            wall_force = 0.0
            for i, n_out, Aw in self.walls:
                dq[1, i] += -n_out * p[i] * Aw
                wall_force += -n_out * p[i] * Aw
            led['wall_force'] = wall_force
            led['friction_force'] = float(fric.sum())
            led['area_force'] = float(src_mom.sum())
            led['pipe_wall_heat'] = float(qwall.sum())
        else:
            rho = u = p = np.zeros(0)
        dmU = np.zeros_like(mU)
        led.setdefault('pipe_wall_heat', 0.0)
        # volume states
        Vs, dVs, pv, Tv = [], [], [], []
        for j, vol in enumerate(self.volumes):
            V, dV = vol.volume(t)
            m, U = mU[0, j], mU[1, j]
            T = U / (m * CV)
            Vs.append(V)
            dVs.append(dV)
            Tv.append(T)
            pv.append(m * R_GAS * T / V)
        pv, Tv = np.array(pv), np.array(Tv)
        for j, vol in enumerate(self.volumes):
            dmU[1, j] -= pv[j] * dVs[j]
            led['piston_work'] = led.get('piston_work', 0.0) + pv[j] * dVs[j]
            ph = getattr(vol, 'phase_fn', None)
            if ph is not None and ph(t) == 'gas exchange':
                led['pumping_work'] = led.get('pumping_work', 0.0) + pv[j] * dVs[j]
            if vol.heat_fn is not None:
                qc, qw = vol.heat_fn(vol, t, pv[j], Tv[j], Vs[j])
                dmU[1, j] += qc - qw
                led['fuel_heat'] = led.get('fuel_heat', 0.0) + qc
                led['cyl_wall_heat'] = led.get('cyl_wall_heat', 0.0) + qw
        # links
        for li, L in enumerate(self.links):
            A_eff = L.area_fn(t) if L.area_fn is not None else None
            if L.kind in ('pipe-vol', 'pipe-res', 'pipe-sink'):
                k, end = L.a
                i = self.offset[k] if end == 'L' else self.offset[k] + self.pipes[k].n - 1
                Ap = self.Aend[(k, end)]
                n_out = -1.0 if end == 'L' else 1.0           # outward normal of the pipe end
                rc, uc, pc = rho[i], u[i], p[i]
                Tc = pc / (rc * R_GAS)
                if A_eff is None:
                    A_eff = Ap
                if L.kind == 'pipe-vol':
                    pO, TO = pv[L.b], Tv[L.b]
                elif L.kind == 'pipe-res':
                    pO, TO = L.b
                else:
                    pO, TO = L.b, None
                bs = boundary_state(rc, uc * n_out, pc, Ap, A_eff, pO, TO, sink=(L.kind == 'pipe-sink'))
                rb, wb, pb = bs                                  # face state, wb = velocity towards the outside
                Eb = pb / (GAMMA - 1) + 0.5 * rb * wb * wb
                m_out = rb * wb * Ap                             # mass flow out of the pipe (negative = in)
                ub = wb * n_out
                Fm, Fp, Fe = rb * ub * Ap, (rb * ub * ub + pb) * Ap, ub * (Eb + pb) * Ap   # +x face flux
                dq[0, i] += -n_out * Fm
                dq[1, i] += -n_out * Fp
                dq[2, i] += -n_out * Fe
                e_out = n_out * Fe                               # energy flow out of the pipe (W)
                s = -1.0 if m_out >= 0 else 1.0
                mdot = abs(m_out)
                h0 = e_out / m_out if m_out != 0 else 0.0
                p0c = pb * (1 + 0.5 * (GAMMA - 1) * wb * wb * rb / (GAMMA * pb)) ** (GAMMA / (GAMMA - 1))
                T0c = h0 / CP if m_out > 0 else pb / (rb * R_GAS)
                if L.kind == 'pipe-vol':
                    dmU[0, L.b] -= s * mdot
                    dmU[1, L.b] -= s * mdot * h0
                else:
                    led[L.tag + '_mass'] = led.get(L.tag + '_mass', 0.0) + s * mdot
                    led[L.tag + '_enthalpy'] = led.get(L.tag + '_enthalpy', 0.0) + s * mdot * h0
                    if L.kind == 'pipe-sink':
                        led[f'{L.tag}_p0T0_{li}'] = (p0c, T0c, mdot)
            elif L.kind == 'vol-vol':   # two volumes through a restriction (e.g. valve straight into a plenum)
                i, j = L.a, L.b
                up, dn = (i, j) if pv[i] >= pv[j] else (j, i)
                mdot = float(orifice_mass_flow(A_eff, pv[up], Tv[up], pv[dn]))
                h0 = CP * Tv[up]
                dmU[0, up] -= mdot
                dmU[1, up] -= mdot * h0
                dmU[0, dn] += mdot
                dmU[1, dn] += mdot * h0
            else:   # volume <-> reservoir / sink
                j = L.a
                if L.kind == 'vol-res':
                    pO, TO = L.b
                else:
                    pO, TO = L.b, None
                if pv[j] >= pO:
                    mdot = float(orifice_mass_flow(A_eff, pv[j], Tv[j], pO))
                    h0, s = CP * Tv[j], -1.0
                    if L.kind == 'vol-sink':
                        led[f'{L.tag}_p0T0_{li}'] = (pv[j], Tv[j], mdot)
                elif L.kind == 'vol-sink':
                    mdot, h0, s = 0.0, 0.0, -1.0
                    led[f'{L.tag}_p0T0_{li}'] = (pv[j], Tv[j], 0.0)
                else:
                    mdot = float(orifice_mass_flow(A_eff, pO, TO, pv[j]))
                    h0, s = CP * TO, 1.0
                dmU[0, j] += s * mdot
                dmU[1, j] += s * mdot * h0
                led[L.tag + '_mass'] = led.get(L.tag + '_mass', 0.0) + s * mdot
                led[L.tag + '_enthalpy'] = led.get(L.tag + '_enthalpy', 0.0) + s * mdot * h0
        if N:
            dq /= self.dx
        return dq, dmU, led

    def max_dt(self, cfl=0.8):
        dts = []
        if self.N:
            rho, u, p = self._prim(self.q)
            c = np.sqrt(GAMMA * p / rho)
            dts.append(cfl * np.min(self.dx / (np.abs(u) + c)))
        return min(dts) if dts else 1e-5

    def step(self, dt, mU):
        """One Heun step; returns new (q, mU) and the RK-weighted ledger increments (J or kg)."""
        q0, m0, t = self.q, mU, self.t
        d1, e1, l1 = self.rates(q0, m0, t)
        q1, m1 = q0 + dt * d1, m0 + dt * e1
        d2, e2, l2 = self.rates(q1, m1, t + dt)
        self.q = 0.5 * (q0 + q1 + dt * d2)
        mU_new = 0.5 * (m0 + m1 + dt * e2)
        inc = {}
        for k in set(l1) | set(l2):
            if '_p0T0' in k:
                inc[k] = l2.get(k, l1.get(k))
                continue
            inc[k] = 0.5 * dt * (l1.get(k, 0.0) + l2.get(k, 0.0))
        self.t = t + dt
        return mU_new, inc

    # ------------------------------------------------------------------ diagnostics
    def contents(self, mU):
        """Total mass and total energy (internal + kinetic) stored in pipes and volumes."""
        M = float((self.q[0] * self.dx).sum()) + float(mU[0].sum())
        E = float((self.q[2] * self.dx).sum()) + float(mU[1].sum())
        return M, E

    def pipe_state(self, k):
        o, n = self.offset[k], self.pipes[k].n
        rho, u, p = self._prim(self.q)
        return rho[o:o + n], u[o:o + n], p[o:o + n]
