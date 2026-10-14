"""Independent checks of Project 1, written separately from the code they check.

1. Closed combustion cycle (valves shut, no wall heat): network solver vs scipy solve_ivp on the textbook
   single-zone ODE dp/dtheta = -g p/V dV/dtheta + (g-1)/V dQ/dtheta, with its own slider-crank formula.
2. Indicated work: closed integral of p dV from a recorded cylinder pressure trace (own volume formula,
   trapezoid rule) vs the solver's energy ledger.
3. Compressor work of matched points recomputed by hand from the air mass and boost in results/matching.csv.
4. Peak exhaust valve flow area by hand.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# ---------------------------------------------------------------- own engine geometry (typed again)
B, S, LAM, CR, RPM = 0.20, 0.28, 4.0, 15.0, 1000.0
Ap = math.pi / 4 * B * B
Vd = Ap * S
Vc = Vd / (CR - 1)
r = S / 2
l = LAM * r
g = 1.35


def V(th_deg):
    th = math.radians(th_deg)
    return Vc + Ap * (r + l - r * math.cos(th) - math.sqrt(l * l - (r * math.sin(th)) ** 2))


def dV(th_deg, h=1e-4):
    return (V(th_deg + h) - V(th_deg - h)) / (2 * h)       # per degree


def wiebe(th, soc=-8.0, dur=60.0, a=6.9, m=1.5):
    z = (th - soc) / dur
    if z <= 0 or z >= 1.6:
        return 0.0
    return a * (m + 1) / dur * z ** m * math.exp(-a * z ** (m + 1))


print('1. closed combustion cycle, 30 kJ fuel, from BDC (-180 deg) at 2.5 bar / 340 K')
Q = 30e3
sol = solve_ivp(lambda th, y: [-g * y[0] / V(th) * dV(th) + (g - 1) / V(th) * Q * wiebe(th)], [-180, 180],
                [2.5e5], rtol=1e-10, atol=1e-6, max_step=0.05, dense_output=True)
th = np.linspace(-180, 180, 72001)
p_ode = sol.sol(th)[0]
W_ode = np.trapezoid(p_ode, [V(t) for t in th])

from gx.engine import Cylinder, EngineSpec              # noqa: E402
from gx.gas import CV, R_GAS                            # noqa: E402
from gx.network import Network, Volume                  # noqa: E402
spec = EngineSpec()
cyl = Cylinder(spec, 0.0, Q)
cyl.htc_on = False
vol = Volume('cyl', 0.0, V_fn=cyl.V_fn, heat_fn=cyl.heat_fn)
net = Network([], [vol], [])
net.t = 540.0 / math.degrees(spec.omega)                 # -180 deg = 540 deg
V0 = cyl.V_fn(net.t)[0]
m0 = 2.5e5 * V0 / (R_GAS * 340.0)
mU = np.array([[m0], [m0 * CV * 340.0]])
W_net, p_max = 0.0, 0.0
t_end = net.t + 360.0 / math.degrees(spec.omega)
while net.t < t_end - 1e-15:
    mU, inc = net.step(min(5e-6, t_end - net.t), mU)
    W_net += inc.get('piston_work', 0.0)
    Vn = cyl.V_fn(net.t)[0]
    p_max = max(p_max, mU[0, 0] * R_GAS * (mU[1, 0] / (mU[0, 0] * CV)) / Vn)
print(f'   peak pressure: ODE {p_ode.max() / 1e5:.4f} bar, network {p_max / 1e5:.4f} bar')
print(f'   work over the closed cycle: ODE {W_ode:.2f} J, network {W_net:.2f} J '
      f'({100 * (W_net - W_ode) / W_ode:+.4f} %)')

print('2. indicated work: closed p dV integral from a recorded trace vs the energy ledger')
from gx.sim import Engine                               # noqa: E402
eng = Engine('cp0d', p_boost=2.5e5, fuel_energy=30e3)
for _ in range(4):
    eng.run_cycle()
res = eng.run_cycle(record=True)
tr = res['trace']
t = np.array([d['t'] for d in tr])
p0 = np.array([d['p_vol'][0] for d in tr])
theta = (np.degrees(spec.omega * t) - spec.offsets[0]) % 720.0
Vtr = np.array([V(x) for x in theta])
W_trace = np.sum(0.5 * (p0[1:] + p0[:-1]) * np.diff(Vtr))
W_ledger = res['ledger']['piston_work'] / spec.n_cyl
print(f'   cylinder 1: trace {W_trace:.1f} J, ledger (mean of 6 cylinders) {W_ledger:.1f} J '
      f'({100 * (W_trace - W_ledger) / W_ledger:+.3f} %)')

print('3. compressor work of the matched points, by hand')
df = pd.read_csv(ROOT / 'results' / 'matching.csv')
worst = 0.0
for _, row in df.iterrows():
    pi_c = (row.p_boost_bar * 1e5 + 3e3) / 1e5
    dh = 1005.0 * 298.15 * (pi_c ** (0.4 / 1.4) - 1) / 0.80
    W_hand = row.m_air_kg_per_cycle * dh
    worst = max(worst, abs(W_hand - row.W_comp_J) / row.W_comp_J)
print(f'   max relative difference over {len(df)} points: {worst:.2e}')
bal = (0.97 * df.W_turbine_J - df.W_comp_J) / df.W_comp_J
print(f'   turbocharger balance eta_m Wt - Wc: max |{np.abs(bal).max():.4f}| of Wc')

print('4. peak exhaust valve flow area')
n, d, L, cd = 2, 0.066, 0.019, 0.65
hand = n * cd * min(math.pi * d * L, math.pi * d * d / 4)
from gx.engine import valve_area                       # noqa: E402
code = max(valve_area(spec.exh, x) for x in np.linspace(130, 400, 2701))
print(f'   hand {hand * 1e6:.1f} mm^2, code {code * 1e6:.1f} mm^2')
