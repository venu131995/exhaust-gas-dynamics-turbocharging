"""Matched operating line for pulse vs constant-pressure turbocharging, 1-D vs 0-D, and a manifold-length sweep.

Writes results/matching.csv, results/length_sweep.csv, results/summary.json and figures/fig2-fig5.
Runs the operating points in parallel processes.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gx.sim import Engine, EngineSpec, Layout, Turbo, cycle_summary, match   # noqa: E402

RES, FIG = ROOT / 'results', ROOT / 'figures'
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
FULL_FUEL = 30e3                   # J of fuel energy per cylinder per cycle at 100 % load
LOADS = [0.25, 0.5, 0.75, 1.0]
SYSTEMS = ['pulse', 'cp', 'pulse0d', 'cp0d']
COLOURS = {'pulse': 'C3', 'cp': 'C0', 'pulse0d': 'C3', 'cp0d': 'C0'}


def match_point(args):
    system, load = args
    t0 = time.time()
    eng, r, s = match(system, FULL_FUEL * load, p0=1.3e5 + 1.6e5 * load)
    # one more recorded cycle at the matched point for the traces
    tr = eng.run_cycle(record=True)['trace']
    t = np.array([d['t'] for d in tr])
    trace = dict(t=(t - t[0]).tolist(), p_cyl0=[float(d['p_vol'][0]) for d in tr])
    if 'p_pipe_end' in tr[0]:
        trace['p_turb_in'] = [float(d['p_pipe_end'][-1]) for d in tr]
    else:
        trace['p_turb_in'] = [float(d['p_vol'][-1]) for d in tr]
    s.update(load=load, runtime_s=time.time() - t0)
    return s, trace


def sweep_point(args):
    L_c, p_boost = args
    eng = Engine('pulse', layout=Layout(coll_len=L_c), p_boost=p_boost, fuel_energy=FULL_FUEL)
    r = eng.converge()
    s = cycle_summary(eng, r)
    s['coll_len'] = L_c
    return s


def main():
    t0 = time.time()
    jobs = [(s, l) for s in SYSTEMS for l in LOADS]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        out = list(ex.map(match_point, jobs))
    rows = [o[0] for o in out]
    traces = {(o[0]['system'], o[0]['load']): o[1] for o in out}
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'matching.csv', index=False, float_format='%.6g')
    print(df[['system', 'load', 'p_boost_bar', 'imep_bar', 'pmep_bar', 'eta_indicated', 'lambda_',
              'W_turbine_avail_J', 'turbine_eff_mean', 'match_residual', 'energy_residual_rel']].round(4).to_string(), flush=True)

    pb_full = float(df[(df.system == 'pulse') & (df.load == 1.0)].p_boost_bar.iloc[0]) * 1e5
    lens = [0.4, 0.8, 1.2, 2.0, 3.0]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        sweep = pd.DataFrame(list(ex.map(sweep_point, [(L, pb_full) for L in lens])))
    sweep.to_csv(RES / 'length_sweep.csv', index=False, float_format='%.6g')
    print(sweep[['coll_len', 'W_turbine_avail_J', 'm_air_kg_per_cycle', 'imep_bar', 'pmep_bar']].round(4).to_string())

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for j, load in enumerate([1.0, 0.5]):
        for s, ls in [('pulse', '-'), ('pulse0d', '--'), ('cp', '-'), ('cp0d', '--')]:
            tr = traces[(s, load)]
            ca = np.array(tr['t']) * EngineSpec().omega * 180 / np.pi
            ax[j].plot(ca, np.array(tr['p_turb_in']) / 1e5, ls, color=COLOURS[s], lw=1,
                       label={'pulse': 'pulse, 1-D', 'pulse0d': 'pulse, 0-D', 'cp': 'constant pressure, 1-D',
                              'cp0d': 'constant pressure, 0-D'}[s])
        ax[j].set(xlabel='crank angle over one cycle (deg)', title=f'turbine inlet pressure, {int(load * 100)} % load')
    ax[0].set_ylabel('pressure (bar)')
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_turbine_inlet_pressure.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for s in SYSTEMS:
        d = df[df.system == s].sort_values('load')
        ls = '--' if s.endswith('0d') else '-'
        lab = s.replace('0d', ', 0-D').replace('pulse', 'pulse').replace('cp', 'constant pressure')
        if not s.endswith('0d'):
            lab += ', 1-D'
        ax[0].plot(d.load * 100, d.p_boost_bar, ls + 'o', color=COLOURS[s], ms=3, label=lab)
        ax[1].plot(d.load * 100, d.W_turbine_avail_J / 1e3, ls + 'o', color=COLOURS[s], ms=3)
        ax[2].plot(d.load * 100, d.eta_indicated * 100, ls + 'o', color=COLOURS[s], ms=3)
    ax[0].set(xlabel='load (%)', ylabel='matched boost pressure (bar)')
    ax[1].set(xlabel='load (%)', ylabel='turbine isentropic energy per cycle (kJ)')
    ax[2].set(xlabel='load (%)', ylabel='indicated efficiency (%)')
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_operating_line.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for base, col in [('pulse', 'C3'), ('cp', 'C0')]:
        a = df[df.system == base].sort_values('load')
        b = df[df.system == base + '0d'].sort_values('load')
        err = (b.W_turbine_avail_J.values - a.W_turbine_avail_J.values) / a.W_turbine_avail_J.values * 100
        ax.plot(a.load * 100, err, 'o-', color=col, label={'pulse': 'pulse', 'cp': 'constant pressure'}[base])
    ax.axhline(0, color='grey', lw=0.8)
    ax.set(xlabel='load (%)', ylabel='0-D error in turbine energy (%)', title='filling-and-emptying vs 1-D')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_0d_vs_1d.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(sweep.coll_len, sweep.W_turbine_avail_J / 1e3, 'o-', color='C3')
    ax[0].set(xlabel='collector length (m)', ylabel='turbine isentropic energy per cycle (kJ)')
    ax[1].plot(sweep.coll_len, sweep.m_air_kg_per_cycle * 1e3, 'o-', color='C2')
    ax[1].set(xlabel='collector length (m)', ylabel='trapped + scavenged air per cycle (g)')
    fig.suptitle(f'pulse system, full load, boost fixed at {pb_full / 1e5:.2f} bar', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_collector_length.png')
    plt.close(fig)

    summ = dict(engine=dict(bore_m=EngineSpec().bore, stroke_m=EngineSpec().stroke, rpm=EngineSpec().rpm,
                            n_cyl=EngineSpec().n_cyl, cr=EngineSpec().cr, full_fuel_J_per_cyl_cycle=FULL_FUEL),
                turbo=Turbo().__dict__, layout=Layout().__dict__,
                matching=df.to_dict(orient='records'), length_sweep=sweep.to_dict(orient='records'),
                runtime_s=round(time.time() - t0, 1))
    (RES / 'summary.json').write_text(json.dumps(summ, indent=2, default=float))
    print('runtime', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
