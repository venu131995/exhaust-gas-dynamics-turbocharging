"""Six-cylinder, four-stroke medium-speed engine: geometry, valves, combustion and wall heat transfer.

Representative of a 20/28 cm bore/stroke generating-set engine at 1000 rpm (not a specific product).
Crank angle theta in degrees after firing TDC, 0..720.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .gas import CV, GAMMA, R_GAS


@dataclass
class EngineSpec:
    bore: float = 0.20
    stroke: float = 0.28
    rod_ratio: float = 4.0           # connecting rod / crank radius
    cr: float = 15.0
    rpm: float = 1000.0
    n_cyl: int = 6
    # valves: number, inner seat diameter (m), max lift (m), discharge coefficient, open/close (deg)
    exh: tuple = (2, 0.066, 0.019, 0.65, 130.0, 400.0)
    inl: tuple = (2, 0.070, 0.020, 0.65, 320.0, 570.0)
    # combustion: single Wiebe function
    soc: float = -8.0
    duration: float = 60.0
    wiebe_a: float = 6.9
    wiebe_m: float = 1.5
    T_wall: float = 450.0
    # firing offsets (deg): two groups of three cylinders, 240 deg apart within a group
    offsets: tuple = (0.0, 240.0, 480.0, 120.0, 360.0, 600.0)

    @property
    def omega(self):
        return self.rpm * 2 * math.pi / 60.0

    @property
    def v_disp(self):
        return math.pi / 4 * self.bore ** 2 * self.stroke

    @property
    def v_clear(self):
        return self.v_disp / (self.cr - 1.0)

    @property
    def t_cycle(self):
        return 2 * 60.0 / self.rpm

    @property
    def piston_speed(self):
        return 2 * self.stroke * self.rpm / 60.0


def crank_angle(spec, t, offset):
    return (math.degrees(spec.omega * t) - offset) % 720.0


def cyl_volume(spec, theta_deg):
    """Volume and dV/dtheta (per radian)."""
    r = spec.stroke / 2
    l = spec.rod_ratio * r
    th = math.radians(theta_deg)
    s = math.sin(th)
    root = math.sqrt(l * l - r * r * s * s)
    x = r * (1 - math.cos(th)) + l - root
    dx = r * s + r * r * s * math.cos(th) / root
    Ap = math.pi / 4 * spec.bore ** 2
    return spec.v_clear + Ap * x, Ap * dx


def valve_area(valve, theta_deg):
    """Effective flow area of a valve set: smooth sin^2 lift, curtain area limited by the port area."""
    n, d, lmax, cd, open_, close = valve
    th = theta_deg
    if close > 720.0 and th < open_:
        th += 720.0
    if th <= open_ or th >= close:
        return 0.0
    lift = lmax * math.sin(math.pi * (th - open_) / (close - open_)) ** 2
    return n * cd * min(math.pi * d * lift, math.pi * d * d / 4)


def wiebe_rate(spec, theta_deg):
    """d(burned fraction)/d(theta), per degree."""
    th = theta_deg if theta_deg < 360 else theta_deg - 720
    z = (th - spec.soc) / spec.duration
    if z <= 0 or z >= 1.6:
        return 0.0
    a, m = spec.wiebe_a, spec.wiebe_m
    return a * (m + 1) / spec.duration * z ** m * math.exp(-a * z ** (m + 1))


class Cylinder:
    """Callbacks that make a network Volume behave as a cylinder."""

    def __init__(self, spec, offset, fuel_energy):
        self.spec, self.offset, self.q_fuel = spec, offset, fuel_energy
        self.ivc_state = None        # (p, T, V) at the last intake valve closing (for Woschni)
        self.htc_on = True

    def V_fn(self, t):
        th = crank_angle(self.spec, t, self.offset)
        V, dVdth = cyl_volume(self.spec, th)
        return V, dVdth * self.spec.omega

    def phase_fn(self, t):
        """'gas exchange' for the exhaust and intake strokes (BDC 180 deg to BDC 540 deg), else 'high pressure'."""
        th = crank_angle(self.spec, t, self.offset)
        return 'gas exchange' if 180.0 <= th < 540.0 else 'high pressure'

    def heat_fn(self, vol, t, p, T, V):
        sp = self.spec
        th = crank_angle(sp, t, self.offset)
        q_comb = self.q_fuel * wiebe_rate(sp, th) * math.degrees(sp.omega)    # W
        if not self.htc_on:
            return q_comb, 0.0
        # Woschni (1967): h = 3.26 B^-0.2 p[kPa]^0.8 T^-0.55 w^0.8, C1 = 6.18 from EVO to IVC, else 2.28;
        # combustion term from start of combustion to EVO
        evo, ivc = sp.exh[4], sp.inl[5]
        c1 = 6.18 if evo <= th <= ivc else 2.28
        w = c1 * sp.piston_speed
        if self.ivc_state is not None and (th >= 720 + sp.soc or th <= evo):
            p_r, T_r, V_r = self.ivc_state
            p_mot = p_r * (V_r / V) ** GAMMA
            w += 3.24e-3 * sp.v_disp * T_r / (p_r * V_r) * max(p - p_mot, 0.0)
        h = 3.26 * sp.bore ** -0.2 * (p / 1e3) ** 0.8 * T ** -0.55 * w ** 0.8
        x = (V - sp.v_clear) / (math.pi / 4 * sp.bore ** 2)
        area = 2 * math.pi / 4 * sp.bore ** 2 + math.pi * sp.bore * x
        return q_comb, h * area * (T - sp.T_wall)
