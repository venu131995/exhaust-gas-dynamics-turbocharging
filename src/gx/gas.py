"""Ideal-gas relations, compressible orifice flow and the HLLC Riemann flux.

One calorically perfect gas is used throughout (gamma = 1.35, R = 287 J/kg/K, a compromise between air and
lean burned gas). State per pipe cell is stored as conserved quantities per unit length:
    q = (rho A, rho u A, E A),  E = p / (gamma - 1) + rho u^2 / 2
"""
import numpy as np

GAMMA = 1.35
R_GAS = 287.0
CP = R_GAS * GAMMA / (GAMMA - 1.0)
CV = CP - R_GAS


def primitive(q, A, g=GAMMA):
    rho = q[0] / A
    u = q[1] / q[0]
    p = (g - 1.0) * (q[2] / A - 0.5 * rho * u * u)
    return rho, u, p


def conserved(rho, u, p, A, g=GAMMA):
    return np.array([rho * A, rho * u * A, (p / (g - 1.0) + 0.5 * rho * u * u) * A])


def hllc(rhoL, uL, pL, rhoR, uR, pR, g=GAMMA):
    """HLLC flux (Toro) per unit area for arrays of left/right states. Returns (F_mass, F_mom, F_energy)."""
    cL = np.sqrt(g * pL / rhoL)
    cR = np.sqrt(g * pR / rhoR)
    EL = pL / (g - 1) + 0.5 * rhoL * uL * uL
    ER = pR / (g - 1) + 0.5 * rhoR * uR * uR
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)
    den = rhoL * (SL - uL) - rhoR * (SR - uR)
    SM = num / den
    FL = (rhoL * uL, rhoL * uL * uL + pL, uL * (EL + pL))
    FR = (rhoR * uR, rhoR * uR * uR + pR, uR * (ER + pR))

    def star(rho, u, p, E, S):
        f = rho * (S - u) / (S - SM)
        return (f, f * SM, f * (E / rho + (SM - u) * (SM + p / (rho * (S - u)))))

    UsL = star(rhoL, uL, pL, EL, SL)
    UsR = star(rhoR, uR, pR, ER, SR)
    UL = (rhoL, rhoL * uL, EL)
    UR = (rhoR, rhoR * uR, ER)
    out = []
    for k in range(3):
        fsl = FL[k] + SL * (UsL[k] - UL[k])
        fsr = FR[k] + SR * (UsR[k] - UR[k])
        out.append(np.where(SL >= 0, FL[k], np.where(SM >= 0, fsl, np.where(SR > 0, fsr, FR[k]))))
    return out


def orifice_mass_flow(A_eff, p0, T0, p_down, g=GAMMA, R=R_GAS):
    """Quasi-steady isentropic flow through a restriction (chokes at the critical pressure ratio).

    Arrays allowed. Returns 0 where p_down >= p0.
    """
    p0 = np.asarray(p0, float)
    pr = np.clip(np.asarray(p_down, float) / p0, 0.0, 1.0)
    pr_crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    pr = np.maximum(pr, pr_crit)
    psi = np.sqrt(np.maximum(2 * g / (g - 1) * (pr ** (2 / g) - pr ** ((g + 1) / g)), 0.0))
    return A_eff * p0 / np.sqrt(R * T0) * psi


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def mc_limiter(a, b):
    """Monotonised-central slope limiter (van Leer 1977): second order on smooth solutions, TVD."""
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.minimum(2 * np.abs(a), 2 * np.abs(b)), 0.5 * np.abs(a + b)), 0.0)
