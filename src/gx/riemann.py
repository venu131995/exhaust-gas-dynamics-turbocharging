"""Exact solution of the Riemann problem for the 1-D Euler equations (Toro, ch. 4), for verification."""
import numpy as np


def _f(p, rho, pk, ck, g):
    if p > pk:   # shock
        A = 2 / ((g + 1) * rho)
        B = (g - 1) / (g + 1) * pk
        return (p - pk) * np.sqrt(A / (p + B)), np.sqrt(A / (B + p)) * (1 - (p - pk) / (2 * (B + p)))
    pr = p / pk   # rarefaction
    return 2 * ck / (g - 1) * (pr ** ((g - 1) / (2 * g)) - 1), 1 / (rho * ck) * pr ** (-(g + 1) / (2 * g))


def star_state(rL, uL, pL, rR, uR, pR, g):
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    p = max(1e-8, 0.5 * (pL + pR))
    for _ in range(100):
        fL, dL = _f(p, rL, pL, cL, g)
        fR, dR = _f(p, rR, pR, cR, g)
        dp = (fL + fR + uR - uL) / (dL + dR)
        p_new = max(1e-10, p - dp)
        if abs(p_new - p) / (0.5 * (p_new + p)) < 1e-14:
            p = p_new
            break
        p = p_new
    fL, _ = _f(p, rL, pL, cL, g)
    fR, _ = _f(p, rR, pR, cR, g)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(x, t, x0, left, right, g):
    """Exact (rho, u, p) at positions x and time t for a discontinuity at x0."""
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    ps, us = star_state(rL, uL, pL, rR, uR, pR, g)
    out = np.zeros((3, len(x)))
    for i, xi in enumerate(x):
        s = (xi - x0) / t
        if s <= us:   # left of contact
            if ps > pL:
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                if s <= SL:
                    out[:, i] = (rL, uL, pL)
                else:
                    r = rL * ((ps / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pL + 1))
                    out[:, i] = (r, us, ps)
            else:
                SHL = uL - cL
                csL = cL * (ps / pL) ** ((g - 1) / (2 * g))
                STL = us - csL
                if s <= SHL:
                    out[:, i] = (rL, uL, pL)
                elif s >= STL:
                    out[:, i] = (rL * (ps / pL) ** (1 / g), us, ps)
                else:
                    c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
                    u = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
                    r = rL * (c / cL) ** (2 / (g - 1))
                    out[:, i] = (r, u, pL * (c / cL) ** (2 * g / (g - 1)))
        else:
            if ps > pR:
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                if s >= SR:
                    out[:, i] = (rR, uR, pR)
                else:
                    r = rR * ((ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1))
                    out[:, i] = (r, us, ps)
            else:
                SHR = uR + cR
                csR = cR * (ps / pR) ** ((g - 1) / (2 * g))
                STR = us + csR
                if s >= SHR:
                    out[:, i] = (rR, uR, pR)
                elif s <= STR:
                    out[:, i] = (rR * (ps / pR) ** (1 / g), us, ps)
                else:
                    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
                    u = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
                    r = rR * (c / cR) ** (2 / (g - 1))
                    out[:, i] = (r, u, pR * (c / cR) ** (2 * g / (g - 1)))
    return out
