"""Analytically pure potentials; only test code uses prescribed initial data."""

import numpy as np

from .reference import LOWER

FREQUENCY = 5.0
SIGMA_Q = 1500.0
CENTER = np.array([-1400.0, -3000.0])
ANGLES = dict(P=25.0, S=15.0)
TIMES = dict(P=2.25, S=3.1)
EXTENT = 7200.0


def parameters(mode):
    theta = np.deg2rad(ANGLES[mode])
    n = np.array([np.sin(theta), np.cos(theta)])
    transverse = np.array([n[1], -n[0]])
    c = LOWER.speed(mode)
    return n, transverse, c, 2 * np.pi * FREQUENCY / c, 0.6 * c / FREQUENCY


def initial(x, mode):
    """Phi=exp(-s²/2σs²-q²/2σq²) cos(ks)/k; v=-c ∂s u.

    P uses grad Phi, S uses (Phi_z,-Phi_x), exactly d_S=(n_z,-n_x).
    """
    n, transverse, c, k, sigma_s = parameters(mode)
    s, q = (x - CENTER) @ n, (x - CENTER) @ transverse
    g = np.exp(-0.5 * ((s / sigma_s) ** 2 + (q / SIGMA_Q) ** 2)) / k
    a, b = -s / sigma_s**2, -q / SIGMA_Q**2
    cs, sn = np.cos(k * s), np.sin(k * s)
    fs = g * (a * cs - k * sn)
    fq = g * b * cs
    fss = g * ((a * a - 1 / sigma_s**2 - k * k) * cs - 2 * a * k * sn)
    fsq = g * b * (a * cs - k * sn)
    u = fs[:, None] * n + fq[:, None] * transverse
    v = -c * (fss[:, None] * n + fsq[:, None] * transverse)
    if mode == "S":
        u, v = u[:, [1, 0]] * np.array([1, -1]), v[:, [1, 0]] * np.array([1, -1])
    return u, v
