"""Physical incident potential and its independent Fourier transform (NumPy only)."""

import numpy as np

FREQUENCY = 5.0
SPEED = 3000.0
SIGMA_S = 360.0
SIGMA_Q = 1500.0
EXTENT = 8400.0
TIME = 2.8
ANGLES = (0.0, 15.0, 25.0, 35.0, -25.0)


def geometry(angle):
    theta = np.deg2rad(angle)
    n = np.array([np.sin(theta), np.cos(theta)])
    transverse = np.array([n[1], -n[0]])
    center = np.array([-3600 * np.tan(theta), -3600.0])
    return n, transverse, center


def initial(x, angle):
    n, t, center = geometry(angle)
    s, q = (x - center) @ n, (x - center) @ t
    k = 2 * np.pi * FREQUENCY / SPEED
    g = np.exp(-0.5 * ((s / SIGMA_S) ** 2 + (q / SIGMA_Q) ** 2)) / k
    a, b = -s / SIGMA_S**2, -q / SIGMA_Q**2
    cs, sn = np.cos(k * s), np.sin(k * s)
    fs = g * (a * cs - k * sn)
    fq = g * b * cs
    fss = g * ((a * a - 1 / SIGMA_S**2 - k * k) * cs - 2 * a * k * sn)
    fsq = g * b * (a * cs - k * sn)
    return fs[:, None] * n + fq[:, None] * t, -SPEED * (fss[:, None] * n + fsq[:, None] * t)


def incident_transform(kx, kz, angle):
    n, t, center = geometry(angle)
    ks, kq = kx * n[0] + kz * n[1], kx * t[0] + kz * t[1]
    radius = np.hypot(kx, kz)
    k0 = 2 * np.pi * FREQUENCY / SPEED
    potential = (
        np.pi
        * SIGMA_S
        * SIGMA_Q
        / k0
        * np.exp(-0.5 * (SIGMA_Q * kq) ** 2)
        * (np.exp(-0.5 * (SIGMA_S * (ks - k0)) ** 2) + np.exp(-0.5 * (SIGMA_S * (ks + k0)) ** 2))
    )
    return (
        1j
        * radius
        * potential
        * np.exp(-1j * (kx * center[0] + kz * center[1]))
        * (1 + ks / np.maximum(radius, 1e-30))
        / 2
    )
