"""Independent angular-spectrum propagation of a Gaussian potential.

Only NumPy and the independent welded-interface reference are imported. Inputs
are physical packet parameters, not FEM functions. No fitted numerical data.
"""

import numpy as np

from .reference import LOWER, NAMES, UPPER, solve


def incident_transform(kx, kz, mode, theta, frequency, sigma_s, sigma_q, center):
    """Positive-time-frequency scalar displacement Fourier density at t=0.

    Exact decomposition of u0 and v0=-c n0.grad(u0) into forward/backward
    continuum waves; the tiny backward packet is excluded from scattering.
    """
    n = np.array([np.sin(theta), np.cos(theta)])
    transverse = np.array([n[1], -n[0]])
    radius = np.hypot(kx, kz)
    ks, kq = kx * n[0] + kz * n[1], kx * transverse[0] + kz * transverse[1]
    k0 = 2 * np.pi * frequency / LOWER.speed(mode)
    potential = (
        np.pi
        * sigma_s
        * sigma_q
        / k0
        * np.exp(-0.5 * (sigma_q * kq) ** 2)
        * (np.exp(-0.5 * (sigma_s * (ks - k0)) ** 2) + np.exp(-0.5 * (sigma_s * (ks + k0)) ** 2))
    )
    potential = potential * np.exp(-1j * (kx * center[0] + kz * center[1]))
    return 1j * radius * potential * (1 + ks / np.maximum(radius, 1e-30)) / 2


def field(mode, theta, frequency, sigma_s, sigma_q, center, time, extent, h):
    """Synthesize all outgoing branches with exact continuum dispersion.

    Uniform Fourier quadrature on a periodic box; packet/image truncation is
    tested by the same estimator. Returned array contains real u_x,u_z,v_x,v_z.
    """
    size = round(2 * extent / h) + 1
    k = 2 * np.pi * np.fft.fftfreq(size, h)
    kx, kz = np.meshgrid(k, k)
    radius = np.hypot(kx, kz)
    spectral = np.zeros((size, size, 4), dtype=complex)
    ci = LOWER.speed(mode)
    for name in NAMES:
        mat = LOWER if name[0] == "R" else UPPER
        sign = -1 if name[0] == "R" else 1
        c = mat.speed(name[1])
        omega = c * radius
        propagating = (sign * kz > 0) & (kx > 0) & (omega > UPPER.vp * abs(kx))
        kiz = np.sqrt(np.maximum((omega / ci) ** 2 - kx * kx, 0))
        incident = incident_transform(kx, kiz, mode, theta, frequency, sigma_s, sigma_q, center)
        active = propagating & (abs(incident) > 1e-9 * abs(incident).max())
        indices = np.argwhere(active)
        for iz, ix in indices:
            w = omega[iz, ix]
            p = kx[iz, ix] / w
            b = solve(mode, p=p)[name]
            ni_z = ci * kiz[iz, ix] / w
            nb_z = abs(c * kz[iz, ix] / w)
            jacobian = ci * ni_z / (c * nb_z)
            amplitude = incident[iz, ix] * b["amplitude"] / jacobian * np.exp(-1j * w * time)
            d = np.array(b["polarization"])
            spectral[iz, ix, :2] += amplitude * d
            spectral[iz, ix, 2:] += -1j * w * amplitude * d
    phase = np.exp(-1j * extent * (kx + kz))
    return 2 * np.fft.ifft2(spectral * phase[:, :, None], axes=(0, 1)).real / h**2


def bandwidth(mode, theta, frequency, sigma_s, sigma_q, center):
    """Energy-weighted incident angular bandwidth and integrated flux partition.

    Integrate in packet-aligned Fourier coordinates with fixed Gaussian support.
    Plane-wave flux fractions are weighted by omega² |U_inc(k)|². This is total
    packet energy weighting, including the Fourier-coordinate Jacobians.
    """
    ci = LOWER.speed(mode)
    k0 = 2 * np.pi * frequency / ci
    ks, kq = np.meshgrid(k0 + np.linspace(-5, 5, 101) / sigma_s, np.linspace(-5, 5, 101) / sigma_q)
    kx = ks * np.sin(theta) + kq * np.cos(theta)
    kz = ks * np.cos(theta) - kq * np.sin(theta)
    radius = np.hypot(kx, kz)
    a = incident_transform(kx, kz, mode, theta, frequency, sigma_s, sigma_q, center)
    weights = (ci * radius) ** 2 * abs(a) ** 2
    valid = (kz > 0) & (UPPER.vp * abs(kx) < ci * radius)
    excluded = float(weights[~valid].sum() / weights.sum())
    weights = np.where(valid, weights, 0)
    weights /= weights.sum()
    angles = np.rad2deg(np.arctan2(kx, kz))
    mean = float(np.sum(weights * angles))
    spread = float(np.sqrt(np.sum(weights * (angles - mean) ** 2)))
    flux = dict.fromkeys(NAMES, 0.0)
    amplitude = dict.fromkeys(NAMES, 0.0)
    for iz, ix in np.argwhere(weights > 1e-12):
        ref = solve(mode, p=kx[iz, ix] / (ci * radius[iz, ix]))
        for name in NAMES:
            flux[name] += float(weights[iz, ix] * ref[name]["flux"])
            amplitude[name] += float(weights[iz, ix] * ref[name]["amplitude"])
    return dict(angle=mean, spread=spread, flux=flux, amplitude=amplitude, excluded_energy=excluded)
