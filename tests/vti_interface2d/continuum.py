"""NumPy-only continuum angular-spectrum scattering; no FEM imports."""

import numpy as np

from .packets import EXTENT, FREQUENCY, SIGMA_Q, SIGMA_S, SPEED, TIME, geometry, incident_transform
from .reference import LOWER, NAMES, UPPER, solve, squared_roots


def dispersion(kx, kz, m, kind):
    D = np.empty(kx.shape + (2, 2))
    D[..., 0, 0] = m.c11 * kx * kx + m.c55 * kz * kz
    D[..., 1, 1] = m.c55 * kx * kx + m.c33 * kz * kz
    D[..., 0, 1] = D[..., 1, 0] = (m.c13 + m.c55) * kx * kz
    ev, d = np.linalg.eigh(D)
    j = 1 if kind == "P" else 0
    return np.sqrt(ev[..., j] / m.rho), d[..., :, j]


def field(angle, h, extent=EXTENT, time=TIME, upper=UPPER):
    size = round(2 * extent / h) + 1
    k = 2 * np.pi * np.fft.fftfreq(size, h)
    kx, kz = np.meshgrid(k, k)
    spectral = np.zeros((size, size, 4), complex)
    for name in NAMES:
        m = LOWER if name[0] == "R" else upper
        sign = -1 if name[0] == "R" else 1
        omega, _ = dispersion(kx, kz, m, name[1])
        p = kx / np.maximum(omega, 1e-30)
        _, valid = squared_roots(p, upper)
        valid &= (abs(p) < 1 / SPEED) & (sign * kz > 0)
        kiz = np.sqrt(np.maximum((omega / SPEED) ** 2 - kx * kx, 0))
        inc = incident_transform(kx, kiz, angle)
        active = valid & (abs(inc) > 1e-10 * abs(inc).max())
        wi, branches = solve(p[active], upper=upper)
        b = branches[name]
        # d kz_out / d kz_in at fixed kx = group_z_in / group_z_out.
        J = abs(wi["g"][..., 1] / b["g"][..., 1])
        amplitude = inc[active] * b["amplitude"] / J * np.exp(-1j * omega[active] * time)
        spectral[active, :2] += amplitude[:, None] * b["d"]
        spectral[active, 2:] += (-1j * omega[active] * amplitude)[:, None] * b["d"]
    phase = np.exp(-1j * extent * (kx + kz))
    return 2 * np.fft.ifft2(spectral * phase[..., None], axes=(0, 1)).real / h**2


def bandwidth(angle, upper=UPPER):
    n, t, _ = geometry(angle)
    ks, kq = np.meshgrid(
        2 * np.pi * FREQUENCY / SPEED + np.linspace(-6, 6, 151) / SIGMA_S,
        np.linspace(-6, 6, 151) / SIGMA_Q,
    )
    kx, kz = ks * n[0] + kq * t[0], ks * n[1] + kq * t[1]
    radius = np.hypot(kx, kz)
    weight = (SPEED * radius) ** 2 * abs(incident_transform(kx, kz, angle)) ** 2
    p = kx / np.maximum(SPEED * radius, 1e-30)
    _, valid = squared_roots(p, upper)
    valid &= (kz > 0) & (abs(p) < 1 / SPEED)
    excluded = float(weight[~valid].sum() / weight.sum())
    weight = np.where(valid, weight, 0)
    weight /= weight.sum()
    theta = np.degrees(np.arctan2(kx, kz))
    mean = float(np.sum(weight * theta))
    active = weight > 1e-14
    _, b = solve(p[active], upper=upper)
    return dict(
        angle=mean,
        spread=float(np.sqrt(np.sum(weight * (theta - mean) ** 2))),
        excluded_energy=excluded,
        integrated_flux={
            name: float(np.sum(weight[active] * w["fraction"])) for name, w in b.items()
        },
    )


def energy(grid, h, extent=EXTENT, upper=UPPER):
    """Physical kinetic+strain integral, independent of the spectral Jacobian."""
    size = len(grid)
    k = 2 * np.pi * np.fft.fftfreq(size, h)
    U = np.fft.fft2(grid[:, :, :2], axes=(0, 1))
    dx = np.fft.ifft2(U * (1j * k)[None, :, None], axes=(0, 1)).real
    dz = np.fft.ifft2(U * (1j * k)[:, None, None], axes=(0, 1)).real
    z = np.linspace(-extent, extent, size)[:, None]
    rho, c11, c33, c13, c55 = [
        np.where(z > 0, getattr(upper, key), getattr(LOWER, key))
        for key in ["rho", "c11", "c33", "c13", "c55"]
    ]
    ex, ez, gamma = dx[:, :, 0], dz[:, :, 1], dz[:, :, 0] + dx[:, :, 1]
    density = 0.5 * (
        rho * np.sum(grid[:, :, 2:] ** 2, axis=-1)
        + c11 * ex**2
        + 2 * c13 * ex * ez
        + c33 * ez**2
        + c55 * gamma**2
    )
    return float(density.sum() * h * h)
