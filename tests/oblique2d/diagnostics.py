"""Windowed Fourier diagnostics; no predicted-angle search windows.

The half-space Fourier transform changes density under refraction:
J = |d kz_out / d kz_in| at fixed kx = c_in |nz_in|/(c_out |nz_out|).
Thus U_out(k_out,t) = Z U_in(k_in,0) exp(-i omega t) / J.
The signed estimate divides complex transforms, retaining the phase residual.
"""

import numpy as np

from .packets import ANGLES, EXTENT, FREQUENCY, TIMES, initial, parameters
from .reference import LOWER, NAMES, UPPER, solve


def transform(field, x, z, kx, kz, h):
    wx, wz = np.exp(-1j * kx * x), np.exp(-1j * kz * z)
    return np.einsum("z,zxc,x->c", wz, field, wx) * h * h


def spectrum(u, h, side, kind, direction=None):
    direction = side if direction is None else direction
    size = len(u)
    z = np.linspace(-EXTENT, EXTENT, size)
    # Away from interface, a cosine ramp of fixed physical width suppresses the cut.
    distance = side * z
    w = np.sin(np.pi / 2 * np.clip((distance - 120) / 360, 0, 1)) ** 2
    ux, uz = u[:, :, 0], u[:, :, 1]
    div = np.gradient(ux, h, axis=1) + np.gradient(uz, h, axis=0)
    curl = np.gradient(ux, h, axis=0) - np.gradient(uz, h, axis=1)
    proxy = (div if kind == "P" else curl) * w[:, None]
    pad = 2 * size
    power = abs(np.fft.fft2(proxy, s=(pad, pad))) ** 2
    k = 2 * np.pi * np.fft.fftfreq(pad, h)
    kx, kz = np.meshgrid(k, k)
    radius = np.hypot(kx, kz)
    c = (LOWER if side < 0 else UPPER).speed(kind)
    f = radius * c / (2 * np.pi)
    # A broad frequency band and outgoing quadrant, no Snell prediction.
    mask = (kx > 0) & (direction * kz > 0) & (f > 0.5 * FREQUENCY) & (f < 1.5 * FREQUENCY)
    power = np.where(mask, power / np.maximum(radius**2, 1e-30), 0)
    # Connected local lobe selected by data: retain points above 2% of its peak.
    eligible = power >= 0.02 * power.max()
    selected = np.zeros(power.shape, dtype=bool)
    pending = [np.unravel_index(np.argmax(power), power.shape)]
    while pending:
        iz, ix = pending.pop()
        if selected[iz, ix] or not eligible[iz, ix]:
            continue
        selected[iz, ix] = True
        for dz, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pending.append(((iz + dz) % pad, (ix + dx) % pad))
    power = np.where(selected, power, 0)
    angle = np.rad2deg(np.arctan2(kx, abs(kz)))
    mean = float(np.sum(power * angle) / power.sum())
    spread = float(np.sqrt(np.sum(power * (angle - mean) ** 2) / power.sum()))
    return mean, spread


def measure(grid, mode, h):
    x = z = np.linspace(-EXTENT, EXTENT, len(grid))
    xx, zz = np.meshgrid(x, z)
    xy = np.column_stack((xx.ravel(), zz.ravel()))
    u0, v0 = initial(xy, mode)
    u0, v0 = u0.reshape(grid.shape[:2] + (2,)), v0.reshape(grid.shape[:2] + (2,))
    n, transverse, c, k, _ = parameters(mode)
    d = n if mode == "P" else transverse
    omega = 2 * np.pi * FREQUENCY
    inc = transform((u0 + 1j * v0 / omega) / 2, x, z, *(k * n), h) @ d
    # Incident spectrum measured with the same proxy in the lower half-space.
    incident_angle, incident_spread = spectrum(u0, h, -1, mode, direction=1)
    ref = solve(mode, ANGLES[mode])
    branches = {}
    for name in NAMES:
        b = ref[name]
        side = -1 if name[0] == "R" else 1
        kind = name[1]
        angle, spread = spectrum(grid[:, :, :2], h, side, kind)
        mat = LOWER if side < 0 else UPPER
        cb = mat.speed(kind)
        nb, db = np.array(b["direction"]), np.array(b["polarization"])
        # Central transfer is evaluated at the known input frequency and conserved
        # kx, not at an angle chosen to maximize agreement. Angle estimator above
        # receives neither incidence angle nor analytic branch angle.
        outgoing = (grid[:, :, :2] + 1j * grid[:, :, 2:] / omega) / 2
        outgoing = outgoing * (side * z[:, None, None] > 0)
        value = transform(outgoing, x, z, *(omega / cb * nb), h) @ db
        jacobian = c * n[1] / (cb * abs(nb[1]))
        coefficient = value / inc * jacobian * np.exp(1j * omega * TIMES[mode])
        flux = b["flux_factor"] * abs(coefficient) ** 2
        branches[name] = dict(
            angle=angle,
            spread=spread,
            angle_error=abs(angle - b["angle"]),
            amplitude=float(coefficient.real),
            imaginary=float(coefficient.imag),
            amplitude_error=float(abs(coefficient - b["amplitude"])),
            signed_error=float(abs(coefficient.real - b["amplitude"])),
            flux=float(flux),
            flux_error=float(abs(flux - b["flux"])),
        )
    total = sum(b["flux"] for b in branches.values())
    return dict(
        incident_angle=incident_angle,
        incident_spread=incident_spread,
        branches=branches,
        flux_sum=total,
        closure_error=abs(total - 1),
    )


def physical_energy(grid, h):
    """Independent continuum integral of kinetic plus elastic strain energy.

    Spectral derivatives, not FEM matrices; used to audit analytic beam synthesis.
    The waves are negligible at the box edges and away from the interface at the
    final measurement, so periodic derivative wrap errors are small.
    """
    size = len(grid)
    k = 2 * np.pi * np.fft.fftfreq(size, h)
    u = grid[:, :, :2]
    dx = np.fft.ifft(np.fft.fft(u, axis=1) * (1j * k)[None, :, None], axis=1).real
    dz = np.fft.ifft(np.fft.fft(u, axis=0) * (1j * k)[:, None, None], axis=0).real
    z = np.linspace(-EXTENT, EXTENT, size)[:, None]
    rho = np.where(z > 0, UPPER.rho, LOWER.rho)
    mu = np.where(z > 0, UPPER.rho * UPPER.vs**2, LOWER.rho * LOWER.vs**2)
    lam = np.where(
        z > 0,
        UPPER.rho * (UPPER.vp**2 - 2 * UPPER.vs**2),
        LOWER.rho * (LOWER.vp**2 - 2 * LOWER.vs**2),
    )
    density = (
        0.5 * rho * np.sum(grid[:, :, 2:] ** 2, axis=2)
        + 0.5 * lam * (dx[:, :, 0] + dz[:, :, 1]) ** 2
        + mu * (dx[:, :, 0] ** 2 + dz[:, :, 1] ** 2 + 0.5 * (dx[:, :, 1] + dz[:, :, 0]) ** 2)
    )
    return float(density.sum() * h * h)
