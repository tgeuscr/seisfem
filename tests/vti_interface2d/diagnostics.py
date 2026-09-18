"""Half-space Fourier projections, complex central transfer, independent angles."""

import numpy as np

from .continuum import dispersion
from .packets import EXTENT, FREQUENCY, SPEED, TIME, initial
from .reference import LOWER, NAMES, UPPER, solve


def transform(u, x, kx, kz, h):
    return np.einsum("z,zxc,x->c", np.exp(-1j * kz * x), u, np.exp(-1j * kx * x)) * h * h


def angles(grid, h, side, kind, m, extent):
    size = len(grid)
    z = np.linspace(-extent, extent, size)
    window = np.sin(np.pi / 2 * np.clip((side * z - 120) / 360, 0, 1)) ** 2
    pad = 2 * size
    U = np.fft.fft2(grid[:, :, :2] * window[:, None, None], s=(pad, pad), axes=(0, 1))
    k = 2 * np.pi * np.fft.fftfreq(pad, h)
    kx, kz = np.meshgrid(k, k)
    omega, d = dispersion(kx, kz, m, kind)
    power = abs(np.sum(U * d, axis=-1)) ** 2
    # Entire outgoing hemisphere and broad frequency range; no Snell-angle input.
    mask = (side * kz > 0) & (omega > np.pi * FREQUENCY) & (omega < 3 * np.pi * FREQUENCY)
    power = np.where(mask, power, 0)
    eligible = power >= 0.02 * power.max()
    chosen = np.zeros(power.shape, bool)
    pending = [np.unravel_index(np.argmax(power), power.shape)]
    while pending:
        iz, ix = pending.pop()
        if chosen[iz, ix] or not eligible[iz, ix]:
            continue
        chosen[iz, ix] = True
        for dz, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pending.append(((iz + dz) % pad, (ix + dx) % pad))
    power = np.where(chosen, power, 0)
    power /= power.sum()
    theta = np.degrees(np.arctan2(kx, abs(kz)))
    mean = float(np.sum(power * theta))
    return mean, float(np.sqrt(np.sum(power * (theta - mean) ** 2)))


def measure(grid, angle, h, extent=EXTENT, upper=UPPER):
    axis = np.linspace(-extent, extent, len(grid))
    x, z = np.meshgrid(axis, axis)
    u, v = initial(np.column_stack((x.ravel(), z.ravel())), angle)
    omega = 2 * np.pi * FREQUENCY
    p = np.sin(np.deg2rad(angle)) / SPEED
    wi, ref = solve(p, upper=upper)
    incident = (
        transform(
            ((u + 1j * v / omega) / 2).reshape(grid.shape[:2] + (2,)),
            axis,
            omega * p,
            omega * wi["q"],
            h,
        )
        @ wi["d"]
    )
    branches = {}
    analytic = (grid[:, :, :2] + 1j * grid[:, :, 2:] / omega) / 2
    for name in NAMES:
        b = ref[name]
        side = -1 if name[0] == "R" else 1
        output = (
            transform(
                analytic * (side * axis[:, None, None] > 0), axis, omega * p, omega * b["q"], h
            )
            @ b["d"]
        )
        J = abs(wi["g"][1] / b["g"][1])
        estimate = output / incident * J * np.exp(1j * omega * TIME)
        flux = float(b["flux_factor"] * abs(estimate) ** 2)
        central = float(np.degrees(np.arctan2(b["n"][0], abs(b["n"][1]))))
        measured, spread = angles(grid, h, side, name[1], LOWER if side < 0 else upper, extent)
        branches[name] = dict(
            real=float(estimate.real),
            imaginary=float(estimate.imag),
            magnitude=float(abs(estimate)),
            phase_residual=float(np.angle(estimate / b["amplitude"]))
            if abs(b["amplitude"]) > 1e-12
            else None,
            signed_error=float(abs(estimate.real - b["amplitude"].real)),
            complex_error=float(abs(estimate - b["amplitude"])),
            flux=flux,
            flux_error=float(abs(flux - b["fraction"])),
            phase_angle=measured,
            phase_spread=spread,
            angle_error=abs(measured - central),
        )
    total = sum(b["flux"] for b in branches.values())
    return dict(angle=angle, h=h, branches=branches, flux_sum=total, closure_error=abs(total - 1))
