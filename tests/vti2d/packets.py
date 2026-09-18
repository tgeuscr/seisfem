"""Test-only exact Christoffel-spectrum packets and Fourier phase diagnostics."""

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators

from .helpers import RHO, VTI, C
from .reference import mode

EXTENT = 2400.0
WIDTH = 400.0
TIME = 0.2


def spectrum(branch, indices, h, extent=EXTENT):
    size = round(2 * extent / h)
    freq = 2 * np.pi * np.fft.fftfreq(size, h)
    x, z = np.meshgrid(freq, freq, indexing="ij")
    k = np.stack((x, z), axis=-1)
    k0 = 2 * np.pi * np.array(indices) / (2 * EXTENT)
    omega, d, g = mode(k, RHO, C, branch)
    speed, pol, group = mode(k0 / np.linalg.norm(k0), RHO, C, branch)
    d *= np.where(d @ pol >= 0, 1, -1)[..., None]
    weight = np.exp(-0.5 * WIDTH**2 * np.sum((k - k0) ** 2, axis=-1))
    # The FFT origin is at the negative corner; center the beam at physical zero.
    amplitudes = weight[..., None] * d * np.exp(1j * extent * (x + z))[..., None]
    amplitudes *= size**2 / np.sum(weight)
    integrated_group = np.sum(weight[..., None] ** 2 * g, axis=(0, 1)) / np.sum(weight**2)
    return (
        amplitudes,
        omega,
        dict(
            speed=float(speed),
            polarization=pol,
            group=group,
            integrated_group=integrated_group,
            k0=k0,
        ),
    )


def field(amplitudes, omega, time=0, velocity=False):
    coefficients = amplitudes * np.exp(-1j * omega * time)[..., None]
    if velocity:
        coefficients *= (-1j * omega)[..., None]
    return np.fft.ifft2(coefficients, axes=(0, 1)).real


def centroid(u, h, extent):
    weights = np.sum(u * u, axis=-1)
    coordinates = np.arange(len(u)) * h - extent
    return (
        np.array([np.sum(weights * coordinates[:, None]), np.sum(weights * coordinates[None, :])])
        / weights.sum()
    )


def run(branch, indices, h, extent=EXTENT):
    A, omega, reference = spectrum(branch, indices, h, extent)
    n = len(A)
    initial = field(A, omega)
    velocity = field(A, omega, velocity=True)
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-extent, -extent), upper=(extent, extent), cells=(n, n)),
            material=VTI,
        )
    )
    grid_indices = tuple(np.rint(reference["k0"] * 2 * extent / (2 * np.pi)).astype(int) % n)
    coefficients = []
    times = []
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        positions = np.rint((op.coordinates + extent) / h).astype(int)
        i, j = (positions % n).T
        step_count = int(np.ceil(TIME / (0.5 * op.stable_dt)))
        dt = TIME / step_count
        step = op.start(dt, u0=initial[i, j].ravel(), v0=velocity[i, j].ravel())
        unique = np.all(positions < n, axis=1)

        def grid(u):
            result = np.zeros_like(initial)
            result[i[unique], j[unique]] = u.reshape(-1, 2)[unique]
            return result

        for tick in range(step_count + 1):
            u = grid(step.current)
            transform = np.fft.fft2(u, axes=(0, 1))
            coefficients.append(transform[grid_indices])
            times.append(tick * dt)
            if tick < step_count:
                nxt, _, _, _ = step.evaluate(np.zeros(op.n))
                step.advance(nxt)
        final = u
    coefficients = np.array(coefficients)
    projected = coefficients @ reference["polarization"]
    phase = np.unwrap(np.angle(projected))
    measured_omega = -np.polyfit(times, phase, 1)[0]
    phase_error = float(
        abs(measured_omega / (reference["speed"] * np.linalg.norm(reference["k0"])) - 1)
    )
    perpendicular = reference["polarization"][[1, 0]] * np.array([1, -1])
    leakage = float(np.max(abs(coefficients @ perpendicular)) / np.max(abs(projected)))
    exact = field(A, omega, TIME)
    relative_field = float(np.linalg.norm(final - exact) / np.linalg.norm(exact))
    displacement = (centroid(final, h, extent) - centroid(initial, h, extent)) / TIME
    continuum_displacement = (centroid(exact, h, extent) - centroid(initial, h, extent)) / TIME
    group_error = float(
        np.linalg.norm(displacement - reference["integrated_group"])
        / np.linalg.norm(reference["integrated_group"])
    )
    spectrum_power = np.sum(abs(np.fft.fft2(final, axes=(0, 1))) ** 2, axis=-1)
    freq = np.fft.fftfreq(n, h)
    # Resolve the real-field +/-k ambiguity using only the outgoing hemisphere.
    kx, kz = np.meshgrid(freq, freq, indexing="ij")
    hemisphere = (kz > 0) if indices[1] else (kx > 0)
    peak = np.unravel_index(
        np.argmax(np.where(hemisphere, spectrum_power, 0)), spectrum_power.shape
    )
    measured_angle = float(np.rad2deg(np.arctan2(kx[peak], kz[peak])))
    return dict(
        branch=branch,
        indices=list(indices),
        h=h,
        dt=dt,
        phase_speed=float(measured_omega / np.linalg.norm(reference["k0"])),
        reference_speed=reference["speed"],
        phase_error=phase_error,
        polarization_leakage=leakage,
        field_error=relative_field,
        phase_angle=measured_angle,
        reference_angle=float(np.rad2deg(np.arctan2(*reference["k0"]))),
        group_velocity=displacement.tolist(),
        reference_group=reference["group"].tolist(),
        spectrum_group=reference["integrated_group"].tolist(),
        continuum_group=continuum_displacement.tolist(),
        group_error=group_error,
    )
