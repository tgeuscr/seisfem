"""Independent NumPy angular-spectrum homogeneous P/SV packets (no FEM imports)."""

import numpy as np

EXTENT = 2400.0
TIME = 0.2
FREQUENCY = 5.0
RHO, VP, VS = 2200.0, 3000.0, 1700.0
WIDTH_S, WIDTH_Q = 360.0, 400.0


class Packet:
    def __init__(self, branch, angle, size=128, period=7200.0):
        self.speed = VP if branch == "P" else VS
        theta = np.deg2rad(angle)
        self.n = np.array([np.sin(theta), np.cos(theta)])
        self.transverse = self.n[[1, 0]] * np.array([1, -1])
        self.polarization = self.n if branch == "P" else self.transverse
        self.k0 = 2 * np.pi * FREQUENCY / self.speed * self.n
        self.center = -self.speed * TIME / 2 * self.n
        self.k = 2 * np.pi * np.fft.fftfreq(size, period / size)
        kx, kz = np.meshgrid(self.k, self.k, indexing="ij")
        radius = np.hypot(kx, kz)
        # Gradient/rotated-gradient potential spectrum. Unlike normalizing k at
        # zero, this remains smooth and gives a genuinely localized initial field.
        direction = np.stack((kx, kz), axis=-1) / np.linalg.norm(self.k0)
        if branch == "S":
            direction = direction[..., [1, 0]] * np.array([1, -1])
        ks = kx * self.n[0] + kz * self.n[1] - np.linalg.norm(self.k0)
        kq = kx * self.transverse[0] + kz * self.transverse[1]
        weights = np.exp(-0.5 * ((WIDTH_S * ks) ** 2 + (WIDTH_Q * kq) ** 2))
        self.A = (
            weights[..., None]
            * direction
            * np.exp(-1j * (kx * self.center[0] + kz * self.center[1]))[..., None]
            / weights.sum()
        )
        self.omega = self.speed * radius

    def grid(self, x, z, time=0.0, velocity=False):
        A = self.A * np.exp(-1j * self.omega * time)[..., None]
        if velocity:
            A = A * (-1j * self.omega)[..., None]
        ex, ez = np.exp(1j * np.outer(x, self.k)), np.exp(1j * np.outer(self.k, z))
        return np.stack([(ex @ A[:, :, j] @ ez).real for j in range(2)], axis=-1)

    def points(self, xy, time=0.0, velocity=False):
        # Structured grids have tensor axes, even when rank-local ownership is irregular.
        x, ix = np.unique(np.round(xy[:, 0], 9), return_inverse=True)
        z, iz = np.unique(np.round(xy[:, 1], 9), return_inverse=True)
        return self.grid(x, z, time, velocity)[ix, iz]

    def trace(self, point, time, velocity=False):
        phase = np.exp(1j * (self.k[:, None] * point[0] + self.k[None, :] * point[1]))
        amplitudes = np.sum(self.A * self.polarization, axis=-1) * phase
        if velocity:
            amplitudes = amplitudes * (-1j * self.omega)
        return np.array([np.sum(amplitudes * np.exp(-1j * self.omega * t)).real for t in time])


def peak_time(time, trace):
    i = int(np.argmax(trace))
    if i == 0 or i == len(trace) - 1:
        raise ValueError("Crest lies outside arrival window")
    a, b, c = trace[i - 1 : i + 2]
    return float(time[i] + 0.5 * (a - c) / (a - 2 * b + c) * (time[1] - time[0]))


def spectral_angle(field, spacing, packet):
    size = len(field)
    k = 2 * np.pi * np.fft.fftfreq(size, spacing)
    kx, kz = np.meshgrid(k, k, indexing="ij")
    radius = np.hypot(kx, kz)
    power = np.sum(abs(np.fft.fft2(field, axes=(0, 1))) ** 2, axis=-1)
    mask = (
        (kx * packet.n[0] + kz * packet.n[1] > 0)
        & (radius > 0.5 * np.linalg.norm(packet.k0))
        & (radius < 1.5 * np.linalg.norm(packet.k0))
    )
    power = np.where(mask, power, 0)
    # Vector centroid avoids angular wrap at either signed horizontal direction.
    return float(np.rad2deg(np.arctan2(np.sum(power * kx), np.sum(power * kz))))
