"""Gaussian potential and independent angular-spectrum interface templates."""

from dataclasses import dataclass

import numpy as np

from .reference import NAMES, A, B, report, solve


@dataclass(frozen=True)
class Packet:
    mode: str = "P"
    reverse: bool = False
    identical: bool = False
    width: float = 1500.0

    @property
    def lower(self):
        return B if self.reverse else A

    @property
    def upper(self):
        return self.lower if self.identical else (A if self.reverse else B)

    @property
    def angle(self):
        return 25.0 if self.mode == "P" else 15.0

    @property
    def time(self):
        # Fix lower-medium path length and post-interface propagation distance.
        return (
            (2.25 if self.mode == "P" else 3.1) * A.speed(self.mode) / self.lower.speed(self.mode)
        )

    frequency = 5.0
    center = np.array([-1400.0, -3000.0])
    analysis = 7200.0
    sampling = 40.0

    @property
    def speed(self):
        return self.lower.speed(self.mode)

    @property
    def n(self):
        a = np.deg2rad(self.angle)
        return np.array([np.sin(a), np.cos(a)])

    @property
    def transverse(self):
        return self.n[[1, 0]] * [1, -1]

    @property
    def sigma(self):
        return 0.6 * self.speed / self.frequency

    @property
    def k0(self):
        return 2 * np.pi * self.frequency / self.speed

    def initial(self, xy):
        s, q = (xy - self.center) @ self.n, (xy - self.center) @ self.transverse
        g = np.exp(-0.5 * ((s / self.sigma) ** 2 + (q / self.width) ** 2)) / self.k0
        a, b = -s / self.sigma**2, -q / self.width**2
        cs, sn = np.cos(self.k0 * s), np.sin(self.k0 * s)
        fs, fq = g * (a * cs - self.k0 * sn), g * b * cs
        fss = g * ((a * a - 1 / self.sigma**2 - self.k0**2) * cs - 2 * a * self.k0 * sn)
        fsq = g * b * (a * cs - self.k0 * sn)
        u = fs[:, None] * self.n + fq[:, None] * self.transverse
        v = -self.speed * (fss[:, None] * self.n + fsq[:, None] * self.transverse)
        if self.mode == "S":
            u, v = u[:, [1, 0]] * [1, -1], v[:, [1, 0]] * [1, -1]
        return u, v

    def transform(self, kx, kz):
        radius = np.hypot(kx, kz)
        ks = kx * self.n[0] + kz * self.n[1]
        kq = kx * self.transverse[0] + kz * self.transverse[1]
        phi = (
            np.pi
            * self.sigma
            * self.width
            / self.k0
            * np.exp(-0.5 * (self.width * kq) ** 2)
            * (
                np.exp(-0.5 * (self.sigma * (ks - self.k0)) ** 2)
                + np.exp(-0.5 * (self.sigma * (ks + self.k0)) ** 2)
            )
        )
        return (
            1j
            * radius
            * phi
            * np.exp(-1j * (kx * self.center[0] + kz * self.center[1]))
            * (1 + ks / np.maximum(radius, 1e-30))
            / 2
        )

    def geometry(self, extent):
        ref = report(self.mode, self.angle, self.lower, self.upper)
        hit_time = -self.center[1] / (self.speed * self.n[1])
        hit = self.center + self.speed * hit_time * self.n
        for name, b in ref["branches"].items():
            c = (self.lower if name[0] == "R" else self.upper).speed(name[1])
            b["center"] = (hit + c * (self.time - hit_time) * np.array(b["direction"])).tolist()
            b["arrival"] = self.time
        # Five-sigma ellipse, not an assertion of compact Gaussian support.
        radii = 5 * np.sqrt((self.sigma * self.n) ** 2 + (self.width * self.transverse) ** 2)
        lo, hi = self.center - radii, self.center + radii
        returns = np.r_[2 * extent + lo - self.analysis, 2 * extent - hi - self.analysis] / max(
            self.lower.vp, self.upper.vp
        )
        return dict(
            reference=ref,
            hit=hit.tolist(),
            hit_time=hit_time,
            initial_five_sigma_box=[lo.tolist(), hi.tolist()],
            exterior_return_times=dict(
                zip(["left", "bottom", "right", "top"], returns.tolist(), strict=True)
            ),
            return_margin=float(min(returns) - self.time),
            gaussian_envelope_at_cut=float(np.exp(-12.5)),
        )

    def spectra(self):
        """Outgoing Fourier densities at t=0, in output-wavevector coordinates."""
        h = self.sampling
        size = round(2 * self.analysis / h) + 1
        k = 2 * np.pi * np.fft.fftfreq(size, h)
        kx, kz = np.meshgrid(k, k)
        r = np.hypot(kx, kz)
        result = {}
        for j, name in enumerate(NAMES):
            sign = -1 if name[0] == "R" else 1
            c = (self.lower if sign < 0 else self.upper).speed(name[1])
            w = c * r
            kiz = np.sqrt(np.maximum((w / self.speed) ** 2 - kx * kx, 0))
            inc = self.transform(kx, kiz)
            valid = (
                (sign * kz > 0)
                & (w > max(self.lower.vp, self.upper.vp) * abs(kx))
                & (abs(inc) > 1e-11 * abs(inc).max())
            )
            wi, wo, a = solve(self.mode, kx[valid] / w[valid], self.lower, self.upper)
            J = self.speed * wi[0][:, 1] / (c * abs(wo[j][0][:, 1]))
            value = inc[valid] * a[:, j] / J
            result[name] = (valid, value[:, None] * wo[j][1], w[valid], kx[valid], kz[valid])
        return result

    def templates(self):
        """Exact finite-bandwidth fields; J=|dkz_out/dkz_in|=ci*nzi/(co*|nzo|).

        No FEM data, fitted direction, delay or beam width enters synthesis.
        """
        h = self.sampling
        size = round(2 * self.analysis / h) + 1
        fields = {}
        for name, (valid, value, w, kx, kz) in self.spectra().items():
            phase = np.exp(-1j * (w * self.time + self.analysis * (kx + kz)))
            spectrum = np.zeros((size, size, 4), complex)
            spectrum[valid, :2] = value * phase[:, None]
            spectrum[valid, 2:] = -1j * w[:, None] * value * phase[:, None]
            fields[name] = 2 * np.fft.ifft2(spectrum, axes=(0, 1)).real / h**2
        return fields

    def traces(self, positions, times):
        """Independent continuum receiver histories, summed per physical half-space."""
        size = round(2 * self.analysis / self.sampling) + 1
        normalization = 2 / (size * self.sampling) ** 2
        output = np.zeros((len(times), len(positions), 4))
        for name, (_, value, w, kx, kz) in self.spectra().items():
            phase = np.exp(-1j * np.outer(times, w))
            for j, point in enumerate(positions):
                if (point[1] < 0) != (name[0] == "R"):
                    continue
                spatial = np.exp(1j * (kx * point[0] + kz * point[1]))
                output[:, j, :2] += normalization * (phase @ (spatial[:, None] * value)).real
                output[:, j, 2:] += (
                    normalization * (phase @ (-1j * w[:, None] * spatial[:, None] * value)).real
                )
        return output

    def bandwidth(self):
        ks, kq = np.meshgrid(
            self.k0 + np.linspace(-5, 5, 81) / self.sigma, np.linspace(-5, 5, 81) / self.width
        )
        kx = ks * self.n[0] + kq * self.transverse[0]
        kz = ks * self.n[1] + kq * self.transverse[1]
        r = np.hypot(kx, kz)
        weight = (self.speed * r) ** 2 * abs(self.transform(kx, kz)) ** 2
        valid = (kz > 0) & (max(self.lower.vp, self.upper.vp) * abs(kx) < self.speed * r)
        excluded = weight[~valid].sum() / weight.sum()
        theta = np.degrees(np.arctan2(kx, kz))
        mean = np.sum(theta * weight) / weight.sum()
        wi, wo, a = solve(self.mode, kx[valid] / (self.speed * r[valid]), self.lower, self.upper)
        flux = [
            float(
                np.sum(weight[valid] * abs(wo[j][3]) / wi[3] * a[:, j] ** 2) / weight[valid].sum()
            )
            for j in range(4)
        ]
        return dict(
            angle=float(mean),
            spread=float(np.sqrt(np.sum(weight * (theta - mean) ** 2) / weight.sum())),
            excluded_energy=float(excluded),
            integrated_flux=dict(zip(NAMES, flux, strict=True)),
        )
