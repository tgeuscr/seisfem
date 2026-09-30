"""NumPy-only welded-interface solution, constructed from full stress tensors.

u=A*d*exp(i*omega*(s.x-t)), dP=n, dSV=(nz,-nx), common normal +z.
Stripped stress is lambda*tr(grad)*I+2*mu*sym(grad), grad=d outer s.
Signed upward time-averaged flux / omega**2 = Re(traction.dot(d*))/2.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Medium:
    rho: float
    vp: float
    vs: float

    def speed(self, mode):
        return self.vp if mode == "P" else self.vs

    def config(self):
        return dict(density=self.rho, vp=self.vp, vs=self.vs)


A = Medium(2200, 3000, 1700)
B = Medium(2500, 4000, 2300)
NAMES = ("RP", "RS", "TP", "TS")


def wave(m, mode, p, sign):
    p = np.asarray(p)
    c = m.speed(mode)
    if np.any(abs(c * p) >= 1):
        raise ValueError("All branches must be strictly propagating, below criticality")
    n = np.stack((c * p, sign * np.sqrt(1 - (c * p) ** 2)), axis=-1)
    d = n if mode == "P" else n[..., [1, 0]] * [1, -1]
    gradient = d[..., :, None] * (n / c)[..., None, :]
    strain = (gradient + np.swapaxes(gradient, -1, -2)) / 2
    mu = m.rho * m.vs**2
    lam = m.rho * (m.vp**2 - 2 * m.vs**2)
    stress = (
        lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(2) + 2 * mu * strain
    )
    traction = stress[..., :, 1]
    flux = np.sum(traction * d, axis=-1) / 2
    return n, d, traction, flux


def solve(mode, p, lower=A, upper=B):
    inc = wave(lower, mode, p, 1)
    waves = [
        wave(m, kind, p, sign)
        for m, kind, sign in [(lower, "P", -1), (lower, "S", -1), (upper, "P", 1), (upper, "S", 1)]
    ]
    columns = [
        np.concatenate((w[1], w[2]), axis=-1) * (1 if j < 2 else -1) for j, w in enumerate(waves)
    ]
    scale = np.array([1, 1, 1 / (lower.rho * lower.vp), 1 / (lower.rho * lower.vp)])
    matrix = np.stack(columns, axis=-1) * scale[:, None]
    rhs = -np.concatenate((inc[1], inc[2]), axis=-1) * scale
    amplitudes = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    return inc, waves, amplitudes


def report(mode, angle, lower=A, upper=B):
    p = np.sin(np.deg2rad(angle)) / lower.speed(mode)
    inc, waves, amplitudes = solve(mode, p, lower, upper)
    jump = np.r_[inc[1], inc[2]].copy()
    branches = {}
    for j, (name, w, a) in enumerate(zip(NAMES, waves, amplitudes, strict=True)):
        jump += (1 if j < 2 else -1) * a * np.r_[w[1], w[2]]
        factor = abs(w[3]) / inc[3]
        branches[name] = dict(
            direction=w[0].tolist(),
            polarization=w[1].tolist(),
            angle=float(np.degrees(np.arctan2(w[0][0], abs(w[0][1])))),
            amplitude=float(a),
            flux_factor=float(factor),
            flux=float(factor * a * a),
        )
    jump[2:] /= lower.rho * lower.vp
    return dict(
        mode=mode,
        angle=angle,
        p=float(p),
        lower=asdict(lower),
        upper=asdict(upper),
        branches=branches,
        residual=float(max(abs(jump))),
        closure=abs(sum(b["flux"] for b in branches.values()) - 1),
    )
