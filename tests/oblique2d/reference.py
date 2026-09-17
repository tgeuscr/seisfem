"""Independent continuum reference: NumPy only, no seisfem/FEM imports.

u = A d exp(i omega (p x + q z - t)); d_P=n, d_S=(n_z,-n_x).
Tractions below omit the common i*omega. Rows are impedance-scaled.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    rho: float
    vp: float
    vs: float

    def speed(self, mode):
        return self.vp if mode == "P" else self.vs


LOWER = Material(2200, 3000, 1700)
UPPER = Material(2500, 4000, 2300)
NAMES = ("RP", "RS", "TP", "TS")


def wave(material, mode, p, sign):
    c = material.speed(mode)
    if abs(p * c) >= 1:
        raise ValueError("Reference is restricted to propagating, below-critical waves")
    n = np.array([p * c, sign * np.sqrt(1 - (p * c) ** 2)])
    d = n if mode == "P" else np.array([n[1], -n[0]])
    slowness = n / c
    mu = material.rho * material.vs**2
    lam = material.rho * (material.vp**2 - 2 * material.vs**2)
    traction = np.array(
        [
            mu * (slowness[1] * d[0] + slowness[0] * d[1]),
            lam * np.dot(slowness, d) + 2 * mu * slowness[1] * d[1],
        ]
    )
    return n, d, traction


def solve(mode, angle=None, *, p=None, lower=LOWER, upper=UPPER):
    if p is None:
        p = np.sin(np.deg2rad(angle)) / lower.speed(mode)
    ni, di, ti = wave(lower, mode, p, 1)
    branches = [(lower, "P", -1), (lower, "S", -1), (upper, "P", 1), (upper, "S", 1)]
    waves = [wave(m, s, p, sign) for m, s, sign in branches]
    scale = np.array([1, 1, 1 / (lower.rho * lower.vp), 1 / (lower.rho * lower.vp)])
    columns = [np.r_[d, t] * (1 if j < 2 else -1) for j, (_, d, t) in enumerate(waves)]
    matrix = np.column_stack(columns) * scale[:, None]
    a = np.linalg.solve(matrix, -np.r_[di, ti] * scale)
    result = {}
    for name, amplitude, (mat, kind, _), (n, d, _t) in zip(NAMES, a, branches, waves, strict=True):
        factor = mat.rho * mat.speed(kind) * abs(n[1]) / (lower.rho * lower.speed(mode) * ni[1])
        result[name] = dict(
            amplitude=float(amplitude),
            flux=float(factor * amplitude**2),
            flux_factor=float(factor),
            direction=n.tolist(),
            polarization=d.tolist(),
            angle=float(np.rad2deg(np.arcsin(n[0]))),
        )
    return result


def residual(mode, angle, result, lower=LOWER, upper=UPPER):
    """Rebuild full stress tensors independently of the traction-column assembly."""
    p = np.sin(np.deg2rad(angle)) / lower.speed(mode)
    jump_u, jump_t = np.zeros(2), np.zeros(2)
    terms = [(lower, mode, 1, 1.0, 1)] + [
        (mat, kind, sign, result[name]["amplitude"], side)
        for name, mat, kind, sign, side in [
            ("RP", lower, "P", -1, 1),
            ("RS", lower, "S", -1, 1),
            ("TP", upper, "P", 1, -1),
            ("TS", upper, "S", 1, -1),
        ]
    ]
    for mat, kind, sign, amplitude, side in terms:
        c = mat.speed(kind)
        n = np.array([p * c, sign * np.sqrt(1 - (p * c) ** 2)])
        d = n if kind == "P" else np.array([n[1], -n[0]])
        gradient = np.outer(d, n / c)
        mu, lam = mat.rho * mat.vs**2, mat.rho * (mat.vp**2 - 2 * mat.vs**2)
        stress = lam * np.trace(gradient) * np.eye(2) + mu * (gradient + gradient.T)
        jump_u += side * amplitude * d
        jump_t += side * amplitude * stress[:, 1]
    return np.r_[jump_u, jump_t / (lower.rho * lower.vp)]
