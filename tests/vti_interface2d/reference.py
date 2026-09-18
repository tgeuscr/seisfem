"""NumPy-only slowness scattering reference; no seisfem or FEM imports.

u=A d exp[i omega (p x+q z-t)]. Traction strips i omega A.
qP has d.n>0; qSV has d.(n_z,-n_x)>0. Outgoing uses group_z, not q.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    rho: float
    c11: float
    c33: float
    c13: float
    c55: float

    @classmethod
    def isotropic(cls, rho, vp, vs):
        return cls(rho, rho * vp**2, rho * vp**2, rho * (vp**2 - 2 * vs**2), rho * vs**2)


LOWER = Material.isotropic(2200, 3000, 1700)
UPPER = Material(2500, 38e9, 34.225e9, 8e9, 12.1e9)
NAMES = ("RP", "RS", "TP", "TS")


def squared_roots(p, m):
    """Solve the even quartic as a dimensionless quadratic in (q*c0)^2."""
    c0 = np.sqrt(m.c33 / m.rho)
    P = np.asarray(p) * c0
    a, s, b = m.c11 / m.c33, m.c55 / m.c33, m.c13 / m.c33
    aa = s
    bb = (a * P**2 - 1) + s * (s * P**2 - 1) - (b + s) ** 2 * P**2
    cc = (a * P**2 - 1) * (s * P**2 - 1)
    discriminant = bb * bb - 4 * aa * cc
    root = np.sqrt(np.maximum(discriminant, 0))
    # Stable quadratic formula; rejected roots are never used as propagating modes.
    large = (-bb - np.copysign(root, bb)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.stack((large / aa, cc / large), axis=-1) / c0**2
    valid = (discriminant > 1e-14) & np.all(np.isfinite(values) & (values > 1e-20), axis=-1)
    return values, valid


def candidates(p, m):
    values, valid = squared_roots(p, m)
    if not np.all(valid):
        raise ValueError(
            "Reference requires four distinct propagating roots, away from criticality"
        )
    q = np.concatenate((np.sqrt(values), -np.sqrt(values)), axis=-1)
    p = np.broadcast_to(np.asarray(p)[..., None], q.shape)
    D = np.empty(q.shape + (2, 2))
    D[..., 0, 0] = m.c11 * p * p + m.c55 * q * q
    D[..., 1, 1] = m.c55 * p * p + m.c33 * q * q
    D[..., 0, 1] = D[..., 1, 0] = (m.c13 + m.c55) * p * q
    eigen, vectors = np.linalg.eigh(D)
    index = np.argmin(abs(eigen - m.rho), axis=-1)
    d = np.take_along_axis(vectors, index[..., None, None], axis=-1)[..., 0]
    n = np.stack((p, q), axis=-1) / np.hypot(p, q)[..., None]
    target = np.where((index == 1)[..., None], n, n[..., [1, 0]] * np.array([1, -1]))
    d *= np.where(np.sum(d * target, axis=-1) >= 0, 1, -1)[..., None]
    x, z = d[..., 0], d[..., 1]
    group = (
        np.stack(
            (
                (m.c11 * x * x + m.c55 * z * z) * p + (m.c13 + m.c55) * x * z * q,
                (m.c55 * x * x + m.c33 * z * z) * q + (m.c13 + m.c55) * x * z * p,
            ),
            axis=-1,
        )
        / m.rho
    )
    traction = np.stack((m.c55 * (q * x + p * z), m.c13 * p * x + m.c33 * q * z), axis=-1)
    residual = np.linalg.norm(np.einsum("...ij,...j->...i", D, d) - m.rho * d, axis=-1) / m.rho
    determinant = np.linalg.det(D - m.rho * np.eye(2)) / m.rho**2
    return dict(
        p=p,
        q=q,
        d=d,
        n=n,
        g=group,
        t=traction,
        kind=index,
        residual=residual,
        determinant=determinant,
        flux=0.5 * np.sum(traction * d, axis=-1),
    )


def wave(p, m, kind, outgoing):
    data = candidates(p, m)
    selected = (data["kind"] == (1 if kind == "P" else 0)) & (outgoing * data["g"][..., 1] > 0)
    if not np.all(selected.sum(axis=-1) == 1):
        raise ValueError("Reference requires one energy-directed root per eigenbranch")
    index = np.argmax(selected, axis=-1)
    result = {}
    for key, value in data.items():
        if key in ["d", "n", "g", "t"]:
            result[key] = np.take_along_axis(value, index[..., None, None], axis=-2)[..., 0, :]
        else:
            result[key] = np.take_along_axis(value, index[..., None], axis=-1)[..., 0]
    return result


def solve(p, incident="P", lower=LOWER, upper=UPPER):
    inc = wave(p, lower, incident, 1)
    branches = [
        wave(p, m, kind, sign)
        for m, kind, sign in [(lower, "P", -1), (lower, "S", -1), (upper, "P", 1), (upper, "S", 1)]
    ]
    impedance = np.sqrt(lower.rho * lower.c33)

    def state(w):
        return np.concatenate((w["d"], w["t"] / impedance), axis=-1)

    matrix = np.stack(
        [state(w) * (1 if j < 2 else -1) for j, w in enumerate(branches)], axis=-1
    ).astype(complex)
    amplitudes = np.linalg.solve(matrix, -state(inc).astype(complex)[..., None])[..., 0]
    result = {}
    for j, (name, w) in enumerate(zip(NAMES, branches, strict=True)):
        factor = abs(w["flux"]) / inc["flux"]
        result[name] = w | dict(
            amplitude=amplitudes[..., j],
            flux_factor=factor,
            fraction=factor * abs(amplitudes[..., j]) ** 2,
        )
    return inc, result


def residual(p, result, incident="P", lower=LOWER, upper=UPPER):
    """Independently reconstruct displacement gradients, strains and stresses."""
    inc = wave(p, lower, incident, 1)
    terms = [(lower, inc, 1.0)] + [
        (
            lower if name[0] == "R" else upper,
            result[name],
            result[name]["amplitude"] * (1 if name[0] == "R" else -1),
        )
        for name in NAMES
    ]
    displacement = np.zeros(np.shape(p) + (2,), complex)
    traction = displacement.copy()
    for m, w, a in terms:
        gradient = w["d"][..., :, None] * np.stack((np.asarray(p), w["q"]), axis=-1)[..., None, :]
        e = (gradient + np.swapaxes(gradient, -1, -2)) / 2
        stress = np.empty_like(gradient)
        stress[..., 0, 0] = m.c11 * e[..., 0, 0] + m.c13 * e[..., 1, 1]
        stress[..., 1, 1] = m.c13 * e[..., 0, 0] + m.c33 * e[..., 1, 1]
        stress[..., 0, 1] = stress[..., 1, 0] = 2 * m.c55 * e[..., 0, 1]
        displacement += np.asarray(a)[..., None] * w["d"]
        traction += np.asarray(a)[..., None] * stress[..., :, 1]
    return np.concatenate((displacement, traction / np.sqrt(lower.rho * lower.c33)), axis=-1)
