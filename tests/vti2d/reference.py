"""NumPy-only Christoffel reference; no seisfem imports or FEM code."""

import numpy as np


def symbol(k, c):
    x, z = np.moveaxis(np.asarray(k), -1, 0)
    c11, c33, c13, c55 = c
    out = np.empty(np.shape(x) + (2, 2))
    out[..., 0, 0] = c11 * x * x + c55 * z * z
    out[..., 1, 1] = c55 * x * x + c33 * z * z
    out[..., 0, 1] = out[..., 1, 0] = (c13 + c55) * x * z
    return out


def mode(k, rho, c, branch):
    """Return omega, eigenpolarization and group velocity d omega / d k.

    qP is the faster eigenbranch, qSV the slower. Eigenvector sign is arbitrary.
    At k=0 group direction is undefined; return zero for spectral synthesis.
    """
    k = np.asarray(k)
    eigenvalues, vectors = np.linalg.eigh(symbol(k, c))
    j = 1 if branch == "P" else 0
    omega = np.sqrt(eigenvalues[..., j] / rho)
    d = vectors[..., :, j]
    x, z = np.moveaxis(k, -1, 0)
    a, b = np.moveaxis(d, -1, 0)
    c11, c33, c13, c55 = c
    derivative = np.stack(
        (
            2 * x * (c11 * a * a + c55 * b * b) + 2 * (c13 + c55) * z * a * b,
            2 * z * (c55 * a * a + c33 * b * b) + 2 * (c13 + c55) * x * a * b,
        ),
        axis=-1,
    )
    group = derivative / (2 * rho * np.where(omega > 0, omega, 1))[..., None]
    return omega, d, group
