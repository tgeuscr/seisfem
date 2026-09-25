"""NumPy-only tensor Lagrange quadrature, independent of Basix and seisfem."""

import numpy as np
from numpy.polynomial.legendre import Legendre


def gll(p):
    polynomial = Legendre.basis(p)
    roots = polynomial.deriv().roots()
    assert np.max(abs(roots.imag), initial=0) < 1e-14
    x = np.r_[-1.0, roots.real, 1.0]
    w = 2 / (p * (p + 1) * polynomial(x) ** 2)
    return (x + 1) / 2, w / 2


def basis(nodes, x):
    values, gradients = [], []
    for j in range(len(nodes)):
        roots = np.delete(nodes, j)
        poly = np.polynomial.Polynomial.fromroots(roots)
        poly /= poly(nodes[j])
        values.append(poly(x))
        gradients.append(poly.deriv()(x))
    return np.array(values).T, np.array(gradients).T


def matrices(p, xy, rho=2.0, lam=9.0, mu=4.5, gaussian=False):
    """Physical rectangle element matrices in supplied physical nodal ordering."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    lengths = hi - lo
    nodes, weights = gll(p)
    q = nodes
    if gaussian:
        q, weights = np.polynomial.legendre.leggauss(p + 1)
        q, weights = (q + 1) / 2, weights / 2
    L, dL = basis(nodes, q)
    indices = np.argmin(abs((xy - lo)[:, :, None] / lengths[None, :, None] - nodes), axis=-1)
    n = len(xy)
    M, K = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n))
    C = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
    for i in range(len(q)):
        for j in range(len(q)):
            shape = L[i, indices[:, 0]] * L[j, indices[:, 1]]
            dx = dL[i, indices[:, 0]] * L[j, indices[:, 1]] / lengths[0]
            dz = L[i, indices[:, 0]] * dL[j, indices[:, 1]] / lengths[1]
            B = np.zeros((3, 2 * n))
            B[0, ::2], B[1, 1::2] = dx, dz
            B[2, ::2], B[2, 1::2] = dz, dx
            weight = weights[i] * weights[j] * np.prod(lengths)
            M += rho * weight * np.kron(np.outer(shape, shape), np.eye(2))
            K += weight * B.T @ C @ B
    return M, K


def layered_matrices(p, xy, cells, layers):
    """Scatter independent rectangle integrals; classify by physical vertices.

    layers contains (lower, upper, rho, vp, vs), without FEM material fields.
    Cell connectivity is an input, not a source of material IDs or coefficients.
    """
    M = np.zeros((2 * len(xy), 2 * len(xy)))
    K = np.zeros_like(M)
    for nodes in cells:
        points = xy[nodes]
        matches = [
            row
            for row in layers
            if points[:, 1].min() >= row[0] - 1e-13 and points[:, 1].max() <= row[1] + 1e-13
        ]
        assert len(matches) == 1
        _, _, rho, vp, vs = matches[0]
        mu, lam = rho * vs**2, rho * (vp**2 - 2 * vs**2)
        a, b = matrices(p, points, rho, lam, mu)
        ids = (2 * nodes[:, None] + np.arange(2)).ravel()
        M[np.ix_(ids, ids)] += a
        K[np.ix_(ids, ids)] += b
    return M, K
