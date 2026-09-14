import numpy as np
from petsc4py import PETSc

from seisfem.config2d import PlaneStrainConfig


def configuration(n=3, diagonal="right", fixed=False, lam=1.7):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(cells=(n, n), diagonal=diagonal),
            material=dict(density=2.3, **{"lambda": lam}, mu=1.2),
            constraints=[dict(side=side) for side in ("left", "right", "lower", "upper")]
            if fixed
            else [],
        )
    )


def dense(matrix):
    indices = np.arange(matrix.getSize()[0], dtype=PETSc.IntType)
    return matrix.getValues(indices, indices)


def hand_matrices(op):
    """Independent triangle B-matrix integration in actual local node ordering (serial)."""
    M, K = np.zeros((op.n, op.n)), np.zeros((op.n, op.n))
    rho = op.config.material.density
    # Tests supply direct Lamé input; do not call the production conversion here.
    lam, mu = op.config.material.lambda_, op.config.material.mu
    C = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
    for cell in range(op.mesh.topology.index_map(2).size_local):
        nodes = op.V.dofmap.cell_dofs(cell)
        xy = op.coordinates[nodes]
        affine = np.column_stack((np.ones(3), xy))
        area = abs(np.linalg.det(affine)) / 2
        gradients = np.linalg.inv(affine)[1:, :].T
        B = np.zeros((3, 6))
        B[0, 0::2] = gradients[:, 0]
        B[1, 1::2] = gradients[:, 1]
        B[2, 0::2] = gradients[:, 1]
        B[2, 1::2] = gradients[:, 0]
        indices = (2 * nodes[:, None] + np.arange(2)).ravel()
        K[np.ix_(indices, indices)] += area * B.T @ C @ B
        M[np.ix_(indices, indices)] += (
            rho * area / 12 * np.kron(np.ones((3, 3)) + np.eye(3), np.eye(2))
        )
    return M, K
