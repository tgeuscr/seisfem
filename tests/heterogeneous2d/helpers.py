import numpy as np

from seisfem import SimulationConfig2D
from seisfem.config2d import PlaneStrainConfig

LOW = dict(density=2.3, vp=3.2, vs=1.8)
HIGH = dict(density=4.1, vp=4.6, vs=2.5)


def layers(lower, interface, upper, first=LOW, second=HIGH):
    return dict(
        type="layered",
        layers=[
            dict(lower=lower, upper=interface, material=first),
            dict(lower=interface, upper=upper, material=second),
        ],
    )


def config(**updates):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-1, -2), upper=(2, 2), cells=(6, 8)),
            material=layers(-2, 0, 2),
        )
        | updates
    )


def public_config():
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1, -2), upper=(2, 2), cells=(18, 24)),
            material=layers(-2, 0, 2),
            time=dict(dt=0.002, duration=1.2),
            source=dict(
                position=(0.413, -0.527),
                direction=(3, -4),
                wavelet=dict(f0=4, amplitude=2.1, time_shift=0.3),
            ),
            receivers=[
                dict(name="above", position=(0.13, 0.74)),
                dict(name="below", position=(0.12, -0.39)),
                dict(name="interface", position=(0, 0)),
            ],
        )
    )


def independent_matrices(op):
    """Analytical triangle integration, classifying vertices without DG0 lookup."""
    M, K = np.zeros((op.n, op.n)), np.zeros((op.n, op.n))
    for cell in range(op.mesh.topology.index_map(2).size_local):
        nodes = op.V.dofmap.cell_dofs(cell)
        coords = op.coordinates[nodes]
        material = LOW if coords[:, 1].max() <= 1e-14 else HIGH
        rho, vp, vs = (material[k] for k in ["density", "vp", "vs"])
        mu = rho * vs**2
        lam = rho * (vp**2 - 2 * vs**2)
        elastic = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
        affine = np.column_stack((np.ones(3), coords))
        area = abs(np.linalg.det(affine)) / 2
        gradients = np.linalg.inv(affine)[1:, :].T
        B = np.zeros((3, 6))
        B[0, 0::2] = gradients[:, 0]
        B[1, 1::2] = gradients[:, 1]
        B[2, 0::2] = gradients[:, 1]
        B[2, 1::2] = gradients[:, 0]
        dofs = (2 * nodes[:, None] + np.arange(2)).ravel()
        M[np.ix_(dofs, dofs)] += rho * area / 12 * np.kron(np.ones((3, 3)) + np.eye(3), np.eye(2))
        K[np.ix_(dofs, dofs)] += area * B.T @ elastic @ B
    return M, K
