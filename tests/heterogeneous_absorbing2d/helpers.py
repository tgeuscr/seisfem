import numpy as np

from seisfem.config2d import PlaneStrainConfig

MATERIALS = [
    dict(density=2.3, vp=3.2, vs=1.8),
    dict(density=4.1, vp=4.6, vs=2.5),
    dict(density=1.9, vp=2.8, vs=1.4),
]


def config(sides=("left", "right", "lower"), diagonal="right", **updates):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-1, -3), upper=(2, 3), cells=(6, 12), diagonal=diagonal),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-3 + 2 * i, upper=-1 + 2 * i, material=m)
                    for i, m in enumerate(MATERIALS)
                ],
            ),
            boundaries={side: "absorbing" for side in sides},
        )
        | updates
    )


def edge_matrix(coordinates, cfg):
    """Independent exact P1 edge integration; material from physical edge midpoint.

    Uses user rho/Vp/Vs, not DG0 fields or the production impedance expression.
    Shared interface vertices receive both edge contributions.
    """
    matrix = np.zeros((2 * len(coordinates), 2 * len(coordinates)))
    sides = dict(
        left=(0, cfg.domain.lower[0]),
        right=(0, cfg.domain.upper[0]),
        lower=(1, cfg.domain.lower[1]),
        upper=(1, cfg.domain.upper[1]),
    )
    for side in cfg.boundaries.absorbing_sides:
        normal, value = sides[side]
        tangent = 1 - normal
        nodes = np.flatnonzero(abs(coordinates[:, normal] - value) < 1e-12)
        nodes = nodes[np.argsort(coordinates[nodes, tangent])]
        for a, b in zip(nodes[:-1], nodes[1:], strict=True):
            z = (coordinates[a, 1] + coordinates[b, 1]) / 2
            material = next(
                layer.material
                for layer in cfg.material.layers
                if layer.lower - 1e-12 <= z <= layer.upper + 1e-12
            )
            length = np.linalg.norm(coordinates[b] - coordinates[a])
            for component in [0, 1]:
                speed = material.speed("P" if component == normal else "S")
                ids = [2 * a + component, 2 * b + component]
                matrix[np.ix_(ids, ids)] += (
                    material.density * speed * length / 6 * np.array([[2, 1], [1, 2]])
                )
    return matrix
