import numpy as np
import pytest
from mpi4py import MPI

from seisfem.config2d import ForceSource2D, PlaneStrainConfig, SimulationConfig2D
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from seisfem.simulation2d import Simulation2D
from seisfem.sources2d import PointForce2D
from tests.plane_strain.helpers import dense

from .helpers import config


@pytest.mark.parametrize(
    "bad",
    [
        dict(discretization=dict(type="quad_gll", degree=0)),
        dict(discretization=dict(type="quad_gll", degree=7)),
        dict(discretization=dict(type="quad_gll", degree=2.5)),
        dict(boundaries=dict(left="pml")),
        dict(constraints=[dict(side="left")]),
        dict(material=dict(type="vti", density=2, c11=18, c33=18, c13=9, c55=4.5)),
        dict(
            material=dict(
                type="layered",
                layers=[
                    dict(lower=0, upper=0.4, material=dict(density=2, vp=3, vs=1.5)),
                    dict(lower=0.4, upper=1.5, material=dict(density=3, vp=4, vs=2)),
                ],
            )
        ),
    ],
)
def test_scope_rejection(bad):
    data = config().model_dump(mode="json", by_alias=True)
    data.update(bad)
    with pytest.raises(ValueError):
        PlaneStrainConfig.model_validate(data)


def test_tri_default_is_exactly_the_explicit_tri_path():
    data = config().model_dump(mode="json", by_alias=True)
    data.pop("discretization")
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as a:
        data["discretization"] = dict(type="tri_p1")
        with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as b:
            for ka, kb in [(a.M, b.M), (a.K, b.K)]:
                np.testing.assert_array_equal(dense(ka), dense(kb))
            np.testing.assert_array_equal(a.mass, b.mass)
            assert a.stable_dt == b.stable_dt


@pytest.mark.parametrize("p", [1, 2, 4, 6])
def test_polynomial_receivers_and_exact_point_load(p):
    with PlaneStrainOperators(config(p), MPI.COMM_SELF) as op:
        positions = np.array([[0.231, 0.317], [0.5, 0.5], [1, 1.5], [0, 0]])
        points = PointMap2D(op.V, positions)
        xy = op.coordinates[: op.n // 2]

        def polynomial(x):
            return np.column_stack((1 + x[:, 0] ** p * x[:, 1], x[:, 1] ** p - 2 * x[:, 0]))

        values = polynomial(xy).ravel()
        np.testing.assert_allclose(points.evaluate(values), polynomial(positions), atol=5e-13)
        load = points.unit_load((3, 4))
        np.testing.assert_allclose(
            load @ values, np.sum(polynomial(positions) @ np.array([0.6, 0.8])), atol=5e-13
        )
        np.testing.assert_allclose(
            load.reshape(-1, 2).sum(axis=0), 4 * np.array([0.6, 0.8]), atol=5e-14
        )
        source = PointForce2D(
            op.V,
            ForceSource2D.model_validate(
                dict(
                    position=positions[0],
                    direction=[3, 4],
                    wavelet=dict(f0=2, amplitude=7, time_shift=0.1),
                )
            ),
        )
        np.testing.assert_allclose(
            source(0.1) @ values,
            7 * polynomial(positions[:1])[0] @ np.array([0.6, 0.8]),
            atol=1e-12,
        )
        assert PointMap2D(op.V, []).evaluate(values).shape == (0, 2)


def test_public_simulation_time_history():
    data = config(4, cells=(3, 3)).model_dump(mode="json", by_alias=True)
    data.update(
        time=dict(dt=0.001, duration=0.02),
        source=dict(position=[0.33, 0.41], direction=[0, 1], wavelet=dict(f0=10, amplitude=1)),
        receivers=[dict(name="offnode", position=[0.37, 0.48])],
    )
    cfg = SimulationConfig2D.model_validate(data)
    result = Simulation2D(cfg, MPI.COMM_SELF).run()
    assert result.displacement.shape == result.velocity.shape == (21, 1, 2)
    assert np.isfinite(result.displacement).all()
    assert np.linalg.norm(result.displacement) > 0
    assert result.metadata["source_units"] == "N m-1"
