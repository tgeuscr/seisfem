import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import dense

from .evidence import record
from .helpers import MATERIALS, config, edge_matrix


@pytest.mark.parametrize("side", ["left", "right", "lower", "upper"])
@pytest.mark.parametrize("diagonal", ["left", "right", "left_right", "right_left"])
def test_facet_material_tensor_and_row_sums(side, diagonal):
    cfg = config((side,), diagonal)
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        exact = edge_matrix(op.coordinates, cfg)
        np.testing.assert_allclose(dense(op.C), exact, rtol=4e-15, atol=5e-15)
        np.testing.assert_allclose(op.damping, exact.sum(axis=1), rtol=4e-15, atol=5e-15)
        np.testing.assert_allclose(dense(op.C), dense(op.C).T, atol=1e-15)
        assert np.min(op.damping) >= 0
        assert np.linalg.eigvalsh(dense(op.C)).min() > -1e-14
        record(
            f"facet-{side}-{diagonal}",
            dict(
                matrix_error=float(abs(dense(op.C) - exact).max()),
                damping_error=float(abs(op.damping - exact.sum(axis=1)).max()),
                minimum_damping=float(op.damping.min()),
            ),
        )


def test_free_top_and_fixed_intersections():
    cfg = config(constraints=[dict(side="upper", components=["x"])])
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        np.testing.assert_allclose(dense(op.C), edge_matrix(op.coordinates, cfg), atol=5e-15)
        x, z = op.coordinates.T
        top = (abs(z - 3) < 1e-12) & (x > -1 + 1e-12) & (x < 2 - 1e-12)
        assert top.sum() == 5
        np.testing.assert_array_equal(op.damping.reshape(-1, 2)[top], 0)
        assert np.any(op.damping[op.fixed] > 0)
        step = op.start(0.01, u0=np.ones(op.n), v0=np.ones(op.n))
        nxt, v, _, _ = step.evaluate(np.zeros(op.n))
        np.testing.assert_array_equal(nxt[op.fixed], 0)
        np.testing.assert_array_equal(v[op.fixed], 0)
    with pytest.raises(ValueError, match="both absorbing and constrained"):
        config(constraints=[dict(side="right")])


@pytest.mark.parametrize("count", [1, 3])
def test_homogeneous_limit_operators_and_receiver_histories(count):
    material = MATERIALS[0]
    layers = [
        dict(lower=-3 + 6 * i / count, upper=-3 + 6 * (i + 1) / count, material=material)
        for i in range(count)
    ]
    base = config(material=dict(type="layered", layers=layers)).model_dump(
        mode="json", by_alias=True
    )
    base.update(
        time=dict(dt=0.01, duration=2.0),
        source=dict(
            position=(0.17, -0.73),
            direction=(1, -2),
            wavelet=dict(f0=4, amplitude=2, time_shift=0.1),
        ),
        receivers=[dict(name="r", position=(0.43, 0.21))],
    )
    outputs = []
    for mat in [material, dict(type="layered", layers=layers)]:
        with Simulation2D(
            SimulationConfig2D.model_validate(base | dict(material=mat)), MPI.COMM_SELF
        ) as sim:
            result = sim.run()
            op = sim.operators
            outputs.append(
                dict(
                    M=dense(op.M),
                    K=dense(op.K),
                    C=dense(op.C),
                    mass=op.mass.copy(),
                    damping=op.damping.copy(),
                    dt=np.array(op.stable_dt),
                    u=result.displacement,
                    v=result.velocity,
                )
            )
    errors = {}
    for key, a in outputs[0].items():
        b = outputs[1][key]
        error = np.max(abs(a - b)) / max(np.max(abs(a)), 1e-30)
        print("homogeneous limit", count, key, error)
        assert error < 3e-13
        errors[key] = dict(relative_error=float(error), bitwise_equal=bool(np.array_equal(a, b)))
    record(f"homogeneous-{count}", errors)


def test_negative_lambda_still_has_positive_impedance():
    material = dict(density=2.3, **{"lambda": -0.5}, mu=1.2)
    cfg = config(material=dict(type="layered", layers=[dict(lower=-3, upper=3, material=material)]))
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        np.testing.assert_allclose(dense(op.C), edge_matrix(op.coordinates, cfg), atol=5e-15)
        assert op.damping.min() >= 0
