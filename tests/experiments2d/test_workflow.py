import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.points2d import PointMap2D
from tests.experiments2d.helpers import small_config


def test_public_workflow_timing_and_xarray():
    cfg = small_config()
    simulation = Simulation2D(cfg, MPI.COMM_SELF)
    with simulation as sim:
        assert sim.build() is sim
        result = sim.run()
        # Independent first two recurrence steps verify F(0), F(dt), centered v,
        # and force startup including the nonzero initial Ricker tail.
        op, dt = sim.operators, cfg.time.dt
        b = sim.source.spatial_load
        f0, f1 = cfg.source.wavelet(0) * b, cfg.source.wavelet(dt) * b
        u1 = 0.5 * dt**2 * f0 / op.mass
        u2 = 2 * u1 + dt**2 * (f1 - op.apply(u1)) / op.mass
        points = PointMap2D(op.V, [r.position for r in cfg.receivers])
        np.testing.assert_array_equal(result.displacement[0], 0)
        np.testing.assert_allclose(result.velocity[0], 0, atol=1e-16)
        np.testing.assert_allclose(result.displacement[1], points.evaluate(u1), atol=1e-17)
        np.testing.assert_allclose(result.displacement[2], points.evaluate(u2), atol=1e-17)
        np.testing.assert_allclose(result.velocity[1], points.evaluate(u2 / (2 * dt)), atol=1e-17)
        np.testing.assert_allclose(
            result.velocity[1:-1],
            (result.displacement[2:] - result.displacement[:-2]) / (2 * dt),
            atol=1e-16,
        )
        # Independently advance the final lookahead from the last two owned states.
        step = op.start(dt, force0=f0)
        for t in result.time[:-1]:
            nxt, _, _, _ = step.evaluate(sim.source(t))
            step.advance(nxt)
        final_next = (
            2 * step.current
            - step.previous
            + dt**2 * (sim.source(result.time[-1]) - op.apply(step.current)) / op.mass
        )
        expected_final_v = points.evaluate((final_next - step.previous) / (2 * dt))
        np.testing.assert_allclose(result.velocity[-1], expected_final_v, atol=1e-15)
        repeated = sim.run()
        np.testing.assert_array_equal(repeated.displacement, result.displacement)
    assert simulation.operators is None
    assert result.displacement.shape == result.velocity.shape == (61, 3, 2)
    assert result.receiver_names == ("B", "A", "edge")
    np.testing.assert_array_equal(result.receiver_coordinates, [r.position for r in cfg.receivers])
    assert result.components == ("x", "z")
    ds = result.to_xarray()
    assert ds.displacement.dims == ("time", "receiver", "component")
    assert ds.velocity.attrs["units"] == "m s-1"
    assert ds.receiver.values.tolist() == list(result.receiver_names)
    np.testing.assert_array_equal(ds.receiver_z, result.receiver_coordinates[:, 1])
    assert "acceleration" not in ds


def test_nonzero_startup_at_source_receiver():
    cfg = small_config(receivers=[dict(name="source", position=(0.413, 0.527))])
    with Simulation2D(cfg, MPI.COMM_SELF) as sim:
        result = sim.run()
        expected = sim.receivers.evaluate(0.5 * cfg.time.dt**2 * sim.source(0) / sim.operators.mass)
        assert np.linalg.norm(expected) > 0
        np.testing.assert_allclose(result.displacement[1], expected, rtol=2e-15)
        np.testing.assert_allclose(result.velocity[0], 0, atol=1e-16)


def test_zero_source_empty_receivers_and_constraints():
    cfg = small_config(source=None, receivers=[])
    sim = Simulation2D(cfg, MPI.COMM_SELF)
    assert sim.run().velocity.shape == (61, 0, 2)
    assert sim.operators is None
    cfg = small_config(source=None)
    np.testing.assert_array_equal(Simulation2D(cfg, MPI.COMM_SELF).run().displacement, 0)
    cfg = small_config(
        constraints=[dict(side="left", components=["x"])],
        receivers=[dict(name="wall", position=(0, 0.5))],
    )
    result = Simulation2D(cfg, MPI.COMM_SELF).run()
    np.testing.assert_allclose(result.displacement[:, 0, 0], 0, atol=1e-16)
    np.testing.assert_allclose(result.velocity[:, 0, 0], 0, atol=1e-16)
    assert np.linalg.norm(result.displacement[:, 0, 1]) > 0


def test_unsafe_dt_rejected_and_resources_closed():
    cfg = small_config(time=dict(dt=0.1, duration=0.2))
    sim = Simulation2D(cfg, MPI.COMM_SELF)
    with pytest.raises(ValueError, match="spectral bound"):
        sim.run()
    assert sim.operators is None


@pytest.mark.parametrize(
    "update",
    [
        dict(source=dict(position=(2, 0.5), direction=(1, 0), wavelet=dict(f0=1))),
        dict(source=dict(position=(0.5, 0.5), direction=(0, 0), wavelet=dict(f0=1))),
        dict(receivers=[dict(name="a", position=(np.nan, 0))]),
        dict(receivers=[dict(name="a", position=(0, 0)), dict(name="a", position=(1, 1))]),
        dict(receivers=[dict(name="a", position=(0, 0), quantities=["acceleration"])]),
        dict(time=dict(dt=0.01, duration=0.015)),
    ],
)
def test_invalid_config(update):
    data = small_config().model_dump(mode="json", by_alias=True) | update
    with pytest.raises(ValueError):
        SimulationConfig2D.model_validate(data)
    with pytest.raises(ValueError, match="Invalid 2D simulation config"):
        Simulation2D(data, MPI.COMM_SELF)
