import importlib.util
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D

from .evidence import record


def example_config():
    path = Path(__file__).resolve().parents[2] / "examples/2d/layered_absorbing.py"
    spec = importlib.util.spec_from_file_location("layered_absorbing_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example.configuration()


def test_public_surface_well_example_and_free_surface():
    cfg = example_config()
    with Simulation2D(cfg, MPI.COMM_SELF) as sim:
        a = sim.run()
        op = sim.operators
        x, z = op.coordinates[: op.n // 2].T
        top = (abs(z) < 1e-10) & (abs(x) < 1199)
        assert top.sum() == 59
        np.testing.assert_array_equal(op.damping.reshape(-1, 2)[top], 0)
        assert op.damping.max() > 0
    data = cfg.model_dump(mode="json", by_alias=True)
    # Change the physical free surface only; it must affect surface records.
    data["boundaries"]["upper"] = "absorbing"
    b = Simulation2D(data, MPI.COMM_SELF).run()
    assert a.displacement.shape == (1201, 5, 2)
    assert np.all(np.isfinite(a.displacement))
    surface_difference = np.linalg.norm(
        a.displacement[:, :2] - b.displacement[:, :2]
    ) / np.linalg.norm(a.displacement[:, :2])
    assert surface_difference > 0.1
    record(
        "example",
        dict(
            displacement_peaks=np.max(abs(a.displacement), axis=0).tolist(),
            velocity_peaks=np.max(abs(a.velocity), axis=0).tolist(),
            surface_difference=float(surface_difference),
        ),
    )
