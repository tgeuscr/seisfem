"""Coordinate-ordered SEM packet plus public point-force MPI workflow."""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import SimulationConfig2D
from seisfem.simulation2d import Simulation2D

from .experiment import run
from .helpers import config


def measure(comm):
    packet = run(4, 96, fractions=(0.02,), comm=comm, audit=True)[0]
    data = config(4, cells=(4, 4)).model_dump(mode="json", by_alias=True)
    data.update(
        time=dict(dt=0.001, duration=0.04),
        source=dict(
            position=[0.337, 0.419],
            direction=[3, 4],
            wavelet=dict(f0=12, amplitude=7, time_shift=0.01),
        ),
        receivers=[
            dict(name="near", position=[0.357, 0.462]),
            dict(name="far", position=[0.631, 1.17]),
        ],
    )
    with Simulation2D(SimulationConfig2D.model_validate(data), comm) as sim:
        op = sim.operators
        xy = op.coordinates[: op.n // 2]
        values = np.column_stack((xy[:, 0] ** 2, xy[:, 1] ** 2)).ravel()
        work = comm.allreduce(float(sim.source.spatial_load @ values))
        expected = 0.6 * 0.337**2 + 0.8 * 0.419**2
        assert abs(work - expected) < 3e-14
        result = sim.run()
    return dict(
        packet=packet,
        source_work=work,
        source_work_expected=expected,
        displacement=result.displacement.tolist(),
        velocity=result.velocity.tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = measure(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
