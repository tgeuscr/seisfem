"""Snapshot the legacy P1 workflow using either checkout on PYTHONPATH.

Run this file directly with baseline/src and then feature/src on PYTHONPATH.
The configuration intentionally omits the new selector. Compare every npz array
bitwise; metadata/config serialization is excluded because the new selector is
now explicit in serialized configurations.
"""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def snapshot(destination):
    cfg = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(0, 0), upper=(1, 1.5), cells=(6, 8)),
            material=dict(density=2, vp=3, vs=1.5),
            time=dict(dt=0.001, duration=0.1),
            source=dict(
                position=(0.337, 0.419),
                direction=(3, 4),
                wavelet=dict(f0=12, amplitude=7, time_shift=0.01),
            ),
            receivers=[dict(name="offnode", position=(0.357, 0.462))],
        )
    )
    with Simulation2D(cfg, MPI.COMM_SELF) as sim:
        op = sim.operators
        result = sim.run()
        arrays = dict(
            mass=op.mass,
            M=op.M.getValuesCSR()[2],
            K=op.K.getValuesCSR()[2],
            dt=op.stable_dt,
            source=sim.source(0.037),
            action=op.apply(np.sin(np.arange(op.n))).copy(),
            displacement=result.displacement,
            velocity=result.velocity,
        )
        np.savez(destination, **arrays)


if __name__ == "__main__":
    snapshot(sys.argv[1])
