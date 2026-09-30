"""Free-boundary GLL SEM point-force example; also supports mpiexec -n N."""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def main():
    cfg = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1600, -1600), upper=(1600, 1600), cells=(16, 16)),
            material=dict(density=2200, vp=3000, vs=1700),
            discretization=dict(type="quad_gll", degree=4),
            time=dict(dt=0.001, duration=0.3),
            source=dict(
                position=(13, -17),
                direction=(1, 1),
                wavelet=dict(f0=5, amplitude=1e8, time_shift=0.08),
            ),
            receivers=[dict(name="off_node", position=(413, 283))],
        )
    )
    result = Simulation2D(cfg).run()
    if MPI.COMM_WORLD.rank == 0:
        print("GLL Q4; positive-up z; point line force excites both P and SV.")
        print(
            "Maximum receiver displacement [m]:", np.linalg.norm(result.displacement, axis=2).max()
        )
        print("History shape [time, receiver, x/z]:", result.displacement.shape)


if __name__ == "__main__":
    main()
