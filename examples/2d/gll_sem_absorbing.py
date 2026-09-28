"""Off-node line force with a free top and three absorbing SEM sides.

Run: python examples/2d/gll_sem_absorbing.py
The same boundary API also accepts element-aligned isotropic layered material.
"""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D

config = SimulationConfig2D.model_validate(
    dict(
        domain=dict(lower=(-1200, -2400), upper=(1200, 0), cells=(24, 24)),
        material=dict(density=2400, vp=3200, vs=1800),
        discretization=dict(type="quad_gll", degree=4),
        boundaries=dict(left="absorbing", right="absorbing", lower="absorbing"),
        time=dict(dt=0.001, duration=1.8),
        source=dict(
            position=(17.3, -1000.7),
            direction=(1, 1),
            wavelet=dict(f0=5, amplitude=1e8, time_shift=0.3),
        ),
        receivers=[
            dict(name="far", position=(700.4, -1000.7)),
            dict(name="near", position=(400.4, -1050.2)),
        ],
    )
)
result = Simulation2D(config, MPI.COMM_WORLD).run()
if MPI.COMM_WORLD.rank == 0:
    print("u/v shape:", result.displacement.shape, result.velocity.shape)
    print("receiver peak displacement [m]:", np.max(abs(result.displacement), axis=0))
