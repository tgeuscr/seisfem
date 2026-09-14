"""Compare all-free outer sides with a free surface and three absorbing sides.

    python examples/2d/absorbing.py
    mpiexec -n 4 python examples/2d/absorbing.py

Print amplitude diagnostics, not energy reflection coefficients. The validation
report adds an enlarged-domain reference to separate artificial boundary returns.
"""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def main():
    model = dict(
        domain=dict(lower=(-1200, -1600), upper=(1200, 0), cells=(120, 80)),
        material=dict(density=2400, vp=3200, vs=1800),
        time=dict(dt=0.001, duration=1.6),
        source=dict(
            type="force",
            position=(17.3, -400.7),
            direction=(1, 1),
            wavelet=dict(type="ricker", f0=5, amplitude=1e8, time_shift=0.3),
        ),
        receivers=[
            dict(name="below", position=(17.3, -600.7)),
            dict(name="right", position=(300.4, -450.2)),
            dict(name="left", position=(-270.5, -350.8)),
        ],
    )
    free = Simulation2D(SimulationConfig2D.model_validate(model)).run()
    model["boundaries"] = dict(left="absorbing", right="absorbing", lower="absorbing", upper="free")
    absorbing = Simulation2D(SimulationConfig2D.model_validate(model)).run()
    early = (free.time >= 0.50) & (free.time <= 0.75)
    late = free.time >= 0.95
    surface_change = np.linalg.norm(
        (absorbing.displacement - free.displacement)[early]
    ) / np.linalg.norm(free.displacement[early])
    late_ratio = np.linalg.norm(absorbing.displacement[late]) / np.linalg.norm(
        free.displacement[late]
    )
    if MPI.COMM_WORLD.rank == 0:
        print(f"Free-surface window relative change: {surface_change:.6g}")
        print(f"Late displacement trace norm ratio, absorbing/free: {late_ratio:.6g}")
        print(
            "The late ratio includes physical surface-wave tails; it is not an energy coefficient."
        )
        print("Shapes:", absorbing.displacement.shape, absorbing.velocity.shape)
        print("Source amplitude: N/m; displacement: m; centered velocity: m/s")
        print("Upper side remains traction-free. Only left, right and lower sides absorb.")
    # Optional xarray: dataset = absorbing.to_xarray()


if __name__ == "__main__":
    main()
