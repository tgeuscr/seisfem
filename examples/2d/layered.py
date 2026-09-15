"""Two mesh-aligned horizontal materials through the public experiment API.

    python examples/2d/layered.py
    mpiexec -n 4 python examples/2d/layered.py

This vector force excites P and SV. The separate normal-P validation uses a
clean test-only plane packet; this example demonstrates the production API.
"""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def main():
    cfg = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1200, -2000), upper=(1200, 2000), cells=(120, 200)),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-2000, upper=0, material=dict(density=2000, vp=2000, vs=1000)),
                    dict(lower=0, upper=2000, material=dict(density=2400, vp=3000, vs=1500)),
                ],
            ),
            time=dict(dt=0.001, duration=1.15),
            source=dict(
                position=(17.3, -600.7),
                direction=(0, 1),
                wavelet=dict(f0=6, amplitude=1e8, time_shift=0.25),
            ),
            receivers=[
                dict(name="above", position=(13.7, 400.3)),
                dict(name="below", position=(-120.2, -800.3)),
                dict(name="interface", position=(0, 0)),
            ],
        )
    )
    result = Simulation2D(cfg).run()
    if MPI.COMM_WORLD.rank == 0:
        print("Layer interface: z=0 m, aligned with mesh row 100; z is positive-up.")
        print("Outer boundaries are traction-free; layered absorbers are not supported.")
        print(
            "Displacement/centered velocity shapes:",
            result.displacement.shape,
            result.velocity.shape,
        )
        for i, name in enumerate(result.receiver_names):
            print(name, "peak displacement [m]:", np.max(abs(result.displacement[:, i]), axis=0))
        print("The vector line-force amplitude is 1e8 N/m; components are (x,z).")
    # Optional: dataset = result.to_xarray()


if __name__ == "__main__":
    main()
