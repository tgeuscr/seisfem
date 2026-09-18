"""Small production surface/well example: free top, absorbing sides and bottom.

python examples/2d/layered_absorbing.py
mpiexec -n 4 python examples/2d/layered_absorbing.py

The local first-order isotropic impedance absorber has nonzero oblique reflection.
No custom damping, experimental extension, PML, or file output is used here.
"""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def configuration(boundaries=None):
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1200, -2400), upper=(1200, 0), cells=(60, 60)),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-2400, upper=-1200, material=dict(density=2600, vp=3800, vs=2100)),
                    dict(lower=-1200, upper=-400, material=dict(density=2400, vp=3200, vs=1800)),
                    dict(lower=-400, upper=0, material=dict(density=2100, vp=2800, vs=1500)),
                ],
            ),
            boundaries=dict(left="absorbing", right="absorbing", lower="absorbing", upper="free")
            if boundaries is None
            else boundaries,
            time=dict(dt=0.001, duration=1.2),
            source=dict(
                position=(-173.3, -120.7),
                direction=(0, 1),
                wavelet=dict(f0=6, amplitude=1e8, time_shift=0.2),
            ),
            receivers=[
                dict(name="surface-left", position=(-400.3, 0)),
                dict(name="surface-right", position=(400.3, 0)),
                dict(name="well-shallow", position=(13.7, -200.3)),
                dict(name="well-middle", position=(13.7, -800.3)),
                dict(name="well-deep", position=(13.7, -1600.3)),
            ],
        )
    )


def main():
    result = Simulation2D(configuration()).run()
    if MPI.COMM_WORLD.rank == 0:
        print("Production layered model: top free, sides/bottom local impedance absorption.")
        print("Positive-up z; vector components (x,z). Not a PML or exact oblique absorber.")
        for i, name in enumerate(result.receiver_names):
            print(
                name,
                "peak displacement [m]",
                abs(result.displacement[:, i]).max(axis=0),
                "peak velocity [m/s]",
                np.max(abs(result.velocity[:, i]), axis=0),
            )


if __name__ == "__main__":
    main()
