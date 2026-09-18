"""Compare VTI and isotropic point-force receiver responses; free boundaries.

Run: python examples/2d/vti.py (also supports mpiexec -n N).
A generic line force excites both qP and qSV; it is not a pure-mode source.
"""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def configuration(vti=True):
    material = (
        dict(type="vti", density=2500, c11=40e9, c33=25e9, c13=9e9, c55=7e9)
        if vti
        else dict(density=2500, vp=np.sqrt(25e9 / 2500), vs=np.sqrt(7e9 / 2500))
    )
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1600, -1600), upper=(1600, 1600), cells=(80, 80)),
            material=material,
            time=dict(dt=0.001, duration=0.35),
            source=dict(
                position=(0, 0),
                direction=(1, 1),
                wavelet=dict(f0=12, amplitude=1e8, time_shift=0.06),
            ),
            receivers=[
                dict(name=name, position=point)
                for name, point in [
                    ("horizontal", (600, 0)),
                    ("vertical", (0, 600)),
                    ("oblique", (420, 420)),
                ]
            ],
        )
    )


def main():
    for vti in [False, True]:
        result = Simulation2D(configuration(vti)).run()
        if MPI.COMM_WORLD.rank == 0:
            print("VTI" if vti else "Isotropic control with identical vertical speeds")
            for i, name in enumerate(result.receiver_names):
                trace = np.linalg.norm(result.displacement[:, i], axis=1)
                print(
                    name,
                    "peak displacement [m]",
                    trace.max(),
                    "peak time [s]",
                    result.time[np.argmax(trace)],
                )
    if MPI.COMM_WORLD.rank == 0:
        print("Peaks are mixed-mode waveform diagnostics, not phase-velocity estimates.")
        print("Positive-up z; free boundaries. VTI absorbing boundaries are unsupported.")


if __name__ == "__main__":
    main()
