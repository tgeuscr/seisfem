"""Public 2D experiment; print reproducible envelope-peak P/S arrival diagnostics.

    python examples/2d/homogeneous.py --cells 640
    mpiexec -n 4 python examples/2d/homogeneous.py --cells 240

The default 240-cell case is lightweight but has measurable S-wave dispersion.
Use --cells 640 to reproduce the finest validation mesh. No files are written.
"""

import argparse

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def envelope_peak(time, trace, predicted):
    """Zero-padded analytic envelope, ±65 ms window, parabolic peak refinement."""
    n = 4 * len(time)
    multiplier = np.zeros(n)
    multiplier[0] = multiplier[n // 2] = 1
    multiplier[1 : n // 2] = 2
    envelope = abs(np.fft.ifft(np.fft.fft(trace, n=n) * multiplier)[: len(time)])
    ids = np.flatnonzero(abs(time - predicted) <= 0.065)
    peak = ids[np.argmax(envelope[ids])]
    if peak in (ids[0], ids[-1]):
        raise ValueError("Arrival peak reached the measurement window edge; refine the mesh")
    a, b, c = envelope[peak - 1 : peak + 2]
    return time[peak] + 0.5 * (a - c) / (a - 2 * b + c) * (time[1] - time[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=240)
    args = parser.parse_args()
    config = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-2400, -2400), upper=(2400, 2400), cells=(args.cells, args.cells)),
            material=dict(density=2400, vp=3200, vs=1800),
            time=dict(dt=0.0005, duration=0.95),
            source=dict(
                type="force",
                position=(3.7, 2.9),
                direction=(1, 0),
                wavelet=dict(type="ricker", f0=8, amplitude=1e8, time_shift=0.1875),
            ),
            receivers=[
                dict(name="axial", position=(903.7, 2.9)),
                dict(name="transverse", position=(3.7, 902.9)),
                dict(name="oblique", position=(640.0961030678928, 639.2961030678928)),
            ],
        )
    )
    result = Simulation2D(config).run()
    # Each rank has complete (time, receiver, x/z) displacement and velocity arrays.
    # Optional: dataset = result.to_xarray()
    if MPI.COMM_WORLD.rank == 0:
        for mode, receiver, speed in [("P", 0, 3200), ("S", 1, 1800)]:
            predicted = 0.1875 + 900 / speed
            measured = envelope_peak(result.time, result.displacement[:, receiver, 0], predicted)
            print(
                f"{mode}: predicted={predicted:.8f} s measured={measured:.8f} s "
                f"error={1e3 * (measured - predicted):+.3f} ms"
            )
        print("Shapes:", result.displacement.shape, result.velocity.shape)
        print("Source amplitude: 1e8 N/m (line force); displacement: m; velocity: m/s")


if __name__ == "__main__":
    main()
