"""Public point-force workflow with a preserved free surface and larger-box truth."""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D

MATERIAL = dict(density=2400, vp=3200, vs=1800)


def config(boundary="absorbing"):
    lo, hi = ((-3600, -4800), (3600, 0)) if boundary == "reference" else ((-1200, -2400), (1200, 0))
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(
                lower=lo,
                upper=hi,
                cells=(round((hi[0] - lo[0]) / 100), round((hi[1] - lo[1]) / 100)),
            ),
            material=MATERIAL,
            discretization=dict(type="quad_gll", degree=4),
            boundaries={}
            if boundary in ("free", "reference")
            else dict(left="absorbing", right="absorbing", lower="absorbing"),
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


def study():
    runs = {
        b: Simulation2D(config(b), MPI.COMM_SELF).run() for b in ["free", "absorbing", "reference"]
    }
    time = runs["free"].time
    late = time > 1.05
    early = (time > 0.35) & (time < 0.75)
    truth = runs["reference"].displacement
    free = runs["free"].displacement - truth
    residual = runs["absorbing"].displacement - truth
    distance = 700.4 - 17.3
    centers = {
        mode: 0.3 + distance / MATERIAL[speed] for mode, speed in [("P", "vp"), ("SV", "vs")]
    }
    peaks = {
        mode: float(np.max(abs(truth[abs(time - center) < 0.075, 0, component])))
        for (mode, center), component in zip(centers.items(), [0, 1], strict=True)
    }
    return dict(
        late_error_ratio=float(np.linalg.norm(residual[late]) / np.linalg.norm(free[late])),
        early_error=float(np.linalg.norm(residual[early]) / np.linalg.norm(truth[early])),
        direct_window_centers=centers,
        direct_component_peaks=peaks,
        shape=list(truth.shape),
        config=config().model_dump(mode="json", by_alias=True),
    )
