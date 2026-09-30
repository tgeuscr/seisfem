"""Snapshot free SEM operators, sources, histories, and final initialized fields.

Run directly with archived/current src on PYTHONPATH, then compare npz arrays.
The boundary module's free path must have no numerical effect.
"""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D


def snapshot(destination, layered):
    low = dict(density=2.3, vp=3.2, vs=1.8)
    high = dict(density=4.1, vp=4.6, vs=2.5)
    cfg = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1, -1), upper=(1, 1), cells=(4, 4)),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-1, upper=0, material=low),
                    dict(lower=0, upper=1, material=high),
                ],
            )
            if layered
            else low,
            discretization=dict(type="quad_gll", degree=4),
            time=dict(dt=0.0005, duration=0.1),
            source=dict(
                position=(0.17, -0.23),
                direction=(3, 4),
                wavelet=dict(f0=20, amplitude=2, time_shift=0.02),
            ),
            receivers=[dict(name="r", position=(0.31, 0.17))],
        )
    )
    with Simulation2D(cfg, MPI.COMM_SELF) as sim:
        op = sim.operators
        result = sim.run()
        assert op.C is None
        u = np.sin(np.arange(op.n))
        step = op.start(cfg.time.dt, u0=u, v0=0.1 * u)
        for _ in range(100):
            nxt, _, _, _ = step.evaluate(np.zeros(op.n))
            step.advance(nxt)
        _, v, _, _ = step.evaluate(np.zeros(op.n))
        np.savez(
            destination,
            M=op.M.getValuesCSR()[2],
            K=op.K.getValuesCSR()[2],
            mass=op.mass,
            dt=op.stable_dt,
            source=sim.source(0.037),
            action=op.apply(u).copy(),
            u=result.displacement,
            v=result.velocity,
            final_u=step.current,
            final_v=v,
            damping=op.damping,
        )


if __name__ == "__main__":
    snapshot(sys.argv[1], sys.argv[2] == "layered")
