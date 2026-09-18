import numpy as np
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators

from .evidence import record
from .helpers import config


def test_layered_discrete_energy_loss_at_accepted_timestep():
    cfg = config(constraints=[dict(side="upper", components=["x"])])
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        rng = np.random.default_rng(188)
        dt = 0.8 * op.stable_dt
        step = op.start(dt, u0=0.01 * rng.normal(size=op.n), v0=0.02 * rng.normal(size=op.n))
        half_v = (step.current - step.previous) / dt
        energy = 0.5 * np.dot(op.mass * half_v, half_v) + 0.5 * np.dot(
            step.current, op.apply(step.previous)
        )
        initial = energy
        work = 0.0
        residual = 0.0
        for _ in range(1000):
            nxt, v, _, new = step.evaluate(np.zeros(op.n), energy=True)
            loss = dt * np.dot(op.damping * v, v)
            assert loss >= 0
            assert new <= energy + 3e-15 * initial
            assert new >= -3e-15 * initial
            residual = max(residual, abs(new - energy + loss) / initial)
            work += loss
            energy = new
            step.advance(nxt)
        print(
            "energy",
            dict(
                step_balance=residual,
                cumulative=abs(energy + work - initial) / initial,
                remaining=energy / initial,
                dt=dt,
            ),
        )
        assert residual < 3e-14
        assert abs(energy + work - initial) / initial < 3e-13
        assert energy / initial < 0.03
        record(
            "energy",
            dict(
                step_balance=float(residual),
                cumulative=float(abs(energy + work - initial) / initial),
                remaining=float(energy / initial),
                dt=float(dt),
            ),
        )
