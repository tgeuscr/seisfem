"""Small MPI kernel smoke: owned masses, loads, matvecs, spectra and dynamics."""

import sys

import numpy as np
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import configuration
from tests.plane_strain.manufactured import analytic_force, initial_data, time_factor


def main():
    comm = MPI.COMM_WORLD
    with PlaneStrainOperators(configuration(n=6, fixed=True, diagonal="left_right"), comm) as op:
        u0, v0 = initial_data(op)
        load = op.assemble_load(analytic_force(op.mesh))
        stiffness_action = op.apply(u0).copy()
        report = op.spectral_diagnostic()
        component_mass = comm.allreduce(op.mass.reshape(-1, 2).sum(axis=0))
        np.testing.assert_allclose(component_mass, [2.3, 2.3], atol=3e-14)
        step = op.start(0.002, u0, v0, load)
        for n in range(31):
            nxt, velocity, acceleration, _ = step.evaluate(load * time_factor(0.002 * n))
            if n < 30:
                step.advance(nxt)
        packed = np.column_stack(
            (
                op.coordinates[: op.n // 2],
                op.mass.reshape(-1, 2),
                load.reshape(-1, 2),
                stiffness_action.reshape(-1, 2),
                step.current.reshape(-1, 2),
                velocity.reshape(-1, 2),
                acceleration.reshape(-1, 2),
                op.fixed.reshape(-1, 2),
            )
        )
        stable_bound = op.stable_dt
        gathered = comm.gather(packed, root=0)
        if comm.rank == 0:
            values = np.concatenate(gathered)
            order = np.lexsort((np.round(values[:, 1], 12), np.round(values[:, 0], 12)))
            np.savez(
                sys.argv[1], values=values[order], eigenvalue=report.lambda_max, bound=stable_bound
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
