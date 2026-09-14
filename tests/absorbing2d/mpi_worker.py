"""Run with mpiexec -n N python -m tests.absorbing2d.mpi_worker OUTPUT.npz."""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.fem2d import PlaneStrainOperators
from tests.absorbing2d.experiments import THREE_SIDES, packet_run, reciprocity_runs, seismic_config
from tests.absorbing2d.test_operators import config
from tests.experiments2d.helpers import small_config


def expect_failure(comm, call):
    failed = False
    try:
        call()
    except ValueError:
        failed = True
    assert comm.allreduce(int(failed)) == comm.size


def invalid_configurations(comm):
    good = small_config().model_dump(mode="json", by_alias=True)
    for change in [
        dict(boundaries=dict(top="absorbing")),
        dict(boundaries=dict(right="pml")),
        dict(boundaries=dict(left=float("nan"))),
        dict(
            boundaries=dict(right="absorbing"), constraints=[dict(side="right", components=["z"])]
        ),
    ]:
        data = good | change if comm.rank == 0 else good
        expect_failure(comm, lambda data=data: Simulation2D(data, comm))
    if comm.size > 1:
        data = good | dict(boundaries=dict(right="absorbing")) if comm.rank == 0 else good
        expect_failure(comm, lambda: Simulation2D(data, comm))
    # Two triangles, including ranks with no owned cells/DOFs or marked facets.
    cfg = config(("left", "right", "lower"), domain=dict(cells=(1, 1)))
    with PlaneStrainOperators(cfg, comm) as op:
        totals = comm.allreduce(op.damping.reshape(-1, 2).sum(axis=0))
        np.testing.assert_allclose(
            totals, 2.3 * np.array([2 * 3.2 + 1.8, 2 * 1.8 + 3.2]), atol=1e-14
        )
        step = op.start(0.01, v0=np.ones(op.n))
        nxt, _, _, _ = step.evaluate(np.zeros(op.n))
        assert np.all(np.isfinite(nxt))


def matrix_data(comm):
    cfg = config(("left", "right", "lower"), constraints=[dict(side="upper", components=["x"])])
    with PlaneStrainOperators(cfg, comm) as op:
        coordinates = np.concatenate(comm.allgather(op.coordinates[: op.n // 2]))
        damping = np.concatenate(comm.allgather(op.damping))
        fixed = np.concatenate(comm.allgather(op.fixed))
        offsets, cols, values = op.C.getValuesCSR()
        first, last = op.index_map.local_range
        rows = np.repeat(np.arange(2 * first, 2 * last), np.diff(offsets))
        pieces = comm.allgather((rows, cols, values))
        C = np.zeros((len(damping), len(damping)))
        for rows, cols, values in pieces:
            C[rows, cols] = values
        order = np.lexsort((np.round(coordinates[:, 1], 12), np.round(coordinates[:, 0], 12)))
        dofs = (2 * order[:, None] + np.arange(2)).ravel()
        np.testing.assert_allclose(C, C.T, atol=2e-15)
        np.testing.assert_allclose(C.sum(axis=1), damping, atol=4e-15)
        assert np.min(damping) >= 0
        return dict(
            C=C[np.ix_(dofs, dofs)],
            damping=damping[dofs],
            fixed=fixed[dofs],
            coordinates=coordinates[order],
        )


def main():
    comm = MPI.COMM_WORLD
    invalid_configurations(comm)
    data = matrix_data(comm)
    free = Simulation2D(seismic_config(), comm).run()
    absorbing = Simulation2D(seismic_config(THREE_SIDES), comm).run()
    late = free.time >= 0.95
    packet = packet_run("S", 20, comm=comm)
    data.update(
        u=absorbing.displacement,
        v=absorbing.velocity,
        receiver_coordinates=absorbing.receiver_coordinates,
        late_trace_ratio=np.linalg.norm(absorbing.displacement[late])
        / np.linalg.norm(free.displacement[late]),
        normal_reflection=packet["reflection"],
        reciprocity=reciprocity_runs(comm),
    )
    if comm.rank == 0:
        np.savez(sys.argv[1], **data)
        print(
            "absorber MPI",
            comm.size,
            "normal S reflection",
            data["normal_reflection"],
            "late trace ratio",
            data["late_trace_ratio"],
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
