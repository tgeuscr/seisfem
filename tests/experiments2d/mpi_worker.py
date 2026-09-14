"""Explicit serial/2/4-rank experiment and coherent-invalid-input probe.

Run with python -m tests.experiments2d.mpi_worker OUTPUT.npz.
Only root writes; all numerical API results are complete on every rank.
"""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.experiments2d.helpers import arrival_report, reciprocity_runs, small_config, wave_config
from tests.experiments2d.test_sources_points import POSITIONS, affine
from tests.plane_strain.helpers import configuration


def expect_collective_failure(comm, action):
    failed = False
    try:
        action()
    except ValueError:
        failed = True
    assert comm.allreduce(int(failed)) == comm.size


def invalid_inputs(comm):
    good = small_config().model_dump(mode="json", by_alias=True)
    for bad in [
        dict(source=dict(position=[0.5], direction=[1, 0], wavelet=dict(f0=1))),
        dict(source=dict(position=[0.5, 0.5], direction=[0, 0], wavelet=dict(f0=1))),
        dict(source=dict(position=[np.inf, 0.5], direction=[1, 0], wavelet=dict(f0=1))),
        dict(receivers=[dict(name="r", position=[np.nan, 0])]),
        dict(receivers=[dict(name="r", position=[1.1, 0])]),
    ]:
        data = good | bad if comm.rank == 0 else good
        expect_collective_failure(comm, lambda data=data: Simulation2D(data, comm))
    if comm.size > 1:
        data = good | dict(receivers=[]) if comm.rank == 0 else good
        expect_collective_failure(comm, lambda data=data: Simulation2D(data, comm))
    # Minimal mesh gives empty cell/DOF ownership on some of four ranks.
    with PlaneStrainOperators(configuration(n=1), comm) as op:
        for bad in [[(np.nan, 0)], [(np.inf, 0)], [1, 2], [(0,)], [(1, 2, 3)]]:
            positions = bad if comm.rank == 0 else [(0.5, 0.5)]
            expect_collective_failure(comm, lambda positions=positions: PointMap2D(op.V, positions))
        expect_collective_failure(comm, lambda: PointMap2D(op.V, [(1.01, 0.5)]))
        if comm.size > 1:
            expect_collective_failure(
                comm, lambda: PointMap2D(op.V, [(0.2 + 0.1 * comm.rank, 0.5)])
            )
        point = PointMap2D(op.V, [(0.5, 0.5)])
        for bad in [(0, 0), (np.nan, 0), (np.inf, 0), (1,)]:
            direction = bad if comm.rank == 0 else (1, 0)
            expect_collective_failure(comm, lambda direction=direction: point.unit_load(direction))
        load = point.unit_load((3, 4))
        np.testing.assert_allclose(
            comm.allreduce(load.reshape(-1, 2).sum(axis=0)), [0.6, 0.8], atol=4e-16
        )
        assert comm.allreduce(len(point.ids)) == 1
        values = point.evaluate(affine(op.coordinates[: op.n // 2]).ravel())
        np.testing.assert_allclose(
            comm.allreduce(values.sum(axis=0)), affine([(0.5, 0.5)])[0], atol=1e-15
        )


def point_checks(comm):
    with PlaneStrainOperators(configuration(n=8), comm) as op:
        points = PointMap2D(op.V, POSITIONS)
        field = affine(op.coordinates[: op.n // 2]).ravel()
        local = points.evaluate(field)
        error = comm.allreduce(
            float(np.max(abs(local - affine(np.array(POSITIONS)[points.ids])), initial=0)),
            op=MPI.MAX,
        )
        counts = np.zeros(len(POSITIONS), dtype=int)
        counts[points.ids] = 1
        np.testing.assert_array_equal(comm.allreduce(counts), 1)
        source_errors, loads = [], []
        for position in [(0.5, 0.5), (0.5 - 1e-12, 0.5 + 1e-12), (0, 0.37), (0.313, 0.487)]:
            source = PointMap2D(op.V, [position])
            load = source.unit_load((3, -4))
            assert comm.allreduce(len(source.ids)) == 1
            normalization = comm.allreduce(load.reshape(-1, 2).sum(axis=0))
            np.testing.assert_allclose(normalization, [0.6, -0.8], atol=4e-16)
            source_errors.append(
                abs(
                    comm.allreduce(float(load @ field))
                    - affine([position])[0] @ np.array([0.6, -0.8])
                )
            )
            loads.append(load.reshape(-1, 2))
        packed = np.column_stack((op.coordinates[: op.n // 2], *loads))
        all_values = np.concatenate(comm.allgather(packed))
        order = np.lexsort((np.round(all_values[:, 1], 12), np.round(all_values[:, 0], 12)))
        return dict(
            point_loads=all_values[order],
            affine_errors=np.array([error, max(source_errors)]),
            cell_keys=np.array(points.cell_keys),
        )


def main():
    comm = MPI.COMM_WORLD
    invalid_inputs(comm)
    data = point_checks(comm)
    # Source exactly at the central mesh vertex, with nearby ordered receivers.
    cfg = small_config(
        source=dict(
            position=(0.5, 0.5),
            direction=(3, -4),
            wavelet=dict(f0=9, amplitude=2.1, time_shift=0.04),
        ),
        receivers=[
            dict(name="right", position=(0.5 + 1e-12, 0.5)),
            dict(name="left", position=(0.5 - 1e-12, 0.5)),
            dict(name="center", position=(0.5, 0.5)),
        ],
    )
    awkward = Simulation2D(cfg, comm).run()
    data.update(
        awkward_u=awkward.displacement,
        awkward_v=awkward.velocity,
        coordinates=awkward.receiver_coordinates,
    )
    wave = Simulation2D(wave_config(240), comm).run()
    expected, measured = arrival_report(wave)
    data.update(
        wave_u=wave.displacement,
        wave_v=wave.velocity,
        arrivals=measured,
        expected_arrivals=expected,
        reciprocity=reciprocity_runs(comm),
    )
    # The public result is identical on every rank, not merely complete at root.
    for key in ["wave_u", "wave_v", "awkward_u"]:
        reference = comm.bcast(data[key] if comm.rank == 0 else None, root=0)
        np.testing.assert_array_equal(data[key], reference)
    if comm.rank == 0:
        np.savez(sys.argv[1], **data)
        print(
            "MPI experiment",
            comm.size,
            "affine errors",
            data["affine_errors"],
            "arrivals",
            measured,
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
