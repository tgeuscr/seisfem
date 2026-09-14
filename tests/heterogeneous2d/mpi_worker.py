"""mpiexec -n N python -m tests.heterogeneous2d.mpi_worker OUTPUT.npz"""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from tests.heterogeneous2d.helpers import HIGH, LOW, config, public_config
from tests.heterogeneous2d.test_materials import invalid_models


def invalid_inputs(comm):
    good = public_config().model_dump(mode="json", by_alias=True)
    for bad in invalid_models():
        data = bad | dict(time=dict(dt=0.001, duration=0.01)) if comm.rank == 0 else good
        failed = False
        try:
            Simulation2D(data, comm)
        except ValueError:
            failed = True
        assert comm.allreduce(int(failed)) == comm.size
    if comm.size > 1:
        data = good | dict(material=LOW) if comm.rank == 0 else good
        failed = False
        try:
            Simulation2D(data, comm)
        except ValueError:
            failed = True
        assert comm.allreduce(int(failed)) == comm.size
    # Empty-cell ranks: one material in two triangles on four processes.
    tiny = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(cells=(1, 1)),
            material=dict(type="layered", layers=[dict(lower=0, upper=1, material=LOW)]),
        )
    )
    with PlaneStrainOperators(tiny, comm) as op:
        np.testing.assert_allclose(
            comm.allreduce(op.mass.reshape(-1, 2).sum(axis=0)), LOW["density"], rtol=2e-15
        )


def operator_probe(comm):
    with PlaneStrainOperators(
        config(constraints=[dict(side="lower", components=["x"])]), comm
    ) as op:
        nf = op.mesh.topology.index_map(2).size_local
        fields = op.material_fields
        centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], :2].mean(axis=1)
        # Include ghost entries in the local assignment check.
        expected = (centers[:, 1] > 0).astype(int)
        np.testing.assert_array_equal(fields.cell_layers, expected)
        values = np.column_stack(
            [f.x.array[fields.cell_dofs] for f in [fields.rho, fields.lam, fields.mu]]
        )
        np.testing.assert_array_equal(
            values[:, 0], np.where(expected, HIGH["density"], LOW["density"])
        )
        cells = np.concatenate(
            comm.allgather(np.column_stack((centers[:nf], expected[:nf], values[:nf])))
        )
        cells = cells[np.lexsort((np.round(cells[:, 1], 12), np.round(cells[:, 0], 12)))]
        coordinates = np.concatenate(comm.allgather(op.coordinates[: op.n // 2]))
        nodes = np.lexsort((np.round(coordinates[:, 1], 12), np.round(coordinates[:, 0], 12)))
        order = (2 * nodes[:, None] + np.arange(2)).ravel()
        x, z = op.coordinates[: op.n // 2].T
        vector = np.column_stack((np.sin(x) + 0.3 * z, np.cos(z) - 0.2 * x)).ravel()
        action = np.concatenate(comm.allgather(op.apply(vector).copy()))[order]
        data = dict(
            cells=cells,
            mass=np.concatenate(comm.allgather(op.mass))[order],
            action=action,
            dt=np.array(op.stable_dt),
            coordinates=coordinates[nodes],
        )
        for name in ["M", "K"]:
            offsets, cols, values = getattr(op, name).getValuesCSR()
            first, last = op.index_map.local_range
            rows = np.repeat(np.arange(2 * first, 2 * last), np.diff(offsets))
            entries = comm.allgather((rows, cols, values))
            matrix = np.zeros((len(order), len(order)))
            for rows, cols, values in entries:
                matrix[rows, cols] = values
            data[name] = matrix[np.ix_(order, order)]
        return data


def main():
    comm = MPI.COMM_WORLD
    invalid_inputs(comm)
    data = operator_probe(comm)
    result = Simulation2D(public_config(), comm).run()
    data.update(
        u=result.displacement, v=result.velocity, receiver_coordinates=result.receiver_coordinates
    )
    if comm.rank == 0:
        np.savez(sys.argv[1], **data)
        print(
            "heterogeneous MPI",
            comm.size,
            "safe dt",
            data["dt"],
            "peak u",
            abs(data["u"]).max(),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
