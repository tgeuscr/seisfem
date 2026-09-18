"""mpiexec -n N python -m tests.heterogeneous_absorbing2d.mpi_worker OUTPUT.npz"""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.fem2d import PlaneStrainOperators

from .helpers import MATERIALS, config, edge_matrix
from .test_example import example_config


def probe(comm):
    cfg = config()
    with PlaneStrainOperators(cfg, comm) as op:
        coordinates = np.concatenate(comm.allgather(op.coordinates[: op.n // 2]))
        nodes = np.lexsort((np.round(coordinates[:, 1], 12), np.round(coordinates[:, 0], 12)))
        order = (2 * nodes[:, None] + np.arange(2)).ravel()
        result = dict(
            coordinates=coordinates[nodes],
            damping=np.concatenate(comm.allgather(op.damping))[order],
            mass=np.concatenate(comm.allgather(op.mass))[order],
            dt=np.array(op.stable_dt),
        )
        for name in ["M", "K", "C"]:
            offsets, columns, values = getattr(op, name).getValuesCSR()
            first, last = op.index_map.local_range
            rows = np.repeat(np.arange(2 * first, 2 * last), np.diff(offsets))
            matrix = np.zeros((len(order), len(order)))
            entries = comm.allgather((rows, columns, values))
            for row_ids, column_ids, coefficients in entries:
                matrix[row_ids, column_ids] = coefficients
            result[name] = matrix[np.ix_(order, order)]
        expected = edge_matrix(coordinates[nodes], cfg)
        np.testing.assert_allclose(result["C"], expected, atol=8e-15, rtol=4e-15)
        np.testing.assert_allclose(result["damping"], expected.sum(axis=1), atol=8e-15, rtol=4e-15)
        # Distributed C action, including ghost synchronization.
        x, z = op.coordinates[: op.n // 2].T
        vector = np.column_stack((np.sin(x) + 0.2 * z, np.cos(z) - 0.3 * x)).ravel()
        op.field.x.array[: op.n] = vector
        op.field.x.scatter_forward()
        action = op.C.createVecLeft()
        try:
            op.C.mult(op.field.x.petsc_vec, action)
            result["action"] = np.concatenate(comm.allgather(action.array_r.copy()))[order]
        finally:
            action.destroy()
        fields = op.material_fields
        centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], :2].mean(axis=1)
        ids = np.searchsorted([-1, 1], centers[:, 1])
        np.testing.assert_array_equal(fields.cell_layers, ids)
        values = np.column_stack(
            [f.x.array[fields.cell_dofs] for f in [fields.rho, fields.lam, fields.mu]]
        )
        expected_values = np.array(
            [
                [
                    m["density"],
                    m["density"] * (m["vp"] ** 2 - 2 * m["vs"] ** 2),
                    m["density"] * m["vs"] ** 2,
                ]
                for m in MATERIALS
            ]
        )
        np.testing.assert_allclose(values, expected_values[ids], rtol=3e-15)
        nc = op.mesh.topology.index_map(2).size_local
        cells = np.concatenate(
            comm.allgather(np.column_stack((centers[:nc], ids[:nc], values[:nc])))
        )
        result["cells"] = cells[np.lexsort((np.round(cells[:, 1], 12), np.round(cells[:, 0], 12)))]
    return result


def main():
    comm = MPI.COMM_WORLD
    data = probe(comm)
    result = Simulation2D(example_config(), comm).run()
    data.update(u=result.displacement, v=result.velocity, receivers=result.receiver_coordinates)
    if comm.rank == 0:
        np.savez(sys.argv[1], **data)
        print(
            "layered absorber MPI", comm.size, "peak displacement", abs(data["u"]).max(), flush=True
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
