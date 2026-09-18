"""Coordinate-ordered MPI invariants for homogeneous and mixed-layer VTI."""

import sys

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.fem2d import PlaneStrainOperators

from .helpers import ISO, RHO, C, config


def probe(layered, comm):
    with PlaneStrainOperators(config(layered), comm) as op:
        coordinates = np.concatenate(comm.allgather(op.coordinates[: op.n // 2]))
        nodes = np.lexsort((np.round(coordinates[:, 1], 12), np.round(coordinates[:, 0], 12)))
        order = (2 * nodes[:, None] + np.arange(2)).ravel()
        result = dict(
            coordinates=coordinates[nodes],
            mass=np.concatenate(comm.allgather(op.mass))[order],
            dt=np.array(op.stable_dt),
        )
        for name in ["M", "K"]:
            offsets, columns, values = getattr(op, name).getValuesCSR()
            first, last = op.index_map.local_range
            rows = np.repeat(np.arange(2 * first, 2 * last), np.diff(offsets))
            matrix = np.zeros((len(order), len(order)))
            for row, col, val in comm.allgather((rows, columns, values)):
                matrix[row, col] = val
            result[name] = matrix[np.ix_(order, order)]
        x, z = op.coordinates[: op.n // 2].T
        vector = np.column_stack((np.sin(x) + 0.2 * z, np.cos(z) - 0.3 * x)).ravel()
        result["action"] = np.concatenate(comm.allgather(op.apply(vector).copy()))[order]
        if layered:
            f = op.material_fields
            centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], :2].mean(axis=1)
            ids = (centers[:, 1] > 0).astype(int)
            np.testing.assert_array_equal(f.cell_layers, ids)
            values = np.column_stack(
                [field.x.array[f.cell_dofs] for field in [f.rho, f.c11, f.c33, f.c13, f.c55]]
            )
            rho = ISO["density"]
            p = rho * ISO["vp"] ** 2
            s = rho * ISO["vs"] ** 2
            expected = np.array([[rho, p, p, p - 2 * s, s], [RHO, *C]])[ids]
            # All local and ghost cells are checked against physical z.
            np.testing.assert_allclose(values, expected, rtol=2e-15)
            count = op.mesh.topology.index_map(2).size_local
            cells = np.concatenate(
                comm.allgather(np.column_stack((centers[:count], ids[:count], values[:count])))
            )
            result["cells"] = cells[
                np.lexsort((np.round(cells[:, 1], 12), np.round(cells[:, 0], 12)))
            ]
    data = config(layered).model_dump(mode="json", by_alias=True)
    data.update(
        time=dict(dt=1e-5, duration=0.002),
        source=dict(
            position=(0.13, -0.37),
            direction=(1, 2),
            wavelet=dict(f0=2000, amplitude=1e6, time_shift=0.0003),
        ),
        receivers=[
            dict(name="lower", position=(0.21, -0.17)),
            dict(name="upper", position=(-0.21, 0.37)),
        ],
    )
    result_sim = Simulation2D(SimulationConfig2D.model_validate(data), comm).run()
    result.update(u=result_sim.displacement, v=result_sim.velocity)
    return result


def main():
    comm = MPI.COMM_WORLD
    result = {}
    for layered in [False, True]:
        result.update({f"{layered}-{k}": v for k, v in probe(layered, comm).items()})
    if comm.rank == 0:
        np.savez(sys.argv[1], **result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
