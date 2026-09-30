"""Physical-coordinate heterogeneous SEM absorber audit on 1/2/4 ranks."""

import json
import sys

import numpy as np
from dolfinx import mesh
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from tests.heterogeneous2d.helpers import HIGH, LOW


def measure(comm):
    cfg = SimulationConfig2D.model_validate(
        dict(
            domain=dict(lower=(-1, -1), upper=(1, 1), cells=(8, 8)),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-1, upper=-0.25, material=LOW),
                    dict(lower=-0.25, upper=0.25, material=HIGH),
                    dict(lower=0.25, upper=1, material=LOW),
                ],
            ),
            discretization=dict(type="quad_gll", degree=4),
            boundaries=dict(left="absorbing", right="absorbing", lower="absorbing"),
            time=dict(dt=0.0005, duration=0.12),
            source=dict(
                position=(0.17, -0.13),
                direction=(3, 4),
                wavelet=dict(f0=20, amplitude=2, time_shift=0.02),
            ),
            receivers=[
                dict(name="above", position=(0.13, 0.12)),
                dict(name="below", position=(-0.11, -0.19)),
            ],
        )
    )
    with Simulation2D(cfg, comm) as sim:
        op = sim.operators
        result = sim.run()
        xy = op.coordinates[: op.n // 2]
        u0 = np.column_stack((np.sin(xy[:, 0]) + xy[:, 1], np.cos(xy[:, 1]))).ravel()
        v0 = 0.1 * u0
        action = op.apply(u0).copy()
        step = op.start(cfg.time.dt, u0=u0, v0=v0)
        for _ in range(100):
            nxt, velocity, _, _ = step.evaluate(np.zeros(op.n))
            step.advance(nxt)
        _, velocity, _, _ = step.evaluate(np.zeros(op.n))
        vector, output = op.C.createVecRight(), op.C.createVecLeft()
        try:
            vector.array[:] = u0
            op.C.mult(vector, output)
            damping_action = output.array_r.copy()
        finally:
            vector.destroy()
            output.destroy()
        fields = np.column_stack(
            (
                xy,
                op.mass.reshape(-1, 2),
                action.reshape(-1, 2),
                step.current.reshape(-1, 2),
                velocity.reshape(-1, 2),
                op.damping.reshape(-1, 2),
                damping_action.reshape(-1, 2),
            )
        )
        fields = np.vstack(comm.allgather(fields))
        keys = np.round(fields[:, :2], 12)
        fields = fields[np.lexsort((keys[:, 1], keys[:, 0]))]
        f = op.material_fields
        count = op.mesh.topology.index_map(2).size_local
        centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0][:count], :2].mean(axis=1)
        cells = np.column_stack(
            (
                centers,
                f.cell_layers[:count],
                f.rho.x.array[f.cell_dofs[:count]],
                f.lam.x.array[f.cell_dofs[:count]],
                f.mu.x.array[f.cell_dofs[:count]],
            )
        )
        cells = np.vstack(comm.allgather(cells))
        cells = cells[np.lexsort((cells[:, 1], cells[:, 0]))]
        facets = mesh.exterior_facet_indices(op.mesh.topology)
        g = mesh.entities_to_geometry(op.mesh, 1, facets)
        midpoint = op.mesh.geometry.x[g, :2].mean(axis=1)
        keep = midpoint[:, 1] < 1 - 1e-12
        facets, midpoint = facets[keep], midpoint[keep]
        adjacent = np.array([op.mesh.topology.connectivity(1, 2).links(i)[0] for i in facets])
        facet_data = np.column_stack(
            (midpoint, f.cell_layers[adjacent], f.rho.x.array[f.cell_dofs[adjacent]])
        )
        facet_counts = comm.allgather(len(facets))
        facet_data = np.vstack(comm.allgather(facet_data))
        facet_data = facet_data[np.lexsort((facet_data[:, 1], facet_data[:, 0]))]
        return dict(
            fields=fields.tolist(),
            facets=facet_data.tolist(),
            facet_counts=facet_counts,
            cells=cells.tolist(),
            u=result.displacement.tolist(),
            v=result.velocity.tolist(),
            safe_dt=op.stable_dt,
            dtcrit=op.spectral_diagnostic().critical_dt,
            rank_layers=comm.allgather(np.unique(f.cell_layers[:count]).tolist()),
        )


if __name__ == "__main__":
    try:
        report = measure(MPI.COMM_WORLD)
        if MPI.COMM_WORLD.rank == 0:
            with open(sys.argv[1], "w") as stream:
                json.dump(report, stream)
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
