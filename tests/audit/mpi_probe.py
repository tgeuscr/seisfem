"""Subprocess worker: discover real partition edges; no guessed rank boundaries."""

import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem import Simulation, SimulationConfig
from seisfem.points import PointMap
from tests.audit.test_independent_numerics import config, hand_matrices


def main():
    comm = MPI.COMM_WORLD
    destination = Path(sys.argv[1])
    cfg = config(cells=12)
    with Simulation(cfg, comm=comm) as sim:
        op = sim.operators
        coordinates = op.V.tabulate_dof_coordinates()[:, 0]
        cell_count = op.mesh.topology.index_map(1).size_local
        owned_cells = [coordinates[op.V.dofmap.cell_dofs(i)].tolist() for i in range(cell_count)]
        by_rank = comm.allgather(owned_cells)
        vertices = np.linspace(-1, 1, 13)
        # Rank incidence determined from owned cell connectivity, not ghost DOF ownership.
        rank_sets = []
        for vertex in vertices:
            rank_sets.append(
                [
                    r
                    for r, cells in enumerate(by_rank)
                    if any(np.min(abs(np.array(c) - vertex)) < 1e-14 for c in cells)
                ]
            )
        shared = [vertices[i] for i, ranks in enumerate(rank_sets) if len(ranks) > 1]
        if comm.size > 1:
            assert shared
        expected_m, expected_k = hand_matrices(12, (1.3, 4.7), (2.1, 3.2))
        node_ids = np.rint((coordinates[: op.n] + 1) * 6).astype(int)
        np.testing.assert_allclose(op.mass, expected_m[node_ids], rtol=3e-14)
        total_mass = comm.allreduce(float(sum(op.mass)))
        assert abs(total_mass - 6) < 2e-14
        nodal_values = np.sin(3.7 * vertices) + vertices**2
        np.testing.assert_allclose(
            op.apply(nodal_values[node_ids]),
            (expected_k @ nodal_values)[node_ids],
            rtol=2e-13,
            atol=3e-13,
        )
        positions = np.array(
            [
                -1,
                1,
                0,
                np.nextafter(0.0, -1.0),
                np.nextafter(0.0, 1.0),
                -0.271,
                0.713,
                *shared,
                *shared,
            ]
        )
        points = PointMap(op.V, positions, cfg.h)
        counts = comm.allreduce(np.bincount(points.ids, minlength=len(positions)))
        np.testing.assert_array_equal(counts, 1)
        for iteration in [0, 1, 2]:
            values = nodal_values + iteration * np.cos(1.9 * vertices)
            actual = points.evaluate(values[node_ids])
            expected = np.interp(positions[points.ids], vertices, values)
            np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
            op.field.x.array[: op.n] = values[node_ids]
            op.field.x.scatter_forward()
            np.testing.assert_allclose(
                actual, op.field.eval(points.points, points.cells).ravel(), atol=3e-14, rtol=0
            )
        for source in [0, -0.271, 0.713, *shared]:
            point = PointMap(op.V, [source], cfg.h)
            load = point.unit_load()
            expected_load = np.maximum(1 - abs(vertices - source) / cfg.h, 0)
            np.testing.assert_allclose(load, expected_load[node_ids], atol=5e-15, rtol=0)
            assert abs(comm.allreduce(float(sum(load))) - 1) < 3e-15
            assert abs(comm.allreduce(float(load @ coordinates[: op.n])) - source) < 3e-15
            if source in shared:
                owner = comm.allreduce(comm.rank if len(point.ids) else comm.size, op=MPI.MIN)
                index = int(np.argmin(abs(vertices - source)))
                assert owner == min(rank_sets[index])
        # One point creates empty local maps when ranks>1; zero points exercises every rank.
        for positions in [[0.137], []]:
            sparse = PointMap(op.V, positions, cfg.h)
            np.testing.assert_allclose(
                sparse.evaluate(nodal_values[node_ids]),
                np.interp(np.array(positions)[sparse.ids], vertices, nodal_values),
                atol=3e-14,
            )
            if comm.size > 1 and positions:
                assert comm.allreduce(int(len(sparse.ids) == 0)) == comm.size - 1
        summary = dict(
            shared_vertices=shared,
            ranks_per_vertex=rank_sets,
            total_mass=total_mass,
            cells_by_rank=by_rank,
        )
    # All-node receivers allow independent HDF readback comparison with actual traces.
    data = cfg.model_dump(mode="json", by_alias=True)
    source = float(shared[0]) if shared else 0.0
    data.update(
        source=dict(position=source, amplitude=-3.7, frequency=8, time_shift=0.02),
        time=dict(dt=0.001, duration=0.02),
        receivers=[dict(name=f"node{i}", position=float(z)) for i, z in enumerate(vertices)],
        output=dict(directory=str(destination), snapshot_stride=7, energy_stride=3),
    )
    result = Simulation(SimulationConfig.model_validate(data), comm=comm).run()
    assert result.component == "z"
    assert result.displacement.shape == (21, len(result.receiver_ids), 1)
    if comm.rank == 0:
        (destination / "probe.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
