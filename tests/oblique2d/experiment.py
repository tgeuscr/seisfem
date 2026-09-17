"""Production FEM runner and coordinate-based MPI collection."""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators

from .packets import EXTENT, TIMES, initial
from .reference import LOWER, UPPER


def purity(op, u, mode):
    op.field.x.array[: op.n] = u.ravel()
    op.field.x.scatter_forward()
    nc = op.mesh.topology.index_map(2).size_local
    nodes = np.array([op.V.dofmap.cell_dofs(j) for j in range(nc)])
    xy = op.coordinates[nodes]
    affine = np.concatenate((np.ones((*xy.shape[:2], 1)), xy), axis=2)
    grad = np.linalg.inv(affine)[:, 1:, :]
    values = op.field.x.array.reshape(-1, 2)[nodes]
    derivative = np.einsum("cij,cjk->cik", grad, values)
    div = derivative[:, 0, 0] + derivative[:, 1, 1]
    curl = derivative[:, 1, 0] - derivative[:, 0, 1]
    energy = np.sum(values**2, axis=(1, 2)) / 3
    peak = op.comm.allreduce(float(energy.max(initial=0)), op=MPI.MAX)
    weight = abs(np.linalg.det(affine)) / 2 * (energy > 1e-4 * peak)
    sums = op.comm.allreduce(np.array([sum(weight * div**2), sum(weight * curl**2)]))
    return float(np.sqrt(sums[1] / sums[0] if mode == "P" else sums[0] / sums[1]))


def run(mode, h, comm=MPI.COMM_WORLD, extent=EXTENT):
    def material(m):
        return dict(density=m.rho, vp=m.vp, vs=m.vs)

    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(-extent, -extent),
                upper=(extent, extent),
                cells=(round(2 * extent / h), round(2 * extent / h)),
            ),
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-extent, upper=0, material=material(LOWER)),
                    dict(lower=0, upper=extent, material=material(UPPER)),
                ],
            ),
        )
    )
    with PlaneStrainOperators(cfg, comm) as op:
        xy = op.coordinates[: op.n // 2]
        u, v = initial(xy, mode)
        contamination = purity(op, u, mode)
        bound = op.stable_dt
        steps = int(np.ceil(TIMES[mode] / (0.8 * bound)))
        dt = TIMES[mode] / steps
        step = op.start(dt, u0=u.ravel(), v0=v.ravel())
        zero = np.zeros(op.n)
        mass = comm.allreduce(op.mass.reshape(-1, 2).sum(axis=0))
        # Classify cells by geometry, including ghosts; never by local numbering.
        cells = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], :2].mean(axis=1)
        fields = op.material_fields
        expected = (cells[:, 1] > 0).astype(int)
        assert np.array_equal(fields.cell_layers, expected)
        assert np.array_equal(
            fields.rho.x.array[fields.cell_dofs], np.where(expected, UPPER.rho, LOWER.rho)
        )
        assert np.array_equal(
            fields.mu.x.array[fields.cell_dofs],
            np.where(expected, UPPER.rho * UPPER.vs**2, LOWER.rho * LOWER.vs**2),
        )
        assert np.array_equal(
            fields.lam.x.array[fields.cell_dofs],
            np.where(
                expected,
                UPPER.rho * (UPPER.vp**2 - 2 * UPPER.vs**2),
                LOWER.rho * (LOWER.vp**2 - 2 * LOWER.vs**2),
            ),
        )
        mass_square = comm.allreduce(float(op.mass @ op.mass))
        action = op.apply(u.ravel()).copy()
        action_norm = comm.allreduce(float(action @ action))
        invariant = comm.allreduce(float(u.ravel() @ action))
        trace = []
        from seisfem.points2d import PointMap2D

        points = PointMap2D(op.V, [(-137.3, -611.7), (241.1, 713.3)])
        stride = max(1, steps // 20)
        for j in range(steps + 1):
            nxt, velocity, _, _ = step.evaluate(zero)
            if j % stride == 0:
                trace.append(points.evaluate(step.current).tolist())
            if j < steps:
                step.advance(nxt)
        gathered = comm.gather(
            np.column_stack((xy, step.current.reshape(-1, 2), velocity.reshape(-1, 2))), root=0
        )
        report = dict(
            mode=mode,
            h=h,
            dt=dt,
            stable_dt=bound,
            steps=steps,
            purity=contamination,
            mass=mass.tolist(),
            stiffness_form=invariant,
            mass_square=mass_square,
            action_square=action_norm,
            trace=trace,
        )
    if comm.rank == 0:
        data = np.concatenate(gathered)
        # Integer physical grid coordinates give deterministic global ordering.
        ix = np.rint((data[:, 0] + extent) / h).astype(int)
        iz = np.rint((data[:, 1] + extent) / h).astype(int)
        size = round(2 * extent / h) + 1
        assert len(np.unique(iz * size + ix)) == size * size
        grid = np.empty((size, size, 4))
        grid[iz, ix] = data[:, 2:]
        return report, grid
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["P", "S"], required=True)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, grid = run(args.mode, args.h)
    if MPI.COMM_WORLD.rank == 0:
        from .diagnostics import measure

        report.update(measure(grid, args.mode, args.h))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
