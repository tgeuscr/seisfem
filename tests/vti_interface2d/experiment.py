"""Production-kernel runner, MPI material checks and coordinate-based gathering."""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.oblique2d.experiment import purity

from .packets import EXTENT, TIME, initial
from .reference import LOWER, UPPER


def run(angle, h, comm=MPI.COMM_WORLD, extent=EXTENT, upper=UPPER, isotropic_upper=False):
    def vti(m):
        return dict(type="vti", density=m.rho, c11=m.c11, c33=m.c33, c13=m.c13, c55=m.c55)

    upper_input = (
        dict(
            density=upper.rho, vp=np.sqrt(upper.c33 / upper.rho), vs=np.sqrt(upper.c55 / upper.rho)
        )
        if isotropic_upper
        else vti(upper)
    )
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(-extent, -extent), upper=(extent, extent), cells=(round(2 * extent / h),) * 2
            ),
            material=dict(
                type="layered",
                layers=[
                    dict(
                        lower=-extent, upper=0, material=dict(density=LOWER.rho, vp=3000, vs=1700)
                    ),
                    dict(lower=0, upper=extent, material=upper_input),
                ],
            ),
        )
    )
    with PlaneStrainOperators(cfg, comm) as op:
        xy = op.coordinates[: op.n // 2]
        u, v = initial(xy, angle)
        bound = op.stable_dt
        steps = int(np.ceil(TIME / (0.8 * bound)))
        dt = TIME / steps
        step = op.start(dt, u0=u.ravel(), v0=v.ravel())
        mass = comm.allreduce(op.mass.reshape(-1, 2).sum(axis=0))
        action = op.apply(u.ravel()).copy()
        report = dict(
            angle=angle,
            h=h,
            dt=dt,
            stable_dt=bound,
            steps=steps,
            purity=purity(op, u, "P"),
            mass=mass.tolist(),
            mass_square=comm.allreduce(float(op.mass @ op.mass)),
            stiffness_form=comm.allreduce(float(u.ravel() @ action)),
            action_square=comm.allreduce(float(action @ action)),
        )
        f = op.material_fields
        centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], 1].mean(axis=1)
        expected = (centers > 0).astype(int)
        np.testing.assert_array_equal(f.cell_layers, expected)
        if cfg.has_vti:
            fields = [f.rho, f.c11, f.c33, f.c13, f.c55]
            low = [LOWER.rho, LOWER.c11, LOWER.c33, LOWER.c13, LOWER.c55]
            high = [upper.rho, upper.c11, upper.c33, upper.c13, upper.c55]
        else:
            fields = [f.rho, f.lam, f.mu]
            low = [LOWER.rho, LOWER.c13, LOWER.c55]
            high = [upper.rho, upper.c13, upper.c55]
        material_error = 0.0
        for field, lo, hi in zip(fields, low, high, strict=True):
            exact = np.where(expected, hi, lo)
            error = float(np.max(abs(field.x.array[f.cell_dofs] - exact)) / max(abs(lo), abs(hi)))
            assert error < 3e-15
            material_error = max(material_error, error)
        report["material_error"] = comm.allreduce(material_error, op=MPI.MAX)
        points = PointMap2D(op.V, [(-137.3, -611.7), (241.1, 713.3)])
        traces = []
        velocities = []
        stride = max(1, steps // 20)
        zero = np.zeros(op.n)
        for j in range(steps + 1):
            nxt, velocity, _, _ = step.evaluate(zero)
            if j % stride == 0:
                for values, history in [(step.current, traces), (velocity, velocities)]:
                    sample = np.zeros((len(points.positions), 2))
                    sample[points.ids] = points.evaluate(values)
                    history.append(comm.allreduce(sample).tolist())
            if j < steps:
                step.advance(nxt)
        report.update(trace=traces, velocity_trace=velocities)
        gathered = comm.gather(
            np.column_stack((xy, step.current.reshape(-1, 2), velocity.reshape(-1, 2))), root=0
        )
    if comm.rank == 0:
        data = np.concatenate(gathered)
        size = round(2 * extent / h) + 1
        ix = np.rint((data[:, 0] + extent) / h).astype(int)
        iz = np.rint((data[:, 1] + extent) / h).astype(int)
        assert len(np.unique(iz * size + ix)) == size * size
        grid = np.empty((size, size, 4))
        grid[iz, ix] = data[:, 2:]
        return report, grid
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=25)
    parser.add_argument("--h", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, grid = run(args.angle, args.h)
    if MPI.COMM_WORLD.rank == 0:
        from .diagnostics import measure

        report.update(measure(grid, args.angle, args.h))
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
