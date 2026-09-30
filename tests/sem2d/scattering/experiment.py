"""Unmodified production operator/start/step path, physical-coordinate diagnostics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D

from .diagnostics import arrivals, measure
from .packets import Packet
from .reference import NAMES


def run(packet, h=300, degree=4, extent=14400, comm=MPI.COMM_SELF, audit=False, homogeneous=False):
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(-extent, -extent), upper=(extent, extent), cells=(round(2 * extent / h),) * 2
            ),
            discretization=dict(type="quad_gll", degree=degree) if degree else dict(type="tri_p1"),
            material=packet.lower.config()
            if homogeneous
            else dict(
                type="layered",
                layers=[
                    dict(lower=-extent, upper=0, material=packet.lower.config()),
                    dict(lower=0, upper=extent, material=packet.upper.config()),
                ],
            ),
        )
    )
    assert abs(round(2 * extent / h) * h - 2 * extent) < 1e-8
    geometry = packet.geometry(extent)
    assert geometry["return_margin"] > 0
    begin = time.perf_counter()
    with PlaneStrainOperators(cfg, comm) as op:
        xy = op.coordinates[: op.n // 2]
        u, v = packet.initial(xy)
        action = op.apply(u.ravel()).copy()
        steps = int(np.ceil(packet.time / 0.0005))
        dt = packet.time / steps
        bound = op.stable_dt
        assert dt < 0.8 * bound
        step = op.start(dt, u0=u.ravel(), v0=v.ravel())
        axis = np.linspace(
            -packet.analysis, packet.analysis, round(2 * packet.analysis / packet.sampling) + 1
        )
        x, z = np.meshgrid(axis, axis)
        sampler = PointMap2D(op.V, np.column_stack((x.ravel(), z.ravel())))
        receiver_positions = []
        arrival = packet.time - 0.4
        for name in NAMES:
            b = geometry["reference"]["branches"][name]
            c = (packet.lower if name[0] == "R" else packet.upper).speed(name[1])
            receiver_positions.append(
                np.array(geometry["hit"])
                + c * (arrival - geometry["hit_time"]) * np.array(b["direction"])
            )
        receivers = PointMap2D(op.V, receiver_positions)

        def sample(points, values):
            a = np.zeros((len(points.positions), 2))
            a[points.ids] = points.evaluate(values)
            return comm.allreduce(a)

        trace, times, energies = [], [], []
        zero = np.zeros(op.n)
        for j in range(steps + 1):
            nxt, velocity, _, energy = step.evaluate(zero, energy=j % 20 == 0)
            if energy is not None:
                energies.append(comm.allreduce(energy))
            if j % 10 == 0:
                trace.append(
                    np.concatenate(
                        (sample(receivers, step.current), sample(receivers, velocity)), axis=1
                    ).tolist()
                )
                times.append(j * dt)
            if j < steps:
                step.advance(nxt)
        grid = np.concatenate(
            (sample(sampler, step.current), sample(sampler, velocity)), axis=1
        ).reshape(len(axis), len(axis), 4)
        fields = op.material_fields
        centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], :2].mean(axis=1)
        if fields is not None:
            assert np.array_equal(fields.cell_layers, (centers[:, 1] > 0).astype(int))
        result = dict(
            config=cfg.model_dump(mode="json", by_alias=True),
            mode=packet.mode,
            reverse=packet.reverse,
            identical=packet.identical,
            h=h,
            degree=degree,
            extent=extent,
            dofs=2 * op.index_map.size_global,
            dt=dt,
            safe_dt=bound,
            dt_safe_ratio=dt / bound,
            steps=steps,
            time=packet.time,
            frequency=packet.frequency,
            sigma_s=packet.sigma,
            sigma_q=packet.width,
            geometry=geometry,
            bandwidth=packet.bandwidth(),
            wavelength_P=min(packet.lower.vp, packet.upper.vp) / packet.frequency,
            wavelength_S=min(packet.lower.vs, packet.upper.vs) / packet.frequency,
            average_spacing=h / max(degree, 1),
            points_per_wavelength_P=min(packet.lower.vp, packet.upper.vp)
            / packet.frequency
            / (h / max(degree, 1)),
            points_per_wavelength_S=min(packet.lower.vs, packet.upper.vs)
            / packet.frequency
            / (h / max(degree, 1)),
            analysis_spacing=packet.sampling,
            energy_drift=float(np.ptp(energies) / np.mean(energies)),
            receiver_positions=np.array(receiver_positions).tolist(),
            receiver_arrival=arrival,
            times=times,
            trace=trace,
            sum_factorization=op.sem_metadata["sum_factorization"] if degree else False,
            seconds=time.perf_counter() - begin,
        )
        if audit:
            pieces = comm.gather(
                np.column_stack(
                    (
                        xy,
                        op.mass.reshape(-1, 2),
                        action.reshape(-1, 2),
                        step.current.reshape(-1, 2),
                        velocity.reshape(-1, 2),
                        op.damping.reshape(-1, 2),
                    )
                ),
                root=0,
            )
            nc = op.mesh.topology.index_map(2).size_local
            material_pieces = comm.gather(
                np.column_stack(
                    (
                        centers[:nc],
                        fields.cell_layers[:nc],
                        fields.rho.x.array[fields.cell_dofs[:nc]],
                    )
                ),
                root=0,
            )
            layers = comm.allgather(np.unique(fields.cell_layers).tolist())
            if comm.rank == 0:
                a = np.concatenate(pieces)
                keys = np.round(a[:, :2], 8)
                result["coordinate_audit"] = a[np.lexsort((keys[:, 1], keys[:, 0]))].tolist()
                result["rank_layers"] = layers
                cells = np.concatenate(material_pieces)
                result["material_audit"] = cells[np.lexsort((cells[:, 1], cells[:, 0]))].tolist()
    if comm.rank == 0:
        if not packet.identical:
            result.update(measure(grid, packet))
            arrivals(result, packet)
        return result, grid
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", default="P", choices=["P", "S"])
    parser.add_argument("--h", type=float, default=300)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--extent", type=float, default=14400)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    row, grid = run(
        Packet(args.mode, args.reverse),
        args.h,
        args.degree,
        args.extent,
        MPI.COMM_WORLD,
        args.audit,
    )
    if MPI.COMM_WORLD.rank == 0:
        args.output.write_text(json.dumps(row, indent=2) + "\n")
        np.save(args.output.with_suffix(".npy"), grid)
        print(
            {k: row[k] for k in ["mode", "h", "seconds", "closure_error", "projection_residual"]},
            flush=True,
        )


if __name__ == "__main__":
    main()
