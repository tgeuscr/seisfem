"""Production-backend packet benchmark; continuum truth lives in packets.py."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from seisfem.timestepping import CentralDifference

from .packets import EXTENT, FREQUENCY, RHO, TIME, VP, VS, Packet, peak_time, spectral_angle


def run(
    p,
    effective,
    branch="P",
    angle=25.0,
    fractions=(0.1,),
    comm=MPI.COMM_SELF,
    extent=EXTENT,
    audit=False,
):
    """p=0 denotes trusted triangles; p>0 denotes quadrilateral GLL degree p."""
    degree = max(p, 1)
    assert effective % degree == 0
    cells = effective // degree
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-extent, -extent), upper=(extent, extent), cells=(cells, cells)),
            material=dict(density=RHO, vp=VP, vs=VS),
            discretization=dict(type="quad_gll", degree=p) if p else dict(type="tri_p1"),
        )
    )
    comm.barrier()
    begin = time.perf_counter()
    with PlaneStrainOperators(cfg, comm) as op:
        assembly = comm.allreduce(time.perf_counter() - begin, op=MPI.MAX)
        begin = time.perf_counter()
        spectral = op.spectral_diagnostic()
        eigen_seconds = comm.allreduce(time.perf_counter() - begin, op=MPI.MAX)
        critical = spectral.critical_dt
        coordinates = op.coordinates[: op.n // 2]
        packet = Packet(branch, angle)
        u0 = packet.points(coordinates).ravel()
        v0 = packet.points(coordinates, velocity=True).ravel()
        exact_nodes = packet.points(coordinates, TIME)
        rho_weights = op.mass.reshape(-1, 2)[:, 0] / RHO
        exponential = rho_weights * np.exp(-1j * (coordinates @ packet.k0))

        def coefficient(u):
            local = np.sum(exponential[:, None] * u.reshape(-1, 2), axis=0)
            return comm.allreduce(local)

        initial_coefficient = coefficient(u0) @ packet.polarization
        exact_coefficient = coefficient(exact_nodes.ravel()) @ packet.polarization
        reference_phase = float(
            np.angle(
                exact_coefficient / initial_coefficient * np.exp(2j * np.pi * FREQUENCY * TIME)
            )
        )
        receiver = PointMap2D(op.V, [(43.7, -27.1)])
        axis = np.linspace(-EXTENT, EXTENT, 97)
        xx, zz = np.meshgrid(axis, axis, indexing="ij")
        gridpoints = PointMap2D(op.V, np.column_stack((xx.ravel(), zz.ravel())))
        exact_grid = packet.grid(axis, axis, TIME)
        action = op.apply(u0).copy()
        invariant = comm.allreduce(float(u0 @ action))
        action_square = comm.allreduce(float(action @ action))
        mass_sum = comm.allreduce(op.mass.reshape(-1, 2).sum(axis=0))
        total_dofs = 2 * op.index_map.size_global
        matrix_bytes = comm.allreduce(float(op.K.getInfo()["memory"] + op.M.getInfo()["memory"]))
        records = []
        for fraction in fractions:
            steps = int(np.ceil(TIME / (fraction * critical)))
            dt = TIME / steps
            # Exercise the identical production recurrence, with eigenvalue-audited dt.
            # Public start() deliberately retains its more conservative sufficient bound.
            step = CentralDifference(
                op.mass, op.damping, op.fixed, dt, op.apply, u0, v0, np.zeros(op.n)
            )
            energies = []
            trace = []
            velocity_trace = []
            comm.barrier()
            begin = time.perf_counter()
            for tick in range(steps + 1):
                nxt, velocity, _, energy = step.evaluate(np.zeros(op.n), energy=True)
                energies.append(comm.allreduce(energy))
                for values, history in [(step.current, trace), (velocity, velocity_trace)]:
                    sample = np.zeros(2)
                    if len(receiver.ids):
                        sample = receiver.evaluate(values)[0]
                    else:
                        receiver.evaluate(values)
                    history.append(comm.allreduce(sample).tolist())
                if tick < steps:
                    step.advance(nxt)
            stepping = comm.allreduce(time.perf_counter() - begin, op=MPI.MAX)
            final_coefficient = coefficient(step.current)
            projection = final_coefficient @ packet.polarization
            phase = float(
                np.angle(projection / initial_coefficient * np.exp(2j * np.pi * FREQUENCY * TIME))
            )
            signed_speed = -(phase - reference_phase) / (2 * np.pi * FREQUENCY * TIME)
            perpendicular = packet.polarization[[1, 0]] * np.array([1, -1])
            samples = np.zeros((len(gridpoints.positions), 2))
            samples[gridpoints.ids] = gridpoints.evaluate(step.current)
            samples = comm.allreduce(samples).reshape(97, 97, 2)
            field_error = float(np.linalg.norm(samples - exact_grid) / np.linalg.norm(exact_grid))
            times = np.arange(steps + 1) * dt
            exact_trace = packet.trace(receiver.positions[0], times)
            projected_trace = np.array(trace) @ packet.polarization
            observed = peak_time(times, projected_trace)
            expected = peak_time(times, exact_trace)
            measured_angle = spectral_angle(samples, axis[1] - axis[0], packet)
            expected_angle = spectral_angle(exact_grid, axis[1] - axis[0], packet)
            report = dict(
                backend="quad_gll" if p else "tri_p1",
                degree=p,
                effective=effective,
                cells_per_axis=cells,
                elements=cells * cells * (1 if p else 2),
                dofs=total_dofs,
                branch=branch,
                angle=angle,
                points_per_wavelength=packet.speed / FREQUENCY / (2 * extent / effective),
                h_element=2 * extent / cells,
                extent=extent,
                dtcrit=critical,
                safe_dt=op.stable_dt,
                requested_fraction=fraction,
                actual_fraction=dt / critical,
                dt=dt,
                steps=steps,
                spectral_residual=spectral.relative_residual,
                phase_speed=packet.speed * (1 + signed_speed),
                speed_error=abs(signed_speed),
                signed_speed_error=signed_speed,
                phase_residual=phase,
                continuum_phase_residual=reference_phase,
                polarization_error=float(abs(final_coefficient @ perpendicular) / abs(projection)),
                waveform_error=float(
                    np.linalg.norm(projected_trace - exact_trace) / np.linalg.norm(exact_trace)
                ),
                field_error=field_error,
                arrival_time=observed,
                reference_arrival=expected,
                arrival_error=abs(observed - expected) / expected,
                phase_angle=measured_angle,
                reference_phase_angle=expected_angle,
                angle_error=abs(measured_angle - expected_angle),
                energy_drift=float(np.ptp(energies) / np.mean(energies)),
                work_proxy=total_dofs * steps,
                assembly_seconds=assembly,
                eigensolve_seconds=eigen_seconds,
                stepping_seconds=stepping,
                solve_seconds=assembly + stepping,
                backend_seconds=assembly + eigen_seconds + stepping,
                matrix_bytes=matrix_bytes,
                mass=mass_sum.tolist(),
                stiffness_form=invariant,
                action_square=action_square,
                trace=trace,
                velocity_trace=velocity_trace,
                field_samples=samples[::12, ::12].tolist(),
            )
            if audit:
                pieces = comm.gather(
                    np.column_stack(
                        (
                            coordinates,
                            op.mass.reshape(-1, 2),
                            action.reshape(-1, 2),
                            step.current.reshape(-1, 2),
                            velocity.reshape(-1, 2),
                        )
                    ),
                    root=0,
                )
                if comm.rank == 0:
                    all_nodes = np.concatenate(pieces)
                    keys = np.round(all_nodes[:, :2], 9)
                    order = np.lexsort((keys[:, 1], keys[:, 0]))
                    report["coordinate_audit"] = all_nodes[order].tolist()
            records.append(report)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(4, 96, fractions=(0.02,), comm=MPI.COMM_WORLD, audit=True)
    if MPI.COMM_WORLD.rank == 0:
        args.output.write_text(json.dumps(report[0], indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
