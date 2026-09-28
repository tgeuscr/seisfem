"""Packet/large-box controls; same time grid for free and absorbing cases."""

import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.heterogeneous_absorbing2d.packets import HIGH, LOW, initial


def run(mode="P", h=100, p=4, boundary="absorbing", angle=0, beam=False, layered=False, layer=0):
    material = LOW if layer == 0 else HIGH
    speed = material["vp" if mode == "P" else "vs"]
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    hit = np.array([2400.0, -2400.0 if beam and angle == 0 else 0.0])
    if layer == 1:
        hit[1] = 2400.0
    center = hit - 1600 * direction
    reflected = direction * np.array([-1, 1])
    receiver = hit + 600 * reflected
    xmax = 4800 if boundary == "reference" else 2400
    zmax = 12000 if not beam else 4800
    nz = 24 if not beam else round(2 * zmax / h)
    # Keep each beam far from z=0 to isolate local boundary behavior.
    if layered:
        mats = dict(
            type="layered",
            layers=[
                dict(lower=-zmax, upper=0, material=LOW),
                dict(lower=0, upper=zmax, material=HIGH),
            ],
        )
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(0, -zmax), upper=(xmax, zmax), cells=(round(xmax / h), nz)),
            material=mats if layered else material,
            boundaries=dict(right="absorbing") if boundary == "absorbing" else {},
            discretization=dict(type="tri_p1") if p == 0 else dict(type="quad_gll", degree=p),
        )
    )
    dt = 0.001 * (h / 100) if p else 0.0005
    end = 2200 / speed + 0.1
    time = np.arange(int(np.ceil(end / dt)) + 1) * dt
    sample = np.array(
        [
            (x, z)
            for x in np.linspace(1200, 2200, 11)
            for z in np.linspace(hit[1] - 150, hit[1] + 150, 5)
        ]
    )
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        xy = op.coordinates[: op.n // 2]
        if beam:
            u, v = initial(xy, center, direction, mode, speed)
        else:
            s = (xy[:, 0] - 800) / (speed / (np.pi * 8))
            q = (1 - 2 * s * s) * np.exp(-s * s)
            dq = (-6 * s + 4 * s**3) * np.exp(-s * s) / (speed / (np.pi * 8))
            u, v = np.zeros_like(xy), np.zeros_like(xy)
            component = 0 if mode == "P" else 1
            u[:, component], v[:, component] = q, -speed * dq
        step = op.start(dt, u0=u.ravel(), v0=v.ravel())
        point = PointMap2D(op.V, [receiver])
        field = PointMap2D(op.V, sample)
        traces = np.empty((len(time), 2))
        for i in range(len(time)):
            nxt, _, _, _ = step.evaluate(np.zeros(op.n))
            traces[i] = point.evaluate(step.current)[0]
            if i < len(time) - 1:
                step.advance(nxt)
        final = field.evaluate(step.current)
        return dict(
            time=time,
            u=traces,
            field=final,
            dt=dt,
            safe_dt=op.stable_dt,
            coordinates=sample,
            mesh=cfg.domain.model_dump(),
            material=cfg.material.model_dump(by_alias=True),
            receiver=receiver.tolist(),
            center=center.tolist(),
            degree=p,
            mode=mode,
            angle=angle,
            beam=beam,
            frequency=8.0,
            transverse_width=400.0 if beam else None,
            longitudinal_width=speed / (np.pi * 8),
            direction=direction.tolist(),
            boundaries=cfg.boundaries.model_dump(),
            duration=float(time[-1]),
        )


def compare(mode="P", h=100, p=4, angle=0, beam=False, layered=False, layer=0):
    runs = {
        b: run(mode, h, p, b, angle, beam, layered, layer)
        for b in ["free", "absorbing", "reference"]
    }
    speed = (LOW if layer == 0 else HIGH)["vp" if mode == "P" else "vs"]
    time = runs["free"]["time"]
    window = abs(time - 2200 / speed) <= 0.1
    incident_window = abs(time - 1000 / speed) <= 0.1
    free = runs["free"]["u"] - runs["reference"]["u"]
    residual = runs["absorbing"]["u"] - runs["reference"]["u"]
    peak = float(np.max(np.linalg.norm(free[window], axis=1)))
    reflected = float(np.max(np.linalg.norm(residual[window], axis=1)))
    incident = float(np.max(np.linalg.norm(runs["reference"]["u"][incident_window], axis=1)))
    ratio = reflected / peak
    field_error = float(np.linalg.norm(runs["absorbing"]["field"] - runs["reference"]["field"]))
    free_field = float(np.linalg.norm(runs["free"]["field"] - runs["reference"]["field"]))
    if p == 4 and h == 50 and not beam and os.environ.get("SEISFEM_SEM_REPORT_DIR"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3))
        component = 0 if mode == "P" else 1
        for label in ["free", "absorbing", "reference"]:
            ax.plot(time, runs[label]["u"][:, component], label=label)
        ax.axvspan(2200 / speed - 0.1, 2200 / speed + 0.1, alpha=0.08, color="black")
        ax.set(xlabel="Time (s)", ylabel="Displacement (m)", title=f"Normal {mode}, p=4, h=50 m")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            Path(os.environ["SEISFEM_SEM_REPORT_DIR"]) / f"absorbing-normal-{mode}.png", dpi=150
        )
        plt.close(fig)
    output = dict(
        mode=mode,
        h=h,
        degree=p,
        angle=angle,
        beam=beam,
        layered=layered,
        layer=layer,
        incident=incident,
        incident_control_difference=float(
            np.max(
                abs(runs["free"]["u"][incident_window] - runs["absorbing"]["u"][incident_window])
            )
            / incident
        ),
        free_reflection=peak / incident,
        residual_reflection=reflected / incident,
        reflection_ratio=ratio,
        suppression_db=float(20 * np.log10(ratio)),
        reflected_waveform_ratio=float(
            np.linalg.norm(residual[window]) / np.linalg.norm(free[window])
        ),
        interior_field_ratio=field_error / free_field,
        interior_field_error_rms=field_error / np.sqrt(runs["reference"]["field"].size),
        free_field_error_rms=free_field / np.sqrt(runs["reference"]["field"].size),
        reflected_waveform_rms=float(np.sqrt(np.mean(residual[window] ** 2))),
        free_arrival_error=float(
            time[window][np.argmax(np.linalg.norm(free[window], axis=1))] - 2200 / speed
        ),
        reference_incident_time=float(
            time[incident_window][
                np.argmax(np.linalg.norm(runs["reference"]["u"][incident_window], axis=1))
            ]
        ),
        provenance={
            b: {k: v for k, v in r.items() if k not in ["time", "u", "field", "coordinates"]}
            for b, r in runs.items()
        },
    )
    return output
