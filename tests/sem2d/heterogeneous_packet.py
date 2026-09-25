"""Normal P packet with independently derived scalar interface solution.

All amplitudes refer to the fixed +z displacement component (also for the
reflected wave). Traction continuity gives Z1(1-R)=Z2*T; displacement gives
1+R=T. Thus R=(Z1-Z2)/(Z1+Z2), T=2*Z1/(Z1+Z2).
"""

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.heterogeneous2d.helpers import layers
from tests.heterogeneous2d.interface import packet

# lambda=0 makes a planar vertical wave satisfy free vertical sides exactly.
FIRST = dict(density=2000, vp=2000, vs=2000 / np.sqrt(2))
SECOND = dict(density=2400, vp=3000, vs=3000 / np.sqrt(2))

F0 = 6.0
STATIONS = (-800.3, 800.3)
DURATION = 1.4


def reference(first, second):
    z1, z2 = (m["density"] * m["vp"] for m in [first, second])
    r, t = (z1 - z2) / (z1 + z2), 2 * z1 / (z1 + z2)
    return dict(R=r, T=t, R_flux=r * r, T_flux=z2 / z1 * t * t)


def analyze(time, trace, first, second):
    c1, c2 = first["vp"], second["vp"]
    centers = [(STATIONS[0] + 1600) / c1, (1600 - STATIONS[0]) / c1, 1600 / c1 + STATIONS[1] / c2]
    amplitudes, arrivals, errors = [], [], []
    exact = reference(first, second)
    for column, center, coeff in zip([0, 0, 1], centers, [1, exact["R"], exact["T"]], strict=True):
        window = abs(time - center) < 0.8 / F0
        a = np.pi * F0 * (time[window] - center)
        template = (1 - 2 * a * a) * np.exp(-a * a)
        values = trace[window, column, 1]
        amplitudes.append(float(template @ values / (template @ template)))
        # Quadratic interpolation of the central signed pulse extremum.
        j = int(np.argmax(abs(values)))
        offset = 0.0
        if 0 < j < len(values) - 1 and abs(coeff) > 1e-10:
            ym, y, yp = values[j - 1 : j + 2]
            offset = 0.5 * (ym - yp) / (ym - 2 * y + yp)
        arrivals.append(
            float(time[window][j] + offset * (time[1] - time[0]) - center)
            if abs(coeff) > 1e-10
            else None
        )
        errors.append(float(np.linalg.norm(values - coeff * template) / np.linalg.norm(template)))
    incident, r, t = amplitudes
    r, t = r / incident, t / incident
    z1, z2 = (m["density"] * m["vp"] for m in [first, second])
    rf, tf = r * r, z2 / z1 * t * t
    return dict(
        incident=incident,
        R=r,
        T=t,
        R_error=abs(r - exact["R"]),
        T_error=abs(t - exact["T"]),
        R_flux=rf,
        T_flux=tf,
        flux_error=max(abs(rf - exact["R_flux"]), abs(tf - exact["T_flux"])),
        flux_closure=abs(rf + tf - 1),
        arrival_errors=arrivals,
        waveform_errors=errors,
        transverse_peak=float(np.max(abs(trace[:, :, 0]))),
        analytical=exact,
    )


def run(p=4, h=100, case="increase", comm=MPI.COMM_SELF, traces=False, width=400):
    first, second = FIRST, SECOND
    if case == "decrease":
        first, second = SECOND, FIRST
    elif case in ("identical", "homogeneous"):
        second = first
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(-width, -4000),
                upper=(width, 4000),
                cells=(4 if p == 0 else 2, round(8000 / h)),
            ),
            material=first if case == "homogeneous" else layers(-4000, 0, 4000, first, second),
            discretization=dict(type="tri_p1") if p == 0 else dict(type="quad_gll", degree=p),
        )
    )
    with PlaneStrainOperators(cfg, comm) as op:
        xy = op.coordinates[: op.n // 2]
        # Rescale the audited packet coordinate so temporal bandwidth remains 6Hz
        # when swapping the incident material; center remains at z=-1600.
        z_equiv = (xy[:, 1] + 1600) * FIRST["vp"] / first["vp"] - 1600
        u0, v0 = np.zeros((op.n // 2, 2)), np.zeros((op.n // 2, 2))
        u0[:, 1], v0[:, 1] = packet(z_equiv)
        steps = int(np.ceil(DURATION / (0.15 * op.stable_dt)))
        dt = DURATION / steps
        time = np.arange(steps + 1) * dt
        step = op.start(dt, u0=u0.ravel(), v0=v0.ravel())
        points = PointMap2D(op.V, [(17.3, z) for z in STATIONS])
        utrace = np.empty((len(time), len(points.ids), 2))
        vtrace = np.empty_like(utrace)
        energy = []
        for i in range(len(time)):
            nxt, velocity, _, e = step.evaluate(np.zeros(op.n), energy=True)
            utrace[i], vtrace[i] = points.evaluate(step.current), points.evaluate(velocity)
            energy.append(comm.allreduce(e))
            if i < steps:
                step.advance(nxt)
        trace, vfull = np.empty((len(time), 2, 2)), np.empty((len(time), 2, 2))
        for ids, u, v in comm.allgather((points.ids, utrace, vtrace)):
            trace[:, ids], vfull[:, ids] = u, v
        metrics = analyze(time, trace, first, second)
        metrics.update(
            degree=p,
            h=h,
            case=case,
            dt=dt,
            safe_dt=op.stable_dt,
            energy_drift=float(np.ptp(energy) / np.mean(energy)),
            dofs=2 * op.index_map.size_global,
        )
        if traces:
            metrics.update(u=trace.tolist(), v=vfull.tolist())
        return metrics
