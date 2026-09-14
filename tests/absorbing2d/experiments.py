"""Test-only packets and public seismic controls; no production initial-data API."""

import numpy as np
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D

MATERIAL = dict(density=2400, vp=3200, vs=1800)


def packet_run(mode="P", h=20, frequency=8, absorbing=True, comm=MPI.COMM_SELF):
    """Plane Ricker displacement q(x-ct), independent of z: exactly P or SV.

    q=Phi'(x), Phi=(x-x0)*exp(-((x-x0)/w)^2). P is grad Phi;
    SV is minus rotated_grad Phi, with the same signed scalar q in z.
    Tangential-side essential components admit the exact plane solution.
    Only the right boundary changes between the paired controls.
    """
    speed = MATERIAL["vp" if mode == "P" else "vs"]
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(0, -2000), upper=(2400, 2000), cells=(round(2400 / h), round(4000 / h))
            ),
            material=MATERIAL,
            boundaries=dict(left="absorbing", right="absorbing" if absorbing else "free"),
            constraints=[
                dict(side=side, components=["z" if mode == "P" else "x"])
                for side in ["lower", "upper"]
            ],
        )
    )
    dt = 0.0005
    incident_time = 1000 / speed
    reflected_time = 2200 / speed
    half_window = 0.8 / frequency
    time = np.arange(int(np.ceil((reflected_time + half_window) / dt)) + 1) * dt
    component = 0 if mode == "P" else 1
    with PlaneStrainOperators(cfg, comm) as op:
        width = speed / (np.pi * frequency)
        s = (op.coordinates[: op.n // 2, 0] - 800) / width
        q = (1 - 2 * s * s) * np.exp(-s * s)
        derivative = (-6 * s + 4 * s**3) * np.exp(-s * s) / width
        u0 = np.zeros((op.n // 2, 2))
        v0 = np.zeros_like(u0)
        u0[:, component] = q
        v0[:, component] = -speed * derivative
        step = op.start(dt, u0=u0.ravel(), v0=v0.ravel())
        receiver = PointMap2D(op.V, [(1800, 0)])
        traces = np.zeros((len(time), 2))
        velocities = np.zeros_like(traces)
        zero = np.zeros(op.n)
        for i in range(len(time)):
            nxt, v, _, _ = step.evaluate(zero)
            sampled = receiver.evaluate(step.current)
            sampled_v = receiver.evaluate(v)
            if len(receiver.ids):
                traces[i] = sampled[0]
                velocities[i] = sampled_v[0]
            if i < len(time) - 1:
                step.advance(nxt)
        traces = comm.allreduce(traces)
        velocities = comm.allreduce(velocities)
    incident = abs(time - incident_time) <= half_window
    reflected = abs(time - reflected_time) <= half_window
    incident_peak = np.max(abs(traces[incident, component]))
    reflected_peak = np.max(abs(traces[reflected, component]))
    return dict(
        time=time,
        u=traces,
        v=velocities,
        incident_peak=float(incident_peak),
        reflected_peak=float(reflected_peak),
        reflection=float(reflected_peak / incident_peak),
        incident_time=incident_time,
        reflected_time=reflected_time,
    )


def seismic_config(boundaries=None, expanded=False, h=20):
    lower, upper = ((-3600, -4000), (3600, 0)) if expanded else ((-1200, -1600), (1200, 0))
    return SimulationConfig2D.model_validate(
        dict(
            domain=dict(
                lower=lower,
                upper=upper,
                cells=(round((upper[0] - lower[0]) / h), round((upper[1] - lower[1]) / h)),
            ),
            material=MATERIAL,
            time=dict(dt=0.001, duration=1.6),
            boundaries=boundaries or {},
            source=dict(
                position=(17.3, -400.7),
                direction=(1, 1),
                wavelet=dict(f0=5, amplitude=1e8, time_shift=0.3),
            ),
            receivers=[
                dict(name="below", position=(17.3, -600.7)),
                dict(name="right", position=(300.4, -450.2)),
                dict(name="left", position=(-270.5, -350.8)),
            ],
        )
    )


THREE_SIDES = dict(left="absorbing", right="absorbing", lower="absorbing")


def seismic_runs(comm=MPI.COMM_SELF):
    """Same physical grid/source: small free, three-side, enlarged and all-side control."""
    return {
        "free": Simulation2D(seismic_config(), comm).run(),
        "absorbing": Simulation2D(seismic_config(THREE_SIDES), comm).run(),
        "reference": Simulation2D(seismic_config(THREE_SIDES, expanded=True), comm).run(),
        "all_absorbing": Simulation2D(
            seismic_config(THREE_SIDES | dict(upper="absorbing")), comm
        ).run(),
    }


def seismic_metrics(runs):
    time = runs["free"].time
    early = (time >= 0.50) & (time <= 0.75)
    late = (time >= 0.95) & (time <= 1.6)
    a = runs["absorbing"].displacement
    f = runs["free"].displacement
    ref = runs["reference"].displacement
    all_a = runs["all_absorbing"].displacement
    return dict(
        late_error_ratio=float(np.linalg.norm((a - ref)[late]) / np.linalg.norm((f - ref)[late])),
        late_per_receiver=(
            np.linalg.norm((a - ref)[late], axis=(0, 2))
            / np.linalg.norm((f - ref)[late], axis=(0, 2))
        ).tolist(),
        surface_preservation=float(np.linalg.norm((a - f)[early]) / np.linalg.norm(f[early])),
        surface_reference_error=float(
            np.linalg.norm((a - ref)[early]) / np.linalg.norm(ref[early])
        ),
        surface_signal_fraction=float(
            np.linalg.norm((a - all_a)[early, 0]) / np.linalg.norm(a[early, 0])
        ),
    )


def reciprocity_runs(comm=MPI.COMM_SELF, dt=0.002, boundary_point=False):
    from tests.experiments2d.helpers import small_config

    A = (0.313, 0.487)
    B = (1.0, 0.723) if boundary_point else (0.679, 0.723)
    traces = []
    for source, receiver, direction, component in [
        (A, B, (1, 0), 0),
        (B, A, (1, 0), 0),
        (A, B, (0, 1), 0),
        (B, A, (1, 0), 1),
    ]:
        cfg = small_config(
            time=dict(dt=dt, duration=1.0),
            boundaries=THREE_SIDES,
            source=dict(
                position=source,
                direction=direction,
                wavelet=dict(f0=9, amplitude=2.1, time_shift=0.04),
            ),
            receivers=[dict(name="r", position=receiver)],
        )
        result = Simulation2D(cfg, comm).run()
        traces.append(
            np.stack((result.displacement[:, 0, component], result.velocity[:, 0, component]))
        )
    return np.array(traces)
