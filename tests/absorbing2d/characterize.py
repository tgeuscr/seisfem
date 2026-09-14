"""Optional resolved-frequency and oblique-incidence study, outside routine pytest.

python -m tests.absorbing2d.characterize --output /tmp/seisfem-absorber-study.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.absorbing2d.experiments import MATERIAL, packet_run


def continuum_reflection(angle, absorbing=True):
    """Independent harmonic P-incidence traction solve at a right boundary.

    Polarizations are unit vectors; slowness p obeys |p|=1/c. For exp(i*omega*
    (p.x-t)), traction/(i*omega) = lambda*(p.d)*n + mu*(d*p.n+p*d.n).
    Solve two equations for reflected P and SV amplitudes. Unlike a receiver
    squared-trace proxy, the returned energy ratio uses actual plane-wave flux.
    """
    rho, cp, cs = 2400.0, 3200.0, 1800.0
    lam, mu = rho * (cp**2 - 2 * cs**2), rho * cs**2
    theta = np.deg2rad(angle)
    tangent_slowness = np.sin(theta) / cp
    normal = np.array([1.0, 0.0])
    B = np.diag([rho * cp, rho * cs]) if absorbing else np.zeros((2, 2))
    incident = np.array([np.cos(theta), np.sin(theta)])
    p_incident = incident / cp
    p_p = np.array([-np.sqrt(1 / cp**2 - tangent_slowness**2), tangent_slowness])
    d_p = cp * p_p
    p_s = np.array([-np.sqrt(1 / cs**2 - tangent_slowness**2), tangent_slowness])
    d_s = cs * np.array([p_s[1], -p_s[0]])

    def boundary_residual(p, d):
        return (
            lam * np.dot(p, d) * normal
            + mu * (d * np.dot(p, normal) + p * np.dot(d, normal))
            - B @ d
        )

    matrix = np.column_stack((boundary_residual(p_p, d_p), boundary_residual(p_s, d_s)))
    amplitudes = np.linalg.solve(matrix, -boundary_residual(p_incident, incident))
    flux_ratio = (
        amplitudes[0] ** 2
        + cs * np.sqrt(1 - (cs * tangent_slowness) ** 2) / (cp * np.cos(theta)) * amplitudes[1] ** 2
    )
    return dict(
        angle=angle,
        P=float(amplitudes[0]),
        SV=float(amplitudes[1]),
        energy_flux_ratio=float(flux_ratio),
    )


def beam_run(angle, boundary="absorbing", h=20):
    """P packet from grad Phi, Phi=s*exp(-s²/w²-q²/W²).

    Both u0 and v0=-Vp*grad(dPhi/ds) are analytically irrotational. The finite
    beam has angular spread and diffraction; it is not an exact single-angle
    plane wave. An enlarged-domain run removes the incident field in the metric.
    """
    cp = 3200.0
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-direction[1], direction[0]])
    center = np.array([1600.0, 0.0]) - 2000 * direction
    receiver = np.array([1600.0, 0.0]) + 600 * np.array([-direction[0], direction[1]])
    right = 4000 if boundary == "reference" else 1600
    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(
                lower=(-1600, -3600),
                upper=(right, 3600),
                cells=(round((right + 1600) / h), round(7200 / h)),
            ),
            material=MATERIAL,
            boundaries=dict(
                left="absorbing",
                lower="absorbing",
                upper="absorbing",
                right="free" if boundary == "free" else "absorbing",
            ),
        )
    )
    dt = 0.001 if h >= 10 else 0.0005
    time = np.arange(round(1.1 / dt) + 1) * dt
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        relative = op.coordinates[: op.n // 2] - center
        s = relative @ direction
        q = relative @ tangent
        width = cp / (np.pi * 8)
        transverse_width = 400
        exponential = np.exp(-((s / width) ** 2) - (q / transverse_width) ** 2)
        phi_s = (1 - 2 * (s / width) ** 2) * exponential
        phi_q = -2 * q * s / transverse_width**2 * exponential
        phi_ss = (-6 * s / width**2 + 4 * s**3 / width**4) * exponential
        phi_sq = -2 * q / transverse_width**2 * (1 - 2 * (s / width) ** 2) * exponential
        u0 = phi_s[:, None] * direction + phi_q[:, None] * tangent
        v0 = -cp * (phi_ss[:, None] * direction + phi_sq[:, None] * tangent)
        step = op.start(dt, u0=u0.ravel(), v0=v0.ravel())
        point = PointMap2D(op.V, [receiver])
        trace = np.empty((len(time), 2))
        zero = np.zeros(op.n)
        for i in range(len(time)):
            nxt, _, _, _ = step.evaluate(zero)
            trace[i] = point.evaluate(step.current)[0]
            if i < len(time) - 1:
                step.advance(nxt)
    return time, trace


def beam_comparison(angle, h=20):
    time, reference = beam_run(angle, "reference", h)
    _, free = beam_run(angle, "free", h)
    _, absorbing = beam_run(angle, "absorbing", h)
    window = (time >= 0.70) & (time <= 1.05)
    free_peak = np.max(np.linalg.norm((free - reference)[window], axis=1))
    abs_peak = np.max(np.linalg.norm((absorbing - reference)[window], axis=1))
    return dict(
        angle=angle,
        h=h,
        free_scattered_peak=float(free_peak),
        absorbing_scattered_peak=float(abs_peak),
        amplitude_proxy=float(abs_peak / free_peak),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-frequency", action="store_true")
    parser.add_argument("--skip-beams", action="store_true")
    args = parser.parse_args()
    report = dict(
        continuum=[continuum_reflection(a) for a in [0, 15, 30, 45, 60, 75]], frequency=[], beams=[]
    )
    for row in report["continuum"]:
        print("continuum", row, flush=True)
    if not args.skip_frequency:
        for mode, h in [("P", 8), ("S", 5)]:
            for frequency in [6, 8, 10]:
                result = packet_run(mode, h, frequency=frequency)
                speed = MATERIAL["vp" if mode == "P" else "vs"]
                row = dict(
                    mode=mode,
                    h=h,
                    f0=frequency,
                    points_per_3f0_wavelength=speed / (3 * frequency * h),
                    reflection=result["reflection"],
                    incident_peak=result["incident_peak"],
                )
                report["frequency"].append(row)
                print("frequency", row, flush=True)
    if not args.skip_beams:
        for angle, h in [(0, 20), (45, 20), (60, 20), (60, 10)]:
            row = beam_comparison(angle, h)
            report["beams"].append(row)
            print("beam", row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
