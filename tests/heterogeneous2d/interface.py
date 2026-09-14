"""Small, fixed-bandwidth normal-P interface study in 2D and the audited 1D core.

python -m tests.heterogeneous2d.interface --output /tmp/seisfem-interface.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from seisfem import SimulationConfig
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem1d import IntervalOperators
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points import PointMap
from seisfem.points2d import PointMap2D
from seisfem.timestepping import CentralDifference
from tests.heterogeneous2d.helpers import layers

FIRST = dict(density=2000, vp=2000, vs=1000)
SECOND = dict(density=2400, vp=3000, vs=1500)
FREQUENCY = 6.0
DT = 0.0005
DURATION = 1.4
STATIONS = (-800.3, 800.3)


def packet(z):
    """u_z=Phi_z, Phi=(z+1600) exp(-((z+1600)/w)^2), v_z=-Vp1 u_z,z."""
    width = FIRST["vp"] / (np.pi * FREQUENCY)
    s = (z + 1600) / width
    displacement = (1 - 2 * s * s) * np.exp(-s * s)
    velocity = -FIRST["vp"] * (-6 * s + 4 * s**3) * np.exp(-s * s) / width
    return displacement, velocity


def run(h=20, dimension=2, contrast=True, homogeneous=False):
    second = SECOND if contrast else FIRST
    material = layers(-4000, 0, 4000, FIRST, second)
    time = np.arange(round(DURATION / DT) + 1) * DT
    if dimension == 2:
        cfg = PlaneStrainConfig.model_validate(
            dict(
                domain=dict(
                    lower=(-400, -4000), upper=(400, 4000), cells=(round(800 / h), round(8000 / h))
                ),
                material=FIRST if homogeneous else material,
                constraints=[dict(side=side, components=["x"]) for side in ["left", "right"]],
            )
        )
        op = PlaneStrainOperators(cfg, MPI.COMM_SELF)
        u0, v0 = np.zeros((op.n // 2, 2)), np.zeros((op.n // 2, 2))
        u0[:, 1], v0[:, 1] = packet(op.coordinates[: op.n // 2, 1])
        step = op.start(DT, u0=u0.ravel(), v0=v0.ravel())
        points = PointMap2D(op.V, [(17.3, z) for z in STATIONS])
    else:
        cfg = SimulationConfig.model_validate(
            dict(
                mode="P",
                domain=dict(lower=-4000, upper=4000),
                mesh=dict(cells=round(8000 / h)),
                materials=material,
                time=dict(dt=DT, duration=DURATION),
            )
        )
        op = IntervalOperators(cfg, MPI.COMM_SELF)
        u0, v0 = packet(op.V.tabulate_dof_coordinates()[: op.n, 0])
        step = CentralDifference(
            op.mass, op.damping, op.fixed, DT, op.apply, u0, v0, np.zeros(op.n)
        )
        points = PointMap(op.V, STATIONS, h)
    trace = np.empty((len(time), 2))
    transverse = 0.0
    zero = np.zeros(op.n)
    try:
        for i in range(len(time)):
            nxt, _, _, _ = step.evaluate(zero)
            sampled = points.evaluate(step.current)
            if dimension == 2:
                trace[i] = sampled[:, 1]
                transverse = max(transverse, float(abs(sampled[:, 0]).max()))
            else:
                trace[i] = sampled
            if i < len(time) - 1:
                step.advance(nxt)
    finally:
        op.close()
    return time, trace, transverse


def analytical(contrast=True):
    second = SECOND if contrast else FIRST
    Z1 = FIRST["density"] * FIRST["vp"]
    Z2 = second["density"] * second["vp"]
    R = (Z1 - Z2) / (Z1 + Z2)
    T = 2 * Z1 / (Z1 + Z2)
    return dict(R=R, T=T, R_energy=R**2, T_energy=Z2 / Z1 * T**2)


def metrics(time, trace, contrast=True):
    speed = SECOND["vp"] if contrast else FIRST["vp"]
    centers = [
        (STATIONS[0] + 1600) / FIRST["vp"],
        (1600 - STATIONS[0]) / FIRST["vp"],
        1600 / FIRST["vp"] + STATIONS[1] / speed,
    ]
    amplitudes = []
    for receiver, center in zip([0, 0, 1], centers, strict=True):
        window = abs(time - center) <= 0.8 / FREQUENCY
        a = np.pi * FREQUENCY * (time[window] - center)
        template = (1 - 2 * a * a) * np.exp(-a * a)
        amplitudes.append(float(template @ trace[window, receiver] / (template @ template)))
    return dict(
        incident=amplitudes[0], R=amplitudes[1] / amplitudes[0], T=amplitudes[2] / amplitudes[0]
    )


def study():
    report = dict(
        material1=FIRST,
        material2=SECOND,
        f0=FREQUENCY,
        dt=DT,
        analytical=analytical(),
        refinement=[],
    )
    for h in [20, 10, 5]:
        time, two, transverse = run(h, 2)
        _, one, _ = run(h, 1)
        m2, m1 = metrics(time, two), metrics(time, one)
        row = dict(
            h=h,
            two_dimensional=m2,
            one_dimensional=m1,
            R_error=abs(m2["R"] - report["analytical"]["R"]),
            T_error=abs(m2["T"] - report["analytical"]["T"]),
            trace_difference=float(np.linalg.norm(two - one) / np.linalg.norm(one)),
            transverse_peak=transverse,
        )
        report["refinement"].append(row)
        print("interface", json.dumps(row), flush=True)
    time, layered, _ = run(20, 2, contrast=False)
    _, homogeneous, _ = run(20, 2, contrast=False, homogeneous=True)
    report["zero_contrast"] = metrics(time, layered, contrast=False) | dict(
        layered_vs_homogeneous=float(
            np.linalg.norm(layered - homogeneous) / np.linalg.norm(homogeneous)
        ),
        spurious_interface_peak=float(np.max(abs(layered - homogeneous))),
    )
    print("zero contrast", json.dumps(report["zero_contrast"]), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = study()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
