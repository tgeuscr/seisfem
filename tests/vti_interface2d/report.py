"""Collect fresh interface measurements and independently recompute reference tables.

python -m tests.vti_interface2d.report DIRECTORY OUTPUT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.oblique2d import reference as iso

from .packets import ANGLES
from .reference import LOWER, NAMES, UPPER, Material, candidates, residual, solve


def analytical():
    reports = {}
    for angle in ANGLES:
        p = np.sin(np.deg2rad(angle)) / 3000
        incident, branches = solve(p)
        roots = [candidates(p, m) for m in [LOWER, UPPER]]
        reports[f"{angle:g}"] = dict(
            p=float(p),
            incident_q=float(incident["q"]),
            determinant_residual=max(float(abs(w["determinant"]).max()) for w in roots),
            eigen_residual=max(float(w["residual"].max()) for w in roots),
            continuity_residual=float(abs(residual(p, branches)).max()),
            flux_closure=float(abs(sum(w["fraction"] for w in branches.values()) - 1)),
            branches={
                name: dict(
                    q=float(w["q"]),
                    polarization=w["d"].tolist(),
                    group=w["g"].tolist(),
                    phase_angle=float(np.degrees(np.arctan2(w["n"][0], abs(w["n"][1])))),
                    group_angle=float(np.degrees(np.arctan2(w["g"][0], abs(w["g"][1])))),
                    real=float(w["amplitude"].real),
                    imaginary=float(w["amplitude"].imag),
                    signed_unit_flux=float(w["flux"]),
                    fraction=float(w["fraction"]),
                )
                for name, w in branches.items()
            },
        )
    errors = dict(amplitude=0.0, flux=0.0, q=0.0, direction=0.0, polarization=0.0)
    upper = Material.isotropic(iso.UPPER.rho, iso.UPPER.vp, iso.UPPER.vs)
    for mode, angles in [("P", [0, 15, 25, 35]), ("S", [0, 5, 15, 20])]:
        for angle in angles:
            p = np.sin(np.deg2rad(angle)) / iso.LOWER.speed(mode)
            _, a = solve(p, incident=mode, upper=upper)
            b = iso.solve(mode, p=p)
            for name in NAMES:
                c = (iso.LOWER if name[0] == "R" else iso.UPPER).speed(name[1])
                for key, error in dict(
                    amplitude=abs(a[name]["amplitude"] - b[name]["amplitude"]),
                    flux=abs(a[name]["fraction"] - b[name]["flux"]),
                    q=abs(a[name]["q"] - b[name]["direction"][1] / c),
                    direction=np.max(abs(a[name]["n"] - b[name]["direction"])),
                    polarization=np.max(abs(a[name]["d"] - b[name]["polarization"])),
                ).items():
                    errors[key] = max(errors[key], float(error))
    return reports, errors


def collect(directory):
    names = [f"continuum-{a:g}" for a in ANGLES]
    names += [f"fem-{a:g}-{h}" for a in ANGLES for h in ([40, 30, 20] if abs(a) == 25 else [20])]
    names += [
        "packet-energy-0",
        "packet-energy-25",
        "packet-energy-35",
        "boundary-25",
        "boundary-35",
        "mpi-2",
        "mpi-4",
        "isotropic-matrices",
        "isotropic-experiment",
    ]
    data = {name: json.loads((directory / (name + ".json")).read_text()) for name in names}
    mpi = {}
    serial = data["fem-25-20"]
    for ranks in [2, 4]:
        other = data[f"mpi-{ranks}"]
        errors = {}
        for key in [
            "mass",
            "mass_square",
            "dt",
            "stable_dt",
            "stiffness_form",
            "action_square",
            "trace",
            "velocity_trace",
        ]:
            a, b = np.asarray(serial[key]), np.asarray(other[key])
            errors[key] = float(np.max(abs(a - b)) / max(np.max(abs(a)), 1e-30))
        for key in ["real", "imaginary", "flux", "phase_angle"]:
            errors[key] = max(
                abs(serial["branches"][n][key] - other["branches"][n][key]) for n in NAMES
            )
        mpi[str(ranks)] = errors
    for value in data.values():
        for key in ["trace", "velocity_trace"]:
            if key in value:
                value[key + "_peak"] = float(np.max(abs(np.asarray(value.pop(key)))))
    ref, limit = analytical()
    return dict(
        baseline="43aca95f199542429792f936284dd466055fa4eb",
        scope=(
            "P incidence from isotropic lower into vertical-axis VTI upper; "
            "single horizontal welded interface; propagating central branches only"
        ),
        analytical=ref,
        isotropic_reference_errors=limit,
        mpi_errors=mpi,
        measurements=data,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
