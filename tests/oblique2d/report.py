"""Collect freshly generated pytest evidence (does not run or replace tests).

python -m tests.oblique2d.report /tmp/oblique-report docs/validation/2d_oblique_measurements.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .packets import ANGLES, CENTER, EXTENT, FREQUENCY, SIGMA_Q, TIMES, parameters
from .reference import NAMES, solve


def collect(directory):
    def read(name):
        return json.loads((directory / (name + ".json")).read_text())

    evidence = dict(
        baseline="v0.5.0 (76e657088370890cfef024554846da82033e2ae7)",
        frequency=FREQUENCY,
        sigma_q=SIGMA_Q,
        center=CENTER.tolist(),
        domain=[[-EXTENT, -EXTENT], [EXTENT, EXTENT]],
        amplitude_error_definition="abs(complex coefficient - real reference)",
        signed_error_definition="abs(real part of coefficient - real reference)",
        flux_definition="central spectral component: rho*c*abs(nz)*abs(A)**2 / incident",
        cases={},
    )
    for mode in ["P", "S"]:
        rows = [read(f"{mode}-{h}") for h in [40, 30, 20]]
        case = dict(
            angle=ANGLES[mode],
            sigma_s=parameters(mode)[4],
            time=TIMES[mode],
            reference=solve(mode, ANGLES[mode]),
            continuum=read("continuum-" + mode),
            refinement=rows,
            boundary_sensitivity=read("boundary-" + mode),
            mpi={},
        )
        serial = rows[-1]
        for ranks in [2, 4]:
            result = read(f"MPI-{mode}-{ranks}")
            differences = {}
            for key in ["angle", "amplitude", "imaginary", "flux"]:
                differences[key] = max(
                    abs(serial["branches"][name][key] - result["branches"][name][key])
                    for name in NAMES
                )
            differences["trace"] = float(
                np.max(abs(np.array(serial["trace"]) - np.array(result["trace"])))
            )
            for key in [
                "stable_dt",
                "dt",
                "purity",
                "mass",
                "mass_square",
                "stiffness_form",
                "action_square",
            ]:
                a, b = np.asarray(serial[key]), np.asarray(result[key])
                differences[key + "_relative"] = float(np.max(abs(a - b)) / np.max(abs(a)))
            case["mpi"][str(ranks)] = dict(differences=differences, result=result)
        evidence["cases"][mode] = case
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
