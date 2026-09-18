"""Collect freshly computed VTI evidence; never used as a cached test oracle.

python -m tests.vti2d.report REPORT_DIRECTORY OUTPUT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .packets import EXTENT, spectrum


def collect(directory):
    names = [f"reference-{a}" for a in [0, 15, 35, 60, 90, -15, -35, -60, -90]]
    names += [
        f"assembly-{layered}-{diagonal}"
        for layered in [False, True]
        for diagonal in ["left", "right", "left_right", "right_left"]
    ]
    names += [f"isotropic-{i}" for i in [0, 1, 2]]
    names += ["stability-1", "stability-30", "mpi", "example", "boundary-P", "boundary-S"]
    bandwidths = {}
    for branch in ["P", "S"]:
        for indices in [(0, 16), (16, 0), (8, 12), (-8, 12)]:
            names += [
                f"packet-{branch}-{indices[0]}-{indices[1]}-{h}"
                for h in ([40, 30, 20] if indices[1] == 12 else [20])
            ]
            A, omega, reference = spectrum(branch, indices, 20)
            power = np.sum(abs(A) ** 2, axis=-1)
            power /= power.sum()
            freq = np.fft.fftfreq(len(A), 20) * 2 * np.pi
            x, z = np.meshgrid(freq, freq, indexing="ij")
            angle = np.arctan2(x, z) - np.arctan2(*reference["k0"])
            angle = np.angle(np.exp(1j * angle))
            omega0 = reference["speed"] * np.linalg.norm(reference["k0"])
            bandwidths[f"{branch}-{indices[0]}-{indices[1]}"] = dict(
                angular_rms_degrees=float(np.rad2deg(np.sqrt(np.sum(power * angle**2)))),
                frequency_rms_fraction=float(
                    np.sqrt(np.sum(power * (omega - omega0) ** 2)) / omega0
                ),
                central_frequency_hz=float(omega0 / (2 * np.pi)),
            )
    return dict(
        baseline="bbf07f5dbc2f639741b20c0f1620fce96ddc7677",
        scope=(
            "2D vertical-axis VTI plane strain; "
            "no anisotropic absorbers or validated interface amplitudes"
        ),
        packet_extent=EXTENT,
        bandwidths=bandwidths,
        measurements={
            name: json.loads((directory / (name + ".json")).read_text()) for name in names
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
