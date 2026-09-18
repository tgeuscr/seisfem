"""Package fresh test diagnostics; this collector never substitutes for simulation.

python -m tests.heterogeneous_absorbing2d.report /tmp/ha-report OUTPUT.json
"""

import argparse
import json
from pathlib import Path

from .packets import FREQUENCY, HIGH, LOW, TRANSVERSE_WIDTH


def collect(directory):
    names = ["energy", "example", "homogeneous-1", "homogeneous-3", "mpi"]
    names += [
        f"facet-{side}-{diagonal}"
        for side in ["left", "right", "lower", "upper"]
        for diagonal in ["left", "right", "left_right", "right_left"]
    ]
    names += [f"packet-{m}-{layer}-right-0" for m in ["P", "S"] for layer in [0, 1]]
    names += [f"packet-{m}-0-lower-0" for m in ["P", "S"]]
    names += [f"packet-P-1-right-{a}" for a in [15, 30, 45]]
    return dict(
        baseline="f467b1d6de3178391c61e9d92a58fe36ef04d64b",
        boundary="local first-order isotropic impedance; not a PML",
        packets=dict(
            lower=LOW,
            upper=HIGH,
            frequency=FREQUENCY,
            h=20,
            dt=0.002,
            transverse_width=TRANSVERSE_WIDTH,
            incident_path=1200,
            reflected_receiver_path=400,
            reflection_proxy="peak norm(absorbing-extended) / peak norm(free-extended)",
            window="abs(t-1600/c_incident) <= 0.8/f0",
        ),
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
