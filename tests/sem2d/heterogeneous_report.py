"""Collect the heterogeneous SEM audit without regenerating homogeneous evidence."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = {
        path.stem.removeprefix("heterogeneous-"): json.loads(path.read_text())
        for path in sorted(args.directory.glob("heterogeneous-*.json"))
    }
    required = (
        [f"operators-{p}" for p in range(1, 7)]
        + [f"homogeneous-{p}" for p in [1, 2, 4, 6]]
        + [f"packet-{case}-h{h}" for case in ["increase", "decrease"] for h in [100, 50]]
        + ["packet-identical", "triangle-h10", "triangle-h5", "mpi-2", "mpi-4", "p1-compatibility"]
    )
    assert set(required) <= records.keys(), set(required) - records.keys()
    report = dict(
        baseline="d87afcdc961f03e0cfd146b66309ecaba37b3fd6",
        scope="element-aligned horizontal isotropic layers, free boundaries, assembled GLL SEM",
        sum_factorization=True,
        records=records,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
