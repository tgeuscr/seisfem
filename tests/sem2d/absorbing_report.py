"""Collect SEM absorber evidence; preserve previous milestone measurements."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = {
        p.stem.removeprefix("absorbing-"): json.loads(p.read_text())
        for p in sorted(args.directory.glob("absorbing-*.json"))
    }
    required = ["compatibility", "source", "mpi-2", "mpi-4"]
    required += [f"normal-{m}-h{h}" for m in ["P", "S"] for h in [100, 50]]
    required += [f"oblique-{m}" for m in ["P", "S"]]
    required += [f"triangle-{m}" for m in ["P", "S"]]
    required += ["local-P-0", "local-S-1"] + [f"energy-p{p}" for p in [1, 2, 4, 6]]
    assert set(required) <= records.keys(), set(required) - records.keys()
    assert sum(k.startswith("operator-") for k in records) == 32
    result = dict(
        baseline="edaec57c75fb2303c8c28860dcded46b6d0d0fa1",
        volume_sum_factorization=True,
        facet_sum_factorization=False,
        boundary="sigma*n + [Zs*I+(Zp-Zs)*n*n^T]*v = 0",
        records=records,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
