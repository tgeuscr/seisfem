"""Compare saved pre-fix packet/public-API runs with tensor-factorized assembly.

The baseline directory contains public-P.json from mpi_worker.measure(COMM_SELF)
and packet-S.json from experiment.run(4, 96, 'S', fractions=(0.02,), audit=True)[0],
executed using the archived 5023ae3 source and tests. Raw baselines stay outside
the repository; the compact output records their comparison, not their timings.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from .experiment import run
from .mpi_worker import measure
from .test_mpi import compare


def coordinate_differences(before, after):
    a, b = np.asarray(before["coordinate_audit"]), np.asarray(after["coordinate_audit"])
    np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=0, atol=1e-9)
    result = {"coordinate_absolute": float(np.max(abs(a[:, :2] - b[:, :2])))}
    for name, columns in [
        ("mass", slice(2, 4)),
        ("stiffness_action", slice(4, 6)),
        ("final_u", slice(6, 8)),
        ("final_v", slice(8, 10)),
    ]:
        error = float(np.max(abs(a[:, columns] - b[:, columns])))
        scale = float(np.max(abs(a[:, columns])))
        assert error <= 3e-11 * scale, (name, error, scale)
        result[name] = dict(absolute=error, relative=error / scale)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before = json.loads((args.baseline / "public-P.json").read_text())
    after = measure(MPI.COMM_SELF)
    result = dict(
        baseline_commit="5023ae3824461547d9d51f7872eae7200691c122",
        P=compare(before, after),
        P_fields=coordinate_differences(before["packet"], after["packet"]),
    )
    before = json.loads((args.baseline / "packet-S.json").read_text())
    after = run(4, 96, "S", fractions=(0.02,), audit=True)[0]
    result.update(SV=compare(before, after), SV_fields=coordinate_differences(before, after))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
