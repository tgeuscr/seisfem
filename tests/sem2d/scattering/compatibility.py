"""Reproduce archived/current snapshots without switching or modifying the checkout."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

BASELINE = "18be5ea031c42ecd6f138dfb536ef0bd771f5c04"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    rows = {}
    with tempfile.TemporaryDirectory(prefix="sem-scattering-compatibility-") as temporary:
        root = Path(temporary)
        archive = root / "src.tar"
        subprocess.run(
            ["git", "archive", "-o", str(archive), BASELINE, "src"], cwd=repository, check=True
        )
        with tarfile.open(archive) as file:
            file.extractall(root, filter="data")
        for label, script, extra in [
            ("sem-free", "tests/sem2d/absorbing_compatibility.py", ["free"]),
            ("sem-layered", "tests/sem2d/absorbing_compatibility.py", ["layered"]),
            ("tri-free", "tests/sem2d/compatibility.py", []),
            ("tri-layered", "tests/heterogeneous2d/mpi_worker.py", []),
        ]:
            for version, source in [("baseline", root / "src"), ("current", repository / "src")]:
                output = root / f"{label}-{version}.npz"
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(source) + os.pathsep + str(repository),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                }
                subprocess.run(
                    [sys.executable, script, str(output), *extra],
                    cwd=repository,
                    env=environment,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            with (
                np.load(root / f"{label}-baseline.npz") as a,
                np.load(root / f"{label}-current.npz") as b,
            ):
                assert set(a.files) == set(b.files)
                rows[label] = {
                    k: dict(
                        bitwise=bool(np.array_equal(a[k], b[k])),
                        absolute_error=float(np.max(abs(a[k] - b[k]))),
                    )
                    for k in a.files
                }
            assert all(r["bitwise"] for r in rows[label].values()), rows[label]
            print(label, "bitwise identical", flush=True)
    (args.directory / "scattering-compatibility.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
