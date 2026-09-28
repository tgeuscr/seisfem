import json
import os
import subprocess
import sys

import numpy as np
import pytest
from mpi4py import MPI

from .absorbing_mpi import measure
from .helpers import record


@pytest.mark.mpi
def test_absorbing_serial_two_four_ranks(tmp_path):
    serial = measure(MPI.COMM_SELF)
    for ranks in [2, 4]:
        path = tmp_path / f"heterogeneous-{ranks}.json"
        process = subprocess.run(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.sem2d.absorbing_mpi",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        assert process.returncode == 0, process.stdout + process.stderr
        parallel = json.loads(path.read_text())
        errors = {}
        for name in ["cells", "facets", "u", "v", "safe_dt", "dtcrit"]:
            a, b = np.asarray(serial[name]), np.asarray(parallel[name])
            error = float(np.max(abs(a - b)) / max(np.max(abs(a)), 1e-30))
            assert error < 3e-11
            errors[name] = error
        a, b = np.array(serial["fields"]), np.array(parallel["fields"])
        np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=0, atol=1e-12)
        for name, cols in [
            ("mass", slice(2, 4)),
            ("action", slice(4, 6)),
            ("final_u", slice(6, 8)),
            ("final_v", slice(8, 10)),
            ("damping", slice(10, 12)),
            ("damping_action", slice(12, 14)),
        ]:
            error = float(np.max(abs(a[:, cols] - b[:, cols])) / np.max(abs(a[:, cols])))
            assert error < 3e-11
            errors[name] = error
        assert all(n > 0 for n in parallel["facet_counts"])
        # Each layer is distributed over multiple ranks, not isolated to one rank.
        assert sum(1 in row for row in parallel["rank_layers"]) >= 2
        assert sum(0 in row or 2 in row for row in parallel["rank_layers"]) >= 2
        record(f"absorbing-mpi-{ranks}", dict(errors=errors, rank_layers=parallel["rank_layers"]))
