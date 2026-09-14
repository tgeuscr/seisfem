import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from tests.experiments2d.helpers import relative_error


@pytest.mark.mpi
def test_serial_two_four_rank_absorbers(tmp_path):
    results = []
    for ranks in [1, 2, 4]:
        destination = tmp_path / f"{ranks}.npz"
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.absorbing2d.mpi_worker",
                str(destination),
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
            print(stdout.strip())
        with np.load(destination) as data:
            results.append({key: data[key].copy() for key in data})
    for ranks, result in zip([2, 4], results[1:], strict=True):
        for key, reference in results[0].items():
            if key == "fixed":
                np.testing.assert_array_equal(reference, result[key])
                continue
            difference = np.max(abs(result[key] - reference))
            scale = np.max(abs(reference))
            print("MPI", ranks, key, "max difference", difference, "scaled", difference / scale)
            assert difference < 3e-11 * scale
        for i, j in [(0, 1), (2, 3)]:
            for k in [0, 1]:
                error = relative_error(result["reciprocity"][i, k], result["reciprocity"][j, k])
                print("MPI reciprocity", ranks, i, k, error)
                assert error < 2e-12
