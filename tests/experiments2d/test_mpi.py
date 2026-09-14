import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from tests.experiments2d.helpers import relative_error


@pytest.fixture(scope="module")
def distributed_results(tmp_path_factory):
    directory = tmp_path_factory.mktemp("experiments2d-mpi")
    results = []
    for ranks in [1, 2, 4]:
        destination = directory / f"{ranks}.npz"
        # Kill the entire MPI process group if coherent error handling regresses.
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.experiments2d.mpi_worker",
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
    return results


@pytest.mark.mpi
@pytest.mark.parametrize("index", [1, 2])
def test_partition_independence(distributed_results, index):
    serial, parallel = distributed_results[0], distributed_results[index]
    for key in serial:
        if key in ["wave_u", "wave_v", "awkward_u", "awkward_v", "reciprocity"]:
            scale = np.max(abs(serial[key]))
            maximum = np.max(abs(serial[key] - parallel[key]))
            print("MPI", [1, 2, 4][index], key, "max_abs", maximum, "scaled_max", maximum / scale)
            assert maximum < 3e-11 * scale
        else:
            np.testing.assert_allclose(serial[key], parallel[key], atol=3e-12, rtol=3e-12)
    assert np.max(parallel["affine_errors"]) < 2e-15
    for i, j in [(0, 1), (2, 3)]:
        for observable in [0, 1]:
            assert (
                relative_error(
                    parallel["reciprocity"][i, observable], parallel["reciprocity"][j, observable]
                )
                < 2e-12
            )
