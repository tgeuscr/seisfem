import os
import signal
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.mpi
def test_one_two_four_rank_materials_operators_and_traces(tmp_path):
    results = []
    for ranks in [1, 2, 4]:
        path = tmp_path / f"{ranks}.npz"
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.heterogeneous2d.mpi_worker",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
            print(stdout.strip())
        with np.load(path) as data:
            results.append({key: data[key].copy() for key in data})
    for ranks, result in zip([2, 4], results[1:], strict=True):
        for key, reference in results[0].items():
            difference = np.max(abs(result[key] - reference))
            scale = np.max(abs(reference))
            print("MPI", ranks, key, "absolute", difference, "relative to peak", difference / scale)
            assert difference < 3e-12 * scale
        np.testing.assert_array_equal(result["cells"][:, 2:], results[0]["cells"][:, 2:])
        np.testing.assert_array_equal(
            result["receiver_coordinates"], results[0]["receiver_coordinates"]
        )
