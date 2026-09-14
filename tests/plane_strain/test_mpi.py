import os
import signal
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_distributed_vector_kernel_matches_serial(tmp_path, ranks):
    outputs = []
    for count in [1, ranks]:
        destination = tmp_path / f"{count}.npz"
        completed = subprocess.run(
            [
                "mpiexec",
                "-n",
                str(count),
                sys.executable,
                "-m",
                "tests.plane_strain.mpi_worker",
                str(destination),
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        with np.load(destination) as data:
            outputs.append({key: data[key].copy() for key in data})
    for key in outputs[0]:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=3e-11, atol=3e-11)


@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_collective_validation_and_empty_ownership(ranks):
    # Kill the entire process group on timeout, including workers stuck in MPI.
    with subprocess.Popen(
        ["mpiexec", "-n", str(ranks), sys.executable, "-m", "tests.plane_strain.mpi_edge_worker"],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        assert process.returncode == 0, stdout + stderr
    print(stdout.strip())
