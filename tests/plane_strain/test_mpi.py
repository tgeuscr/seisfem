import os
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
