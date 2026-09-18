import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from .conftest import record


def compare(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], path + "." + key)
    elif a is None:
        assert b is None
    else:
        np.testing.assert_allclose(a, b, rtol=3e-10, atol=3e-12, err_msg=path)


@pytest.mark.mpi
def test_one_two_four_rank_anisotropic_scattering(refinement, tmp_path):
    serial = refinement[(25.0, 20)]
    for ranks in [2, 4]:
        output = tmp_path / f"{ranks}.json"
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.vti_interface2d.experiment",
                "--angle",
                "25",
                "--h",
                "20",
                "--output",
                str(output),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
        result = json.loads(output.read_text())
        compare(serial, result)
        record(f"mpi-{ranks}", result)
