import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from .conftest import record


def compare(a, b, path=""):
    """Compare every diagnostic, including signed complex coefficients and traces."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], path + "." + key)
    elif isinstance(a, str):
        assert a == b
    else:
        np.testing.assert_allclose(a, b, rtol=3e-10, atol=3e-12, err_msg=path)


@pytest.mark.mpi
@pytest.mark.parametrize("mode", ["P", "S"])
def test_resolved_serial_two_four_ranks(refinement, tmp_path, mode):
    serial = refinement[mode][-1]
    for ranks in [2, 4]:
        output = tmp_path / f"{mode}-{ranks}.json"
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.oblique2d.experiment",
                "--mode",
                mode,
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
                stdout, stderr = process.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
        result = json.loads(output.read_text())
        record(f"MPI-{mode}-{ranks}", result)
        compare(serial, result)
