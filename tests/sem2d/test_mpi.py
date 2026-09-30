import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest
from mpi4py import MPI

from .helpers import record
from .mpi_worker import measure

TIMINGS = {
    "assembly_seconds",
    "eigensolve_seconds",
    "stepping_seconds",
    "solve_seconds",
    "backend_seconds",
    "matrix_bytes",
}


def compare(a, b, path="", differences=None):
    if differences is None:
        differences = {}
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key not in TIMINGS and key != "coordinate_audit":
                compare(a[key], b[key], path + "." + key, differences)
    elif isinstance(a, str):
        assert a == b
    else:
        x, y = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(x, y, rtol=3e-10, atol=3e-11, err_msg=path)
        maximum = float(np.max(abs(x - y)))
        differences[path] = dict(
            absolute=maximum, relative=maximum / max(float(np.max(abs(x))), 1e-30)
        )
    return differences


@pytest.mark.mpi
def test_sem_serial_two_four_ranks(tmp_path):
    serial = measure(MPI.COMM_SELF)
    for ranks in [2, 4]:
        output = tmp_path / f"{ranks}.json"
        with subprocess.Popen(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.sem2d.mpi_worker",
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
        parallel = json.loads(output.read_text())
        differences = compare(serial, parallel)
        # Separate physically ordered mass, action, displacement and velocity errors.
        a = np.asarray(serial["packet"]["coordinate_audit"])
        b = np.asarray(parallel["packet"]["coordinate_audit"])
        np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=0, atol=1e-9)
        for name, cols in [
            ("diagonal_mass", slice(2, 4)),
            ("stiffness_action", slice(4, 6)),
            ("final_u", slice(6, 8)),
            ("final_v", slice(8, 10)),
        ]:
            maximum = float(np.max(abs(a[:, cols] - b[:, cols])))
            scale = float(np.max(abs(a[:, cols])))
            # Ku contains near-zero entries from cancellation of large elastic
            # terms. Compare against the physical field's global scale, not a
            # pointwise relative error at those zeros.
            assert maximum < 3e-11 * scale, (name, maximum, scale)
            differences[name] = dict(absolute=maximum, relative=maximum / scale)
        record(f"mpi-{ranks}", differences)
