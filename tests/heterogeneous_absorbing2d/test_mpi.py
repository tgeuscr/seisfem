import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from .evidence import record


@pytest.mark.mpi
def test_one_two_four_rank_facet_assembly_and_surface_well_traces(tmp_path):
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
                "tests.heterogeneous_absorbing2d.mpi_worker",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
        with np.load(path) as data:
            results.append({key: data[key].copy() for key in data})
    report = {}
    for ranks, result in zip([2, 4], results[1:], strict=True):
        errors = {}
        for key, reference in results[0].items():
            difference = float(np.max(abs(result[key] - reference)))
            scale = float(np.max(abs(reference)))
            assert difference < 3e-12 * scale, key
            errors[key] = dict(absolute=difference, relative_to_peak=difference / scale)
        report[str(ranks)] = errors
    record("mpi", report)
