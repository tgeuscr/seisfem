import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D

from .helpers import record


@pytest.mark.mpi
def test_homogeneous_and_mixed_vti_one_two_four_ranks(tmp_path):
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
                "tests.vti2d.mpi_worker",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            assert process.returncode == 0, stdout + stderr
        with np.load(path) as data:
            results.append({k: data[k].copy() for k in data})
    errors = {}
    for ranks, result in zip([2, 4], results[1:], strict=True):
        errors[str(ranks)] = {}
        for key, ref in results[0].items():
            error = float(np.max(abs(ref - result[key])) / np.max(abs(ref)))
            assert error < 3e-12, (key, error)
            errors[str(ranks)][key] = error
    record("mpi", errors)


def test_public_vti_example():
    path = Path(__file__).resolve().parents[2] / "examples/2d/vti.py"
    spec = importlib.util.spec_from_file_location("vti_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    results = [
        Simulation2D(example.configuration(vti), MPI.COMM_SELF).run() for vti in [False, True]
    ]
    differences = {
        key: float(
            np.linalg.norm(getattr(results[1], key) - getattr(results[0], key))
            / np.linalg.norm(getattr(results[0], key))
        )
        for key in ["displacement", "velocity"]
    }
    assert all(np.isfinite(value) and value > 0.1 for value in differences.values())
    record("example", differences)
