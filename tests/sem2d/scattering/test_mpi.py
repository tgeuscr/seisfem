"""Coordinate-ordered equivalence of an oblique packet through a partitioned interface."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from mpi4py import MPI

from tests.sem2d.helpers import record

from .experiment import run
from .packets import Packet
from .reference import NAMES


@pytest.mark.mpi
def test_oblique_packet_one_two_four_ranks(tmp_path):
    # Coarser than the accuracy study; this gate tests partition equivalence.
    serial, _ = run(Packet("P"), h=600, comm=MPI.COMM_SELF, audit=True)
    for ranks in [2, 4]:
        path = tmp_path / f"ranks-{ranks}.json"
        process = subprocess.run(
            [
                "mpiexec",
                "-n",
                str(ranks),
                sys.executable,
                "-m",
                "tests.sem2d.scattering.experiment",
                str(path),
                "--h",
                "600",
                "--audit",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        assert process.returncode == 0, process.stdout + process.stderr
        parallel = json.loads(path.read_text())
        a, b = np.array(serial["coordinate_audit"]), np.array(parallel["coordinate_audit"])
        np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=0, atol=1e-8)
        errors = {}
        for name, columns in [
            ("mass", slice(2, 4)),
            ("action", slice(4, 6)),
            ("final_u", slice(6, 8)),
            ("final_v", slice(8, 10)),
            ("damping", slice(10, 12)),
        ]:
            errors[name] = float(
                np.max(abs(a[:, columns] - b[:, columns])) / max(np.max(abs(a[:, columns])), 1e-30)
            )
        for name in ["safe_dt", "trace", "material_audit"]:
            a, b = np.asarray(serial[name]), np.asarray(parallel[name])
            errors[name] = float(np.max(abs(a - b)) / max(np.max(abs(a)), 1e-30))
        for name in NAMES:
            for key in ["amplitude", "imaginary", "flux", "angle"]:
                errors[name + "_" + key] = abs(
                    serial["branches"][name][key] - parallel["branches"][name][key]
                )
        assert max(errors.values()) < 3e-11, errors
        assert sum(0 in ids for ids in parallel["rank_layers"]) >= 2
        assert sum(1 in ids for ids in parallel["rank_layers"]) >= 2
        record(
            f"scattering-mpi-{ranks}",
            dict(errors=errors, rank_layers=parallel["rank_layers"], h=600, degree=4),
        )
