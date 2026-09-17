import json
import os
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from .continuum import bandwidth, field
from .diagnostics import measure
from .experiment import run
from .packets import ANGLES, CENTER, EXTENT, FREQUENCY, SIGMA_Q, TIMES, parameters


def record(name, result):
    """Optional audit output; never read cached evidence in place of a simulation."""
    directory = os.environ.get("SEISFEM_OBLIQUE_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")


@pytest.fixture(scope="session")
def continuum():
    result = {}
    for mode in ["P", "S"]:
        args = (mode, np.deg2rad(ANGLES[mode]), FREQUENCY, parameters(mode)[4], SIGMA_Q, CENTER)
        grid = field(*args, TIMES[mode], EXTENT, 20)
        result[mode] = measure(grid, mode, 20)
        result[mode]["bandwidth"] = bandwidth(*args)
        record("continuum-" + mode, result[mode])
    return result


@pytest.fixture(scope="session")
def refinement():
    result = {}
    for mode in ["P", "S"]:
        result[mode] = []
        for h in [40, 30, 20]:
            report, grid = run(mode, h, MPI.COMM_SELF)
            report.update(measure(grid, mode, h))
            result[mode].append(report)
            record(f"{mode}-{h}", report)
    return result
