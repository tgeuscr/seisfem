import json
import os
from pathlib import Path

import pytest
from mpi4py import MPI

from .continuum import bandwidth, field
from .diagnostics import measure
from .experiment import run
from .packets import ANGLES


def record(name, result):
    directory = os.environ.get("SEISFEM_VTI_INTERFACE_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")


@pytest.fixture(scope="session")
def continuum():
    reports = {}
    for angle in ANGLES:
        report = measure(field(angle, 30), angle, 30)
        report["bandwidth"] = bandwidth(angle)
        reports[angle] = report
        record(f"continuum-{angle:g}", report)
    return reports


@pytest.fixture(scope="session")
def refinement():
    reports = {}
    for angle in ANGLES:
        for h in [40, 30, 20] if abs(angle) == 25 else [20]:
            report, grid = run(angle, h, MPI.COMM_SELF)
            report.update(measure(grid, angle, h))
            reports[(angle, h)] = report
            record(f"fem-{angle:g}-{h}", report)
    return reports
