"""Run each physical refinement once per pytest session; never read cached evidence."""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.sem2d.helpers import record

from .experiment import run
from .packets import Packet


def save(name, row, grid):
    record("scattering-" + name, row)
    directory = os.environ.get("SEISFEM_SEM_REPORT_DIR")
    if directory:
        np.save(Path(directory) / ("scattering-" + name + ".npy"), grid)


@pytest.fixture(scope="session")
def scattering_refinement():
    rows = {}
    for mode in ["P", "S"]:
        rows[mode] = []
        for h in [300, 200] if mode == "P" else [300, 200, 150]:
            row, grid = run(Packet(mode), h=h)
            save(f"{mode}-{h}", row, grid)
            rows[mode].append(row)
    return rows
