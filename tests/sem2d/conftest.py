import pytest

from .experiment import run
from .helpers import record


@pytest.fixture(scope="session")
def benchmark():
    rows = {}
    for p in [0, 2, 4, 6]:
        for n in [48, 96, 144]:
            for branch in ["P", "S"]:
                fractions = (0.02, 0.1, 0.2, 0.4, 0.6, 0.8) if n == 96 else (0.02,)
                for row in run(p, n, branch, fractions=fractions):
                    f = row["requested_fraction"]
                    rows[p, n, branch, f] = row
                    record(f"benchmark-p{p}-n{n}-{branch}-f{f}", row)
    return rows
