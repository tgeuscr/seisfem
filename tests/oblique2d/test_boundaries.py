import pytest
from mpi4py import MPI

from .conftest import record
from .diagnostics import measure
from .experiment import run
from .reference import NAMES


@pytest.mark.parametrize("mode", ["P", "S"])
def test_enlarged_domain_has_negligible_effect(refinement, mode):
    # Keep physical initialization, h, time, and diagnostic window fixed.
    # Move every free boundary 1200 m farther away, then crop to the original box.
    _, grid = run(mode, 40, MPI.COMM_SELF, extent=8400)
    larger = measure(grid[30:-30, 30:-30], mode, 40)
    original = refinement[mode][0]
    differences = {}
    for name in NAMES:
        a, b = original["branches"][name], larger["branches"][name]
        differences[name] = {
            key: abs(a[key] - b[key]) for key in ["angle", "amplitude", "imaginary", "flux"]
        }
        assert differences[name]["angle"] < 0.02
        assert differences[name]["amplitude"] < 5e-4
        assert differences[name]["imaginary"] < 5e-4
        assert differences[name]["flux"] < 5e-4
    record("boundary-" + mode, differences)
