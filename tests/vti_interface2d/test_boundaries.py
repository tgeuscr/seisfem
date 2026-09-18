import pytest
from mpi4py import MPI

from .conftest import record
from .diagnostics import measure
from .experiment import run
from .reference import NAMES


@pytest.mark.parametrize("angle", [25.0, 35.0])
def test_outer_boundary_isolation(refinement, angle):
    # Move all free boundaries out by 1200 m, retaining physical initialization,
    # h and time. Crop for an identical Fourier integration window.
    _, large = run(angle, 20, MPI.COMM_SELF, extent=9600)
    result = measure(large[60:-60, 60:-60], angle, 20)
    original = refinement[(angle, 20)]
    differences = {
        name: {
            key: abs(result["branches"][name][key] - original["branches"][name][key])
            for key in ["real", "imaginary", "flux", "phase_angle"]
        }
        for name in NAMES
    }
    for b in differences.values():
        assert max(b[k] for k in ["real", "imaginary", "flux"]) < 1e-3
        assert b["phase_angle"] < 0.03
    record(f"boundary-{angle:g}", differences)
