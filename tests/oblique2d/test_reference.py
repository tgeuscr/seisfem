import numpy as np
import pytest

from .reference import LOWER, UPPER, residual, solve


@pytest.mark.parametrize(
    "mode,angle", [("P", 25), ("S", 15), ("P", 0), ("S", 0), ("P", -25), ("S", -15)]
)
def test_welded_residual_and_flux(mode, angle):
    ref = solve(mode, angle)
    assert abs(sum(b["flux"] for b in ref.values()) - 1) < 2e-15
    assert np.max(abs(residual(mode, angle, ref))) < 6e-16
    p = np.sin(np.deg2rad(angle)) / LOWER.speed(mode)
    for name, b in ref.items():
        mat = LOWER if name[0] == "R" else UPPER
        assert abs(b["direction"][0] / mat.speed(name[1]) - p) < 1e-19


@pytest.mark.parametrize("mode", ["P", "S"])
def test_zero_contrast_and_normal_impedance(mode):
    zero = solve(mode, 20, upper=LOWER)
    for name, b in zero.items():
        assert abs(b["amplitude"] - (name == "T" + mode)) < 5e-16
    ref = solve(mode, 0)
    z1, z2 = LOWER.rho * LOWER.speed(mode), UPPER.rho * UPPER.speed(mode)
    # Reflected polarizations reverse at normal incidence for both P and S.
    assert abs(ref["R" + mode]["amplitude"] - (z2 - z1) / (z1 + z2)) < 3e-16
    assert abs(ref["T" + mode]["amplitude"] - 2 * z1 / (z1 + z2)) < 3e-16


def test_critical_rejected():
    with pytest.raises(ValueError, match="below-critical"):
        solve("S", 35)
