import numpy as np
import pytest

from .conftest import record
from .continuum import energy, field
from .packets import ANGLES, EXTENT, SPEED, initial
from .reference import LOWER, NAMES, UPPER, solve, wave


@pytest.mark.parametrize("angle", ANGLES)
def test_continuum_complex_recovery_and_flux(continuum, angle):
    r = continuum[angle]
    assert max(b["complex_error"] for b in r["branches"].values()) < 2.5e-4
    assert r["closure_error"] < 2e-4
    assert r["bandwidth"]["excluded_energy"] < 2e-6
    assert abs(sum(r["bandwidth"]["integrated_flux"].values()) - 1) < 1e-10
    assert r["bandwidth"]["spread"] < 2.4


@pytest.mark.parametrize("angle", [0, 25, 35])
def test_jacobian_by_independent_slowness_differentiation(angle):
    omega = 2 * np.pi * 5
    p = np.sin(np.deg2rad(angle)) / SPEED
    inc, branches = solve(p)
    kx, kz = omega * p, omega * inc["q"]
    for name, w in branches.items():
        m = LOWER if name[0] == "R" else UPPER
        side = -1 if name[0] == "R" else 1

        def mapped(z, m=m, name=name, side=side):
            freq = SPEED * np.hypot(kx, z)
            return freq * wave(kx / freq, m, name[1], side)["q"]

        eps = 1e-8
        numerical = abs((mapped(kz + eps) - mapped(kz - eps)) / (2 * eps))
        analytic = abs(inc["g"][1] / w["g"][1])
        assert abs(numerical / analytic - 1) < 3e-9


@pytest.mark.parametrize("angle", [0, 25, 35])
def test_continuum_physical_energy_audits_jacobian(angle):
    h = 30
    axis = np.linspace(-EXTENT, EXTENT, round(2 * EXTENT / h) + 1)
    x, z = np.meshgrid(axis, axis)
    u, v = initial(np.column_stack((x.ravel(), z.ravel())), angle)
    original = np.column_stack((u, v)).reshape(len(axis), len(axis), 4)
    ratio = energy(field(angle, h), h) / energy(original, h)
    record(f"packet-energy-{angle}", dict(ratio=ratio, error=abs(ratio - 1)))
    assert abs(ratio - 1) < 3e-5


def test_continuum_plus_minus_symmetry(continuum):
    a, b = continuum[25], continuum[-25]
    for name in NAMES:
        sign = 1 if name[1] == "P" else -1
        for key in ["real", "imaginary"]:
            assert abs(a["branches"][name][key] - sign * b["branches"][name][key]) < 3e-13
        assert abs(a["branches"][name]["flux"] - b["branches"][name]["flux"]) < 3e-13
