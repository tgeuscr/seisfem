import numpy as np
import pytest

from .diagnostics import spectrum
from .packets import EXTENT
from .reference import LOWER, UPPER


@pytest.mark.parametrize("mode", ["P", "S"])
def test_continuum_signed_estimator_and_spectral_bandwidth(continuum, mode):
    result = continuum[mode]
    for b in result["branches"].values():
        assert b["amplitude_error"] < 4e-4
        assert abs(b["imaginary"]) < 3e-4
    assert result["closure_error"] < 6e-4
    assert result["bandwidth"]["excluded_energy"] < 3e-7
    assert abs(sum(result["bandwidth"]["flux"].values()) - 1) < 1e-9
    assert result["bandwidth"]["spread"] < (2.5 if mode == "P" else 1.5)


@pytest.mark.parametrize("angle,kind,side", [(9, "P", 1), (38, "S", -1)])
def test_angle_estimator_blind_to_snell(angle, kind, side):
    h = 40
    axis = np.linspace(-EXTENT, EXTENT, round(2 * EXTENT / h) + 1)
    x, z = np.meshgrid(axis, axis)
    theta = np.deg2rad(angle)
    n = np.array([np.sin(theta), side * np.cos(theta)])
    d = n if kind == "P" else np.array([n[1], -n[0]])
    z = z - side * 3600
    envelope = np.exp(-(x * x + z * z) / (2 * 1500**2))
    k = 2 * np.pi * 5 / (LOWER if side < 0 else UPPER).speed(kind)
    u = (envelope * np.cos(k * (x * n[0] + z * n[1])))[:, :, None] * d
    measured, spread = spectrum(u, h, side, kind)
    assert abs(measured - angle) < 0.25
    assert 0.5 < spread < 4


@pytest.mark.parametrize("mode", ["P", "S"])
def test_continuum_physical_energy_independently_checks_jacobian(mode):
    from .continuum import field
    from .diagnostics import physical_energy
    from .packets import ANGLES, CENTER, FREQUENCY, SIGMA_Q, TIMES, initial, parameters

    h = 30
    axis = np.linspace(-EXTENT, EXTENT, round(2 * EXTENT / h) + 1)
    x, z = np.meshgrid(axis, axis)
    u, v = initial(np.column_stack((x.ravel(), z.ravel())), mode)
    grid0 = np.column_stack((u, v)).reshape(len(axis), len(axis), 4)
    grid = field(
        mode,
        np.deg2rad(ANGLES[mode]),
        FREQUENCY,
        parameters(mode)[4],
        SIGMA_Q,
        CENTER,
        TIMES[mode],
        EXTENT,
        h,
    )
    ratio = physical_energy(grid, h) / physical_energy(grid0, h)
    assert abs(ratio - 1) < 2e-6
