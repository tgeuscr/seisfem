import numpy as np
import pytest

from .experiment import run
from .helpers import record
from .packets import TIME, WIDTH_Q, WIDTH_S, Packet


@pytest.mark.parametrize("branch", ["P", "S"])
def test_continuum_potential_and_spectral_resolution(branch):
    packet = Packet(branch, 25)
    axis = np.linspace(-1500, 1500, 39)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    positions = np.stack((x, z), axis=-1) - packet.center
    s, q = positions @ packet.n, positions @ packet.transverse
    radius = np.linalg.norm(packet.k0)
    g = np.exp(-0.5 * ((s / WIDTH_S) ** 2 + (q / WIDTH_Q) ** 2))
    grad = -s[..., None] * packet.n / WIDTH_S**2 - q[..., None] * packet.transverse / WIDTH_Q**2
    exact = g[..., None] * (
        packet.n * np.cos(radius * s)[..., None] + grad * np.sin(radius * s)[..., None] / radius
    )
    if branch == "S":
        exact = exact[..., [1, 0]] * np.array([1, -1])
    np.testing.assert_allclose(packet.grid(axis, axis), exact, atol=1e-12)
    other = Packet(branch, 25, size=160)
    error = float(np.max(abs(packet.grid(axis, axis, TIME) - other.grid(axis, axis, TIME))))
    assert error < 1e-12
    record(
        f"continuum-{branch}",
        dict(
            initial_potential_error=float(np.max(abs(packet.grid(axis, axis) - exact))),
            resolution_error=error,
        ),
    )


@pytest.mark.parametrize("p", [2, 4, 6])
@pytest.mark.parametrize("branch", ["P", "S"])
def test_spatial_refinement_and_resolved_wave(benchmark, p, branch):
    rows = [benchmark[p, n, branch, 0.02] for n in [48, 96, 144]]
    for key in ["field_error", "waveform_error", "speed_error"]:
        errors = [r[key] for r in rows]
        assert errors[2] < errors[1] < errors[0], (p, branch, key, errors)
        assert errors[2] < 0.1 * errors[0], (p, branch, key, errors)
    fine = rows[-1]
    assert fine["speed_error"] < 0.001
    assert fine["polarization_error"] < 0.001
    assert fine["arrival_error"] < 0.005
    assert fine["angle_error"] < 0.01
    assert max(r["energy_drift"] for r in rows) < 1e-11


@pytest.mark.parametrize("p", [2, 4, 6])
@pytest.mark.parametrize("branch", ["P", "S"])
@pytest.mark.parametrize("angle", [0, 45, -20])
def test_axial_diagonal_and_signed_oblique(p, branch, angle):
    # Quadratic SV needs more points to resolve its shorter wavelength to 1.5%.
    # Equal-DOF comparisons remain separate and use identical grids for all orders.
    row = run(p, 192 if p == 2 else 144, branch, angle, fractions=(0.02,))[0]
    record(f"angle-p{p}-{branch}-{angle}", row)
    assert row["speed_error"] < 0.0015
    assert row["polarization_error"] < 0.0015
    assert row["field_error"] < 0.015
    assert row["arrival_error"] < 0.008
    assert row["angle_error"] < 0.02
    assert row["energy_drift"] < 1e-11


@pytest.mark.parametrize("branch", ["P", "S"])
def test_boundary_isolation(benchmark, branch):
    small = benchmark[6, 144, branch, 0.02]
    # Same h=200 m elements, packet and common diagnostic grid; boundaries +600 m.
    large = run(6, 180, branch, fractions=(0.02,), extent=3000)[0]
    differences = {
        k: abs(large[k] - small[k])
        for k in ["field_error", "speed_error", "waveform_error", "angle_error"]
    }
    record(f"boundary-{branch}", differences)
    assert max(differences.values()) < 2e-5
