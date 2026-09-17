import numpy as np
import pytest

from .packets import ANGLES
from .reference import LOWER, NAMES, UPPER, solve


@pytest.mark.parametrize("mode", ["P", "S"])
def test_initial_purity_improves(refinement, mode):
    purity = [r["purity"] for r in refinement[mode]]
    assert np.all(np.diff(purity) < 0)
    assert purity[-1] < 0.07
    assert purity[-1] < 0.52 * purity[0]


@pytest.mark.parametrize("mode", ["P", "S"])
def test_snell_geometry_and_continuum_beam_convergence(refinement, continuum, mode):
    rows = refinement[mode]
    # Centroids can be displaced from the central ray by spectral weighting.
    # Compare to the independently propagated finite beam, not a fitted angle.
    errors = [
        np.linalg.norm(
            [
                r["branches"][name]["angle"] - continuum[mode]["branches"][name]["angle"]
                for name in NAMES
            ]
        )
        for r in rows
    ]
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.12
    for name in NAMES:
        b = rows[-1]["branches"][name]
        assert b["angle_error"] < 1.1
        assert 0.5 < b["spread"] < 4.5
        assert abs(b["spread"] - continuum[mode]["branches"][name]["spread"]) < 0.1


@pytest.mark.parametrize("mode", ["P", "S"])
def test_signed_zoeppritz_and_phase_converge(refinement, mode):
    rows = refinement[mode]
    reference = solve(mode, ANGLES[mode])
    errors = [
        np.linalg.norm([r["branches"][name]["amplitude_error"] for name in NAMES]) for r in rows
    ]
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.3 * errors[0]
    for name in NAMES:
        b = rows[-1]["branches"][name]
        assert np.sign(b["amplitude"]) == np.sign(reference[name]["amplitude"])
        assert abs(b["amplitude"] - reference[name]["amplitude"]) < 0.006
        assert abs(b["imaginary"]) < 0.10
        assert b["amplitude_error"] < rows[0]["branches"][name]["amplitude_error"]


@pytest.mark.parametrize("mode", ["P", "S"])
def test_normalized_flux_refines(refinement, continuum, mode):
    rows = refinement[mode]
    reference = solve(mode, ANGLES[mode])
    for name in NAMES:
        errors = [r["branches"][name]["flux_error"] for r in rows]
        assert np.all(np.diff(errors) < 0)
        assert errors[-1] < 0.014
        # Finite bandwidth changes integrated fractions by <0.0004 here.
        assert abs(continuum[mode]["bandwidth"]["flux"][name] - reference[name]["flux"]) < 4e-4
    closure = [r["closure_error"] for r in rows]
    assert np.all(np.diff(closure) < 0)
    assert closure[-1] < 0.016


@pytest.mark.parametrize("mode", ["P", "S"])
def test_material_mass_and_stability(refinement, mode):
    from .packets import EXTENT

    expected = (LOWER.rho + UPPER.rho) * 2 * EXTENT**2
    for r in refinement[mode]:
        np.testing.assert_allclose(r["mass"], expected, rtol=3e-14)
        assert r["dt"] <= 0.8 * r["stable_dt"]
    scaled = [r["stable_dt"] / r["h"] for r in refinement[mode]]
    np.testing.assert_allclose(scaled, scaled[0], rtol=3e-14)
