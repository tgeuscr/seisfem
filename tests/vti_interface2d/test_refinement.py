import numpy as np
import pytest

from .packets import ANGLES, EXTENT
from .reference import LOWER, NAMES, UPPER, solve


@pytest.mark.parametrize("angle", [25.0, -25.0])
def test_complex_coefficients_flux_and_angles_refine(refinement, continuum, angle):
    rows = [refinement[(angle, h)] for h in [40, 30, 20]]
    for key in ["complex_error", "signed_error", "flux_error"]:
        errors = [np.linalg.norm([r["branches"][n][key] for n in NAMES]) for r in rows]
        assert errors[2] < errors[1] < errors[0], (angle, key, errors)
        assert errors[2] < 0.4 * errors[0], (angle, key, errors)
    for name in NAMES:
        errors = [r["branches"][name]["complex_error"] for r in rows]
        assert errors[2] < errors[1] < errors[0], (angle, name, errors)
    closure = [r["closure_error"] for r in rows]
    assert closure[2] < closure[1] < closure[0], closure
    phase_geometry = [
        np.linalg.norm(
            [
                r["branches"][n]["phase_angle"] - continuum[angle]["branches"][n]["phase_angle"]
                for n in NAMES
            ]
        )
        for r in rows
    ]
    assert phase_geometry[2] < phase_geometry[1] < phase_geometry[0], phase_geometry


@pytest.mark.parametrize("angle", ANGLES)
def test_resolved_signed_scattering(refinement, continuum, angle):
    row = refinement[(angle, 20)]
    _, ref = solve(np.sin(np.deg2rad(angle)) / 3000)
    assert row["closure_error"] < 0.012
    for name, b in row["branches"].items():
        assert b["signed_error"] < 0.018
        assert b["complex_error"] < 0.15
        assert b["flux_error"] < 0.012
        if abs(ref[name]["amplitude"]) > 1e-8:
            assert np.sign(b["real"]) == np.sign(ref[name]["amplitude"].real)
            assert abs(b["phase_angle"] - continuum[angle]["branches"][name]["phase_angle"]) < 0.2
            assert b["angle_error"] < 1.0
        else:
            # A finite beam has oblique sidebands; symmetry forbids conversion
            # of its central p=0 component, not all finite-bandwidth conversion.
            assert b["magnitude"] < 3e-4
            assert b["flux"] < 1e-7


def test_plus_minus_symmetry_improves(refinement):
    errors = []
    for h in [40, 30, 20]:
        terms = []
        for name in NAMES:
            a = refinement[(25, h)]["branches"][name]
            b = refinement[(-25, h)]["branches"][name]
            sign = 1 if name[1] == "P" else -1
            terms.append(
                complex(a["real"], a["imaginary"]) - sign * complex(b["real"], b["imaginary"])
            )
        errors.append(np.linalg.norm(terms))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.4 * errors[0]


def test_material_mass_and_purity(refinement):
    for r in refinement.values():
        np.testing.assert_allclose(r["mass"], (LOWER.rho + UPPER.rho) * 2 * EXTENT**2, rtol=3e-14)
        assert r["material_error"] < 3e-15
        assert r["dt"] <= 0.8 * r["stable_dt"]
    for angle in [25, -25]:
        purity = [refinement[(angle, h)]["purity"] for h in [40, 30, 20]]
        assert purity[2] < purity[1] < purity[0]
