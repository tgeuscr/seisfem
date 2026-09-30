"""Quantitative gates selected after the fixed-geometry 300/200 m pilot."""

import numpy as np
import pytest

from tests.sem2d.helpers import record

from .conftest import save
from .experiment import run
from .packets import Packet
from .reference import NAMES


@pytest.mark.parametrize("mode", ["P", "S"])
def test_signed_converted_branches_refine(scattering_refinement, mode):
    rows = scattering_refinement[mode]
    coarse, resolved = rows[:2]
    # The 300 m SV grid is intentionally retained as under-resolution evidence.
    # Acceptance is on the resolved grid, not a forced fit of its spurious lobe.
    for metric in ["complex_error", "signed_error", "flux_error", "spectral_waveform_error"]:
        errors = [np.linalg.norm([r["branches"][n][metric] for n in NAMES]) for r in rows]
        assert np.all(np.diff(errors) < 0), (mode, metric, errors)
        assert errors[1] < 0.15 * errors[0], (mode, metric, errors)
    assert resolved["closure_error"] < (4e-4 if mode == "P" else 3.5e-3)
    assert resolved["closure_error"] < coarse["closure_error"] / 20
    for candidate in rows[1:]:
        assert candidate["closure_error"] < (4e-4 if mode == "P" else 3.5e-3)
        for name in NAMES:
            b = candidate["branches"][name]
            truth = candidate["geometry"]["reference"]["branches"][name]
            assert np.sign(b["amplitude"]) == np.sign(truth["amplitude"])
            assert b["complex_error"] < (1.3e-3 if mode == "P" else 5e-3)
            assert b["signed_error"] < (8e-4 if mode == "P" else 4.8e-3)
            assert b["angle_discretization_error"] < (0.003 if mode == "P" else 0.015)
            assert b["angle_error"] < 1.1
            assert b["polarization_leakage"] < (0.0017 if mode == "P" else 0.0035)
            assert b["spectral_waveform_error"] < (0.022 if mode == "P" else 0.055)
            assert b["arrival_resolved"]
            assert b["arrival_error"] < 0.0055
    for row in rows:
        assert row["energy_drift"] < 5e-13
        assert row["geometry"]["return_margin"] > 0.1
        assert row["dt_safe_ratio"] < 0.2
        assert row["sum_factorization"]
        assert row["geometry"]["reference"]["closure"] < 2e-15
        assert row["bandwidth"]["excluded_energy"] < 3e-7
        assert max(row["template_overlap"].values()) < 1e-6


@pytest.mark.parametrize("mode", ["P", "S"])
def test_reverse_materials(mode):
    row, grid = run(Packet(mode, reverse=True), h=200 if mode == "P" else 150)
    save("reverse-" + mode, row, grid)
    for name, b in row["branches"].items():
        truth = row["geometry"]["reference"]["branches"][name]
        assert np.sign(b["amplitude"]) == np.sign(truth["amplitude"])
        assert b["complex_error"] < 0.005
        assert b["angle_discretization_error"] < 0.025
        assert b["polarization_leakage"] < 0.005
    assert row["closure_error"] < 0.004
    assert row["energy_drift"] < 5e-13


def test_larger_free_domain(scattering_refinement):
    small = scattering_refinement["S"][0]
    large, grid = run(Packet("S"), h=300, extent=15600)
    save("larger-domain", large, grid)
    differences = {}
    for name in NAMES:
        for key in ["amplitude", "imaginary", "angle", "flux"]:
            differences[name + "_" + key] = abs(
                small["branches"][name][key] - large["branches"][name][key]
            )
    differences["history_relative"] = float(
        np.linalg.norm(np.array(small["trace"]) - large["trace"]) / np.linalg.norm(small["trace"])
    )
    assert max(differences.values()) < 1e-7, differences
    record(
        "scattering-boundary-control",
        dict(differences=differences, extent=[14400, 15600], h=300, mode="S"),
    )
