import numpy as np
import pytest

from .absorbing_packets import compare, run
from .absorbing_source import study
from .helpers import record


@pytest.mark.parametrize("mode,limit,field_limit", [("P", 8e-5, 8e-5), ("S", 2.5e-4, 5e-4)])
def test_normal_free_absorbing_and_large_box(mode, limit, field_limit):
    rows = [compare(mode, h) for h in [100, 50]]
    for row in rows:
        assert row["incident"] > 0.95
        assert 0.95 < row["free_reflection"] < 1.05
        assert row["incident_control_difference"] < 1e-8
        record(f"absorbing-normal-{mode}-h{row['h']}", row)
    assert rows[1]["residual_reflection"] < limit
    assert rows[1]["interior_field_ratio"] < field_limit
    assert rows[1]["reflected_waveform_ratio"] < 0.001
    assert rows[1]["residual_reflection"] < rows[0]["residual_reflection"] / 4
    assert rows[1]["interior_field_ratio"] < rows[0]["interior_field_ratio"] / 4
    assert abs(rows[1]["free_arrival_error"]) <= rows[1]["provenance"]["free"]["dt"]


@pytest.mark.parametrize("mode", ["P", "S"])
def test_oblique_large_box_characterization(mode):
    row = compare(mode, angle=25, beam=True)
    # Characterization, not a zero-reflection assertion. Finite beams include
    # angular spread; local impedance reflects strongly for oblique SV here.
    assert 0.005 < row["reflection_ratio"] < 0.6
    assert row["interior_field_ratio"] < 0.3
    record(f"absorbing-oblique-{mode}", row)


@pytest.mark.parametrize("mode,layer", [("P", 0), ("S", 1)])
def test_layer_local_packet_matches_homogeneous_material(mode, layer):
    a = run(mode, beam=True, layered=True, layer=layer)
    b = run(mode, beam=True, layered=False, layer=layer)
    error = float(np.linalg.norm(a["u"] - b["u"]) / np.linalg.norm(b["u"]))
    record(
        f"absorbing-local-{mode}-{layer}",
        dict(
            history_relative_error=error,
            mode=mode,
            layer=layer,
            provenance={
                k: v for k, v in a.items() if k not in ["time", "u", "field", "coordinates"]
            },
        ),
    )
    assert error < 1e-7


@pytest.mark.parametrize("mode", ["P", "S"])
def test_triangular_large_box_control(mode):
    row = compare(mode, 20, p=0)
    assert row["reflection_ratio"] < 0.04
    assert row["interior_field_ratio"] < 0.04
    record(f"absorbing-triangle-{mode}", row)


def test_public_point_force_free_top():
    row = study()
    assert row["late_error_ratio"] < 0.15
    assert row["early_error"] < 0.001
    assert min(row["direct_component_peaks"].values()) > 1e-4
    assert row["config"]["boundaries"]["upper"] == "free"
    record("absorbing-source", row)
