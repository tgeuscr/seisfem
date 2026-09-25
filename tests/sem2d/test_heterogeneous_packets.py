import numpy as np
import pytest

from .helpers import record
from .heterogeneous_packet import run


@pytest.mark.parametrize("case", ["increase", "decrease"])
def test_normal_interface_refinement(case):
    rows = [run(4, h, case) for h in [100, 50]]
    for key in ["R_error", "T_error", "flux_error", "flux_closure"]:
        assert rows[1][key] < rows[0][key]
    assert rows[1]["R_error"] < 1e-6
    assert rows[1]["T_error"] < 1e-5
    assert rows[1]["flux_closure"] < 2e-5
    assert max(abs(e) for e in rows[1]["arrival_errors"]) < 3e-5
    for row in rows:
        assert row["transverse_peak"] < 1e-11
        assert row["energy_drift"] < 3e-13
        ref = row["analytical"]
        assert abs(ref["R_flux"] + ref["T_flux"] - 1) < 1e-15
        record(f"heterogeneous-packet-{case}-h{row['h']}", row)


def test_identical_material_no_reflection():
    a = run(6, 50, "identical", traces=True)
    b = run(6, 50, "homogeneous", traces=True)
    errors = {}
    for key in ["u", "v"]:
        x, y = np.array(a.pop(key)), np.array(b.pop(key))
        errors[key] = float(np.linalg.norm(x - y) / np.linalg.norm(y))
        assert errors[key] < 3e-12
    assert abs(a["R"]) < 1e-10
    assert abs(a["T"] - 1) < 2e-6
    assert a["transverse_peak"] < 1e-11
    a["homogeneous_trace_errors"] = errors
    record("heterogeneous-packet-identical", a)


def test_triangular_interface_converges_to_same_analytical_solution():
    rows = [run(0, h) for h in [10, 5]]
    for key in ["R_error", "T_error", "flux_error", "flux_closure"]:
        assert rows[1][key] < rows[0][key]
    assert rows[1]["R_error"] < 3e-4
    assert rows[1]["T_error"] < 7e-4
    assert max(abs(e) for e in rows[1]["arrival_errors"]) < 0.002
    assert max(rows[1]["waveform_errors"]) < 0.03
    for row in rows:
        record(f"heterogeneous-triangle-h{row['h']}", row)
