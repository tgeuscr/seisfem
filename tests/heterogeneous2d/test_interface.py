import numpy as np
import pytest

from tests.heterogeneous2d.interface import analytical, study


def test_normal_p_interface_refinement_1d_bridge_and_zero_contrast():
    reference = analytical()
    assert reference["R"] == pytest.approx(-2 / 7, abs=1e-16)
    assert reference["T"] == pytest.approx(5 / 7, abs=1e-16)
    assert reference["R_energy"] + reference["T_energy"] == pytest.approx(1, abs=2e-16)
    report = study()
    rows = report["refinement"]
    for name in ["R_error", "T_error", "trace_difference", "transverse_peak"]:
        values = [row[name] for row in rows]
        assert values[2] < values[1] < values[0]
    # Absolute coefficient errors after measuring the resolved h=5 m case.
    assert rows[-1]["R_error"] < 3e-4
    assert rows[-1]["T_error"] < 7e-4
    assert rows[-1]["trace_difference"] < 5e-6
    for row in rows:
        for coefficient in ["R", "T"]:
            assert (
                abs(row["two_dimensional"][coefficient] - row["one_dimensional"][coefficient])
                < 1e-5
            )
        assert row["two_dimensional"]["R"] < 0
        assert row["two_dimensional"]["T"] > 0
    zero = report["zero_contrast"]
    assert abs(zero["R"]) < 1e-11
    assert zero["layered_vs_homogeneous"] < 1e-12
    assert zero["spurious_interface_peak"] < 1e-12
    np.testing.assert_allclose([analytical(False)["R"], analytical(False)["T"]], [0, 1], atol=1e-16)
