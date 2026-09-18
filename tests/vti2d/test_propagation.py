import pytest

from .helpers import record
from .packets import run


@pytest.fixture(scope="module")
def packets():
    rows = {}
    for branch in ["P", "S"]:
        for indices in [(0, 16), (16, 0), (8, 12), (-8, 12)]:
            for h in [40, 30, 20] if indices[1] == 12 else [20]:
                key = (branch, indices, h)
                rows[key] = run(*key)
                record(f"packet-{branch}-{indices[0]}-{indices[1]}-{h}", rows[key])
    return rows


@pytest.mark.parametrize("branch", ["P", "S"])
@pytest.mark.parametrize("indices", [(0, 16), (16, 0), (8, 12), (-8, 12)])
def test_phase_and_polarization(packets, branch, indices):
    row = packets[(branch, indices, 20)]
    assert row["phase_error"] < 0.03
    assert row["polarization_leakage"] < 0.04
    assert abs(row["phase_angle"] - row["reference_angle"]) < 1e-10
    assert row["group_error"] < 0.06


@pytest.mark.parametrize("branch", ["P", "S"])
@pytest.mark.parametrize("indices", [(8, 12), (-8, 12)])
def test_refinement(packets, branch, indices):
    for key in ["phase_error", "field_error", "group_error", "polarization_leakage"]:
        values = [packets[(branch, indices, h)][key] for h in [40, 30, 20]]
        assert values[2] < values[1] < values[0], (key, values)
        assert values[2] < 0.6 * values[0], (key, values)


@pytest.mark.parametrize("branch", ["P", "S"])
def test_boundary_isolation(packets, branch):
    large = run(branch, (8, 12), 20, extent=3600)
    small = packets[(branch, (8, 12), 20)]
    errors = {
        key: abs(large[key] - small[key])
        for key in ["phase_error", "polarization_leakage", "group_error", "field_error"]
    }
    assert max(errors.values()) < 2e-5
    record("boundary-" + branch, errors)
