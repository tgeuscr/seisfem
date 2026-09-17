import numpy as np
import pytest

from .evidence import record
from .packets import comparison

NORMAL = [(mode, layer, "right", 0) for mode in ["P", "S"] for layer in [0, 1]] + [
    (mode, 0, "lower", 0) for mode in ["P", "S"]
]
OBLIQUE = [("P", 1, "right", angle) for angle in [15, 30, 45]]


@pytest.fixture(scope="module")
def packets():
    result = {}
    for mode, layer, side, angle in NORMAL + OBLIQUE:
        row = comparison(mode, layer, side, angle)
        result[(mode, layer, side, angle)] = row
        record(f"packet-{mode}-{layer}-{side}-{angle}", row)
    return result


@pytest.mark.parametrize("case", NORMAL)
def test_local_normal_absorption_in_each_layer_and_both_orientations(packets, case):
    row = packets[case]
    assert row["results"]["layered"]["free_peak"] > 0.4
    assert row["results"]["layered"]["reflection_proxy"] < 0.08
    assert row["proxy_difference"] < 1e-8
    assert max(row["representation_difference"].values()) < 1e-8


@pytest.mark.parametrize("angle", [15, 30, 45])
def test_oblique_consistency_with_local_homogeneous_absorber(packets, angle):
    row = packets[("P", 1, "right", angle)]
    assert row["results"]["layered"]["free_peak"] > 0.4
    assert row["proxy_difference"] < 1e-8
    assert max(row["representation_difference"].values()) < 1e-8


def test_oblique_reflection_increases_without_an_exactness_claim(packets):
    proxies = [
        packets[("P", 1, "right", a)]["results"]["layered"]["reflection_proxy"]
        for a in [0, 15, 30, 45]
    ]
    assert np.all(np.diff(proxies) > 0)
    assert proxies[-1] > 0.05
    assert proxies[-1] > 3 * proxies[0]
