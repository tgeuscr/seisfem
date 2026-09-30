"""Stress/traction reference audits, independent of any production assembly."""

import numpy as np
import pytest

from tests.oblique2d.reference import Material
from tests.oblique2d.reference import solve as old_solve

from .packets import Packet
from .reference import NAMES, A, B, report, solve, wave


@pytest.mark.parametrize("mode", ["P", "S"])
@pytest.mark.parametrize("angle", [0, 15, 25, -15])
@pytest.mark.parametrize("reverse", [False, True])
def test_welded_flux_snell_and_existing_independent_reference(mode, angle, reverse):
    lower, upper = (B, A) if reverse else (A, B)
    # 25 degree SV A->B is just below critical; its packet is not an acceptance case.
    r = report(mode, angle, lower, upper)
    old = old_solve(
        mode,
        angle,
        lower=Material(lower.rho, lower.vp, lower.vs),
        upper=Material(upper.rho, upper.vp, upper.vs),
    )
    assert r["residual"] < 2e-15
    assert r["closure"] < 2e-15
    for name in NAMES:
        b = r["branches"][name]
        m = lower if name[0] == "R" else upper
        np.testing.assert_allclose(b["amplitude"], old[name]["amplitude"], atol=8e-16)
        np.testing.assert_allclose(b["flux"], old[name]["flux"], atol=8e-16)
        assert abs(b["direction"][0] / m.speed(name[1]) - r["p"]) < 1e-19
        w = wave(m, name[1], r["p"], -1 if name[0] == "R" else 1)
        assert abs(w[3] / (m.rho * m.speed(name[1]) * w[0][1] / 2) - 1) < 8e-16


@pytest.mark.parametrize("mode", ["P", "S"])
def test_normal_and_identical(mode):
    r = report(mode, 0)
    z1, z2 = A.rho * A.speed(mode), B.rho * B.speed(mode)
    # dP and dSV both reverse their excited Cartesian component on reflection.
    assert abs(r["branches"]["R" + mode]["amplitude"] - (z2 - z1) / (z1 + z2)) < 5e-16
    assert abs(r["branches"]["T" + mode]["amplitude"] - 2 * z1 / (z1 + z2)) < 5e-16
    other = "S" if mode == "P" else "P"
    assert r["branches"]["R" + other]["amplitude"] == 0
    assert r["branches"]["T" + other]["amplitude"] == 0
    for angle in [15, 25, -20]:
        r = report(mode, angle, A, A)
        for name in NAMES:
            assert abs(r["branches"][name]["amplitude"] - (name == "T" + mode)) < 8e-16


def test_critical_rejected():
    with pytest.raises(ValueError, match="propagating"):
        solve("P", np.sin(np.deg2rad(55)) / A.vp)


@pytest.mark.parametrize("mode", ["P", "S"])
def test_geometry_bandwidth_and_jacobian(mode):
    packet = Packet(mode)
    g = packet.geometry(14400)
    assert g["return_margin"] > 0.1
    bw = packet.bandwidth()
    assert bw["excluded_energy"] < 1e-5
    assert abs(sum(bw["integrated_flux"].values()) - 1) < 2e-15
    p = g["reference"]["p"]
    kx = 2 * np.pi * packet.frequency * p
    kiz = packet.k0 * packet.n[1]
    for name, b in g["reference"]["branches"].items():
        c = (packet.lower if name[0] == "R" else packet.upper).speed(name[1])

        def output(k, c=c):
            return np.sqrt((packet.speed / c) ** 2 * (kx * kx + k * k) - kx * kx)

        eps = kiz * 1e-5
        numeric = (output(kiz + eps) - output(kiz - eps)) / (2 * eps)
        expected = packet.speed * packet.n[1] / (c * abs(b["direction"][1]))
        assert abs(numeric / expected - 1) < 1e-9


@pytest.mark.parametrize("mode", ["P", "S"])
def test_horizontal_reflection_symmetry(mode):
    positive = report(mode, 15)["branches"]
    negative = report(mode, -15)["branches"]
    for name in NAMES:
        sign = 1 if name[1] == mode else -1
        np.testing.assert_allclose(
            negative[name]["amplitude"], sign * positive[name]["amplitude"], atol=4e-16
        )
        np.testing.assert_allclose(negative[name]["flux"], positive[name]["flux"], atol=4e-16)
