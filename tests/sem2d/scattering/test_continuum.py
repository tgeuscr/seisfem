"""Audit the estimator on continuum fields before accepting SEM coefficients."""

import numpy as np
import pytest

from tests.sem2d.helpers import record

from .diagnostics import measure
from .packets import Packet
from .reference import NAMES, report


@pytest.mark.parametrize("mode", ["P", "S"])
def test_continuum_estimator_and_bandwidth(mode):
    packet = Packet(mode)
    templates = packet.templates()
    row = measure(sum(templates.values()), packet, templates)
    row["bandwidth"] = packet.bandwidth()
    assert max(b["complex_error"] for b in row["branches"].values()) < 2e-8
    assert max(b["central_error"] for b in row["branches"].values()) < 3e-4
    assert row["closure_error"] < 1e-8
    # Finite-beam spectral centroids are not literally central-ray angles.
    assert max(b["angle_error"] for b in row["branches"].values()) < 1.1
    narrow = Packet(mode, width=750).bandwidth()
    broad = packet.bandwidth()
    analytic = report(mode, packet.angle)["branches"]
    errors = []
    for bandwidth in [narrow, broad]:
        errors.append(
            float(
                np.linalg.norm(
                    [bandwidth["integrated_flux"][n] - analytic[n]["flux"] for n in NAMES]
                )
            )
        )
    assert errors[1] < errors[0] / 3
    row["width_study"] = dict(widths=[750, 1500], flux_errors=errors, narrow=narrow)
    record("scattering-continuum-" + mode, row)


@pytest.mark.parametrize("mode", ["P", "S"])
def test_estimator_recovers_independent_signed_branch_scalings(mode):
    packet = Packet(mode)
    templates = packet.templates()
    # A negative injected factor ensures the estimator cannot hide a sign flip.
    factors = [-0.7, 1.2, 0.9, 1.1]
    field = sum(f * templates[n] for f, n in zip(factors, NAMES, strict=True))
    result = measure(field, packet, templates)
    analytic = report(mode, packet.angle)["branches"]
    errors = [
        abs(result["branches"][n]["amplitude"] - f * analytic[n]["amplitude"])
        for n, f in zip(NAMES, factors, strict=True)
    ]
    assert max(errors) < 2e-8


@pytest.mark.parametrize("kind,theta,side", [("P", 9, 1), ("S", 38, -1)])
def test_angle_estimator_without_snell_hint(kind, theta, side):
    from .diagnostics import angles

    packet = Packet("P")
    axis = np.arange(-packet.analysis, packet.analysis + packet.sampling / 2, packet.sampling)
    x, z = np.meshgrid(axis, axis)
    n = np.array([np.sin(np.deg2rad(theta)), side * np.cos(np.deg2rad(theta))])
    d = n if kind == "P" else n[[1, 0]] * [1, -1]
    z = z - side * 3600
    speed = (packet.lower if side < 0 else packet.upper).speed(kind)
    scalar = np.exp(-(x * x + z * z) / (2 * 1500**2)) * np.cos(
        2 * np.pi * packet.frequency / speed * (n[0] * x + n[1] * z)
    )
    field = np.zeros((*x.shape, 4))
    field[:, :, :2] = scalar[:, :, None] * d
    angle, spread = angles(field, packet, side, kind)
    assert abs(angle - theta) < 0.25
    assert 0.5 < spread < 4
    if side < 0:
        # A stronger upward wave has a downward negative-frequency conjugate.
        # The diagnostic must recover the weak outgoing lobe, not that image.
        envelope = np.exp(-(x * x + z * z) / (2 * 1500**2))
        omega = 2 * np.pi * packet.frequency
        field[:, :, :2] *= 0.2
        phase = omega / speed * (n[0] * x + n[1] * z)
        field[:, :, 2:] = (0.2 * omega * envelope * np.sin(phase))[:, :, None] * d
        incoming = np.array([np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15))])
        polarization = incoming[[1, 0]] * [1, -1]
        phase = omega / speed * (incoming[0] * x + incoming[1] * z)
        field[:, :, :2] += (3 * envelope * np.cos(phase))[:, :, None] * polarization
        field[:, :, 2:] += (3 * omega * envelope * np.sin(phase))[:, :, None] * polarization
        measured, _ = angles(field, packet, side, kind)
        assert abs(measured - theta) < 0.25


def test_frequency_separation_retains_true_polarization_error():
    from .diagnostics import frequency_isolate

    d = np.array([0.6, 0.8])
    perpendicular = np.array([0.8, -0.6])
    target = (0.8 + 0.3j) * (d + 0.01 * perpendicular)
    contaminant = (3 - 4j) * perpendicular
    omega, other = 31.0, 17.0
    u = target + contaminant
    v = -1j * (omega * target + other * contaminant)
    measured = frequency_isolate(u, v, omega, other)
    np.testing.assert_allclose(measured, target, rtol=0, atol=3e-15)
    assert abs(abs(measured @ perpendicular) / abs(measured @ d) - 0.01) < 3e-15
