"""End-to-end physical diagnostics, interpreted before boundary returns."""

import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D
from tests.experiments2d.helpers import (
    arrival_report,
    reciprocity_runs,
    relative_error,
    wave_config,
)


@pytest.fixture(scope="module")
def refinement():
    results = [Simulation2D(wave_config(n), MPI.COMM_SELF).run() for n in [320, 480, 640]]
    for n, result in zip([320, 480, 640], results, strict=True):
        expected, measured = arrival_report(result)
        print(
            "mesh",
            n,
            "predicted",
            expected,
            "measured",
            measured,
            "error",
            measured - expected,
            "peak_u",
            np.max(abs(result.displacement), axis=0),
        )
    return results


def test_p_and_s_arrivals_and_radiation(refinement):
    result = refinement[-1]
    expected, measured = arrival_report(result)
    # Envelope peaks in a 2D tail-bearing Green function are diagnostics, not
    # sharp onsets. This tolerance is 4% of a wavelet period, not one sample.
    assert np.max(abs(measured - expected)) < 0.005
    assert measured[1] - measured[0] > 0.21
    u, time = result.displacement, result.time
    p_window = abs(time - expected[0]) < 0.065
    s_window = abs(time - expected[1]) < 0.065
    p_peaks = np.max(abs(u[p_window, :2, 0]), axis=0)
    s_peaks = np.max(abs(u[s_window, :2, 0]), axis=0)
    print("radiation P window", p_peaks, "S window", s_peaks)
    # Horizontal force: axial far-field P maximum and transverse S maximum.
    # Near-field contributions mean the other branch need not be identically zero.
    assert p_peaks[1] / p_peaks[0] < 0.1
    assert s_peaks[0] / s_peaks[1] < 0.1
    leakage = np.max(abs(u[:, :2, 1]), axis=0) / np.max(abs(u[:, :2, 0]), axis=0)
    print("forbidden component fractions", leakage)
    assert np.max(leakage) < 0.015
    # All samples precede the shortest source->wall->receiver geometric P path,
    # even for a force applied at t=0, not just for its peak at t0.
    cfg = wave_config(640)
    source = np.array(cfg.source.position)
    reflected = []
    for axis in [0, 1]:
        for wall in [cfg.domain.lower[axis], cfg.domain.upper[axis]]:
            image = source.copy()
            image[axis] = 2 * wall - source[axis]
            reflected.extend(np.linalg.norm(result.receiver_coordinates - image, axis=1) / 3200)
    print("earliest geometric boundary return", min(reflected))
    assert result.time[-1] < min(reflected)


def test_fixed_physical_point_source_refinement(refinement):
    # No comparison is made at the singular source; stations are 900 m away.
    traces = [r.displacement[:, :2, 0] for r in refinement]
    differences = [relative_error(a, b) for a, b in zip(traces[:-1], traces[1:], strict=True)]
    s_differences = [
        relative_error(a[:, 1], b[:, 1]) for a, b in zip(traces[:-1], traces[1:], strict=True)
    ]
    amplitudes = np.array([np.max(abs(trace), axis=0) for trace in traces])
    print(
        "refinement trace differences",
        differences,
        "S differences",
        s_differences,
        "peak amplitudes",
        amplitudes,
    )
    assert differences[1] < 0.5 * differences[0]
    assert differences[1] < 0.08
    # Fixed N/m normalization: amplitudes settle, rather than scaling as h or h².
    np.testing.assert_allclose(amplitudes[-1], amplitudes[-2], rtol=0.03)
    s_errors = [abs(arrival_report(r)[1][1] - arrival_report(r)[0][1]) for r in refinement]
    assert s_errors[2] < s_errors[1] < s_errors[0]


@pytest.mark.parametrize("constrained", [False, True])
def test_reciprocity(constrained):
    traces = reciprocity_runs(MPI.COMM_SELF, constrained)
    for i, j, name in [(0, 1, "xx"), (2, 3, "xz/zx")]:
        for observable, quantity in enumerate(["u", "v"]):
            error = relative_error(traces[i, observable], traces[j, observable])
            print("reciprocity", constrained, name, quantity, error)
            # Observed errors are ~1e-14; allow accumulated roundoff across platforms.
            assert error < 2e-12
