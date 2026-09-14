import numpy as np
import pytest
from mpi4py import MPI

from tests.absorbing2d.experiments import (
    packet_run,
    reciprocity_runs,
    seismic_metrics,
    seismic_runs,
)
from tests.experiments2d.helpers import relative_error


@pytest.fixture(scope="module")
def normal_runs():
    results = {}
    for mode in ["P", "S"]:
        for h in [40, 20, 10]:
            for absorbing in [False, True]:
                result = packet_run(mode, h, absorbing=absorbing)
                results[mode, h, absorbing] = result
                print(
                    "normal",
                    mode,
                    h,
                    absorbing,
                    "incident",
                    result["incident_peak"],
                    "reflected amplitude proxy",
                    result["reflection"],
                )
    return results


@pytest.mark.parametrize("mode,limit", [("P", 0.003), ("S", 0.009)])
def test_normal_incidence_refinement_and_free_control(normal_runs, mode, limit):
    residuals = [normal_runs[mode, h, True]["reflection"] for h in [40, 20, 10]]
    assert residuals[2] < residuals[1] < residuals[0]
    assert residuals[2] < limit
    control = normal_runs[mode, 10, False]
    assert control["incident_peak"] > 0.95
    assert 0.95 < control["reflection"] < 1.05
    assert residuals[2] < 0.01 * control["reflection"]
    for h in [40, 20, 10]:
        # The boundary control must not change the incoming packet.
        assert normal_runs[mode, h, True]["incident_peak"] == pytest.approx(
            normal_runs[mode, h, False]["incident_peak"], rel=1e-12
        )


def test_public_free_surface_and_artificial_returns():
    runs = seismic_runs()
    metrics = seismic_metrics(runs)
    print("public seismic", metrics)
    assert metrics["late_error_ratio"] < 0.12
    assert max(metrics["late_per_receiver"]) < 0.2
    assert metrics["surface_preservation"] < 1e-4
    assert metrics["surface_reference_error"] < 1e-5
    assert metrics["surface_signal_fraction"] > 0.6
    assert runs["absorbing"].metadata["config"]["boundaries"]["upper"] == "free"
    assert runs["absorbing"].displacement.shape == (1601, 3, 2)


def test_absorbing_reciprocity():
    traces = reciprocity_runs(MPI.COMM_SELF)
    for i, j, name in [(0, 1, "xx"), (2, 3, "xz/zx")]:
        for k, field in enumerate(["u", "v"]):
            error = relative_error(traces[i, k], traces[j, k])
            print("damped reciprocity", name, field, error)
            assert error < 2e-12


def test_boundary_force_startup_reciprocity_is_second_order():
    # The unchanged Taylor startup preserves a0=D^-1 f0. If a point force acts
    # directly on damped DOFs and f(0)!=0, its initial discrete impulse weighting
    # is not the same as an interior force. This is an O(dt²) startup error,
    # not asymmetry of C. Quantify it rather than claim bitwise reciprocity here.
    errors = []
    for dt in [0.004, 0.002, 0.001]:
        traces = reciprocity_runs(dt=dt, boundary_point=True)
        errors.append(
            [
                relative_error(traces[i, k], traces[j, k])
                for i, j in [(0, 1), (2, 3)]
                for k in [0, 1]
            ]
        )
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("boundary-load startup reciprocity", errors, "rates", rates)
    assert np.all((rates > 1.9) & (rates < 2.1))
    assert max(errors[-1]) < 4e-4
