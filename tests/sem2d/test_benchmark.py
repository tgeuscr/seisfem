import numpy as np
import pytest
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from seisfem.timestepping import CentralDifference
from tests.plane_strain.helpers import dense

from .experiment import run
from .helpers import config, record


@pytest.mark.parametrize("branch", ["P", "S"])
def test_equal_dof_comparison(benchmark, branch):
    rows = [benchmark[p, 96, branch, 0.02] for p in [0, 2, 4, 6]]
    assert len({r["dofs"] for r in rows}) == 1
    for r in rows[1:]:
        assert r["speed_error"] < 0.1 * rows[0]["speed_error"]
        assert r["field_error"] < 0.2 * rows[0]["field_error"]
    assert rows[3]["dtcrit"] < rows[2]["dtcrit"] < rows[1]["dtcrit"]
    for r in rows:
        assert r["work_proxy"] == r["dofs"] * r["steps"]
        assert r["backend_seconds"] > r["stepping_seconds"] > 0


@pytest.mark.parametrize("branch", ["P", "S"])
def test_order_at_fixed_element_count(benchmark, branch):
    rows = []
    for p in [2, 4, 6]:
        r = (
            benchmark[p, 48, branch, 0.02]
            if p == 4
            else run(p, 12 * p, branch, fractions=(0.02,))[0]
        )
        rows.append(r)
        record(f"order-p{p}-{branch}", r)
    assert len({r["elements"] for r in rows}) == 1
    for key in ["speed_error", "field_error", "waveform_error"]:
        errors = [r[key] for r in rows]
        assert errors[2] < errors[1] < errors[0], (branch, key, errors)


def test_temporal_cancellation_is_not_spatial_convergence(benchmark):
    rows = [benchmark[2, 96, "P", f] for f in [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]]
    signed = [r["signed_speed_error"] for r in rows]
    assert signed[0] < 0 < signed[-1]
    assert 0 < np.argmin(abs(np.array(signed))) < len(signed) - 1
    for p in [0, 2, 4, 6]:
        for mode in ["P", "S"]:
            for f in [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]:
                assert benchmark[p, 96, mode, f]["energy_drift"] < 1e-11


@pytest.mark.parametrize("p", [2, 4, 6])
def test_time_convergence_to_exact_semidiscrete_solution(p):
    with PlaneStrainOperators(config(p, cells=(2, 2)), MPI.COMM_SELF) as op:
        K = dense(op.K)
        ev, Q = np.linalg.eigh(K / np.sqrt(np.outer(op.mass, op.mass)))
        initial = Q[:, 7] / np.sqrt(op.mass)
        exact = initial * np.cos(np.sqrt(ev[7]) * 0.1)
        errors = []
        for count in [40, 80, 160, 320]:
            dt = 0.1 / count
            assert dt < op.spectral_diagnostic().critical_dt
            step = CentralDifference(
                op.mass, op.damping, op.fixed, dt, op.apply, initial, np.zeros(op.n), np.zeros(op.n)
            )
            for _ in range(count):
                nxt, *_ = step.evaluate(np.zeros(op.n))
                step.advance(nxt)
            errors.append(float(np.sqrt(op.mass @ (step.current - exact) ** 2)))
        rates = np.log2(np.array(errors[:-1]) / errors[1:])
        assert np.all((rates > 1.98) & (rates < 2.02))
        record(f"temporal-exact-{p}", dict(errors=errors, rates=rates.tolist()))


@pytest.mark.parametrize("branch", ["P", "S"])
def test_tri_accuracy_extension(branch):
    # Predetermined extension gives the baseline a chance to reach 1% waveform
    # and 0.1% phase error. Failure to attain a target remains a reported result.
    for n in [192, 288, 384]:
        row = run(0, n, branch, fractions=(0.1,))[0]
        record(f"accuracy-tri-n{n}-{branch}", row)
        assert row["energy_drift"] < 1e-11
        assert row["speed_error"] < 0.015
