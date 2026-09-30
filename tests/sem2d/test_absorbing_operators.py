"""Collocated damping, facet material, corners, and exact discrete work."""

import numpy as np
import pytest
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from tests.heterogeneous2d.helpers import HIGH, LOW
from tests.plane_strain.helpers import dense

from .helpers import record
from .reference import boundary_matrix

SIDES = [
    ("right",),
    ("right", "lower"),
    ("left", "right", "lower", "upper"),
    ("left", "right", "lower"),
]


def config(p=4, sides=SIDES[-1], layered=True):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-1, -1.5), upper=(2, 1.5), cells=(2, 3)),
            discretization=dict(type="quad_gll", degree=p),
            boundaries={side: "absorbing" for side in sides},
            material=dict(
                type="layered",
                layers=[
                    dict(lower=-1.5, upper=-0.5, material=LOW),
                    dict(lower=-0.5, upper=0.5, material=HIGH),
                    dict(lower=0.5, upper=1.5, material=LOW),
                ],
            )
            if layered
            else LOW,
        )
    )


@pytest.mark.parametrize("p", [1, 2, 4, 6])
@pytest.mark.parametrize("sides", SIDES)
@pytest.mark.parametrize("layered", [False, True])
def test_independent_edges_and_corners(p, sides, layered):
    cfg = config(p, sides, layered)
    rows = (
        [(-1.5, -0.5, LOW), (-0.5, 0.5, HIGH), (0.5, 1.5, LOW)] if layered else [(-1.5, 1.5, LOW)]
    )
    layers = [(a, b, m["density"], m["vp"], m["vs"]) for a, b, m in rows]
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        C = dense(op.C)
        ref = boundary_matrix(
            p, op.coordinates, op.V.dofmap.list, (cfg.domain.lower, cfg.domain.upper), sides, layers
        )
        error = float(np.linalg.norm(C - ref) / np.linalg.norm(ref))
        assert error < 3e-11
        np.testing.assert_array_equal(op.damping, np.diag(C))
        off = float(np.max(abs(C - np.diag(np.diag(C)))) / np.max(op.damping))
        assert off < 1e-13
        assert op.damping.min() >= 0
        assert np.max(abs(C - C.T)) < 2e-14 * np.max(op.damping)
        assert op.sem_metadata["sum_factorization"]
        # Explicit bottom-right corner: both adjacent segment endpoints add.
        if "right" in sides and "lower" in sides:
            node = np.flatnonzero(np.linalg.norm(op.coordinates - [2, -1.5], axis=1) < 1e-12)[0]
            rho, vp, vs = (LOW[k] for k in ["density", "vp", "vs"])
            weight = 1 / (p * (p + 1))
            exact = weight * rho * (np.array([vp, vs]) * 1 + np.array([vs, vp]) * 1.5)
            np.testing.assert_allclose(op.damping[2 * node : 2 * node + 2], exact, rtol=3e-14)
        record(
            f"absorbing-operator-p{p}-{'-'.join(sides)}-layered{layered}",
            dict(degree=p, sides=sides, layered=layered, reference_error=error, offdiagonal=off),
        )


@pytest.mark.parametrize("p", [1, 2, 4, 6])
def test_work_balance_and_unchanged_stability(p):
    with PlaneStrainOperators(config(p), MPI.COMM_SELF) as op:
        with PlaneStrainOperators(config(p, sides=()), MPI.COMM_SELF) as free:
            assert free.C is None
            np.testing.assert_array_equal(free.damping, 0)
            np.testing.assert_array_equal(dense(free.M), dense(op.M))
            np.testing.assert_array_equal(dense(free.K), dense(op.K))
            np.testing.assert_array_equal(free.mass, op.mass)
            assert free.stable_dt == op.stable_dt
        dt = 0.8 * op.stable_dt
        K = dense(op.K)
        critical = 2 / np.sqrt(np.linalg.eigvalsh(K / np.sqrt(np.outer(op.mass, op.mass)))[-1])
        assert op.stable_dt <= critical
        assert abs(op.spectral_diagnostic().critical_dt / critical - 1) < 3e-11
        rng = np.random.default_rng(702)
        step = op.start(dt, u0=0.01 * rng.normal(size=op.n), v0=0.02 * rng.normal(size=op.n))
        half = (step.current - step.previous) / dt
        energy = 0.5 * (op.mass * half) @ half + 0.5 * step.current @ op.apply(step.previous)
        initial = energy
        work = 0.0
        residual = 0.0
        for _ in range(int(np.ceil(20 / dt))):
            nxt, v, _, new = step.evaluate(np.zeros(op.n), energy=True)
            loss = dt * (op.damping * v) @ v
            residual = max(residual, abs(new - energy + loss) / initial)
            assert new <= energy + 3e-15 * initial
            assert new >= -3e-15 * initial
            work += loss
            energy = new
            step.advance(nxt)
        cumulative = abs(energy + work - initial) / initial
        assert residual < 3e-14
        assert cumulative < 3e-13
        assert energy / initial < 0.1
        record(
            f"absorbing-energy-p{p}",
            dict(
                balance=residual,
                cumulative=cumulative,
                remaining=energy / initial,
                dt=dt,
                dtcrit=critical,
            ),
        )


def test_configuration():
    config()
    for sides in SIDES:
        config(sides=sides)
    data = config().model_dump(mode="json", by_alias=True)
    with pytest.raises(ValueError, match="displacement constraints"):
        PlaneStrainConfig.model_validate(
            data | dict(constraints=[dict(side="upper", components=["x"])])
        )
