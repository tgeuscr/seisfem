import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from tests.experiments2d.helpers import small_config
from tests.plane_strain.helpers import dense


def config(sides=("right",), scale=1, density=2.3, vp=3.2, vs=1.8, **updates):
    data = dict(
        domain=dict(lower=(-scale, 2 * scale), upper=(2 * scale, 4 * scale), cells=(6, 4)),
        material=dict(density=density, vp=vp, vs=vs),
        boundaries={side: "absorbing" for side in sides},
    )
    return PlaneStrainConfig.model_validate(data | updates)


def hand_boundary_matrix(op):
    """Independent edge mass h/6 [[2,1],[1,2]] with Cartesian impedances."""
    cfg = op.config
    answer = np.zeros((op.n, op.n))
    sides = {
        "left": (0, cfg.domain.lower[0]),
        "right": (0, cfg.domain.upper[0]),
        "lower": (1, cfg.domain.lower[1]),
        "upper": (1, cfg.domain.upper[1]),
    }
    for side in cfg.boundaries.absorbing_sides:
        normal, value = sides[side]
        tangent = 1 - normal
        nodes = np.flatnonzero(abs(op.coordinates[:, normal] - value) < 1e-12)
        nodes = nodes[np.argsort(op.coordinates[nodes, tangent])]
        for a, b in zip(nodes[:-1], nodes[1:], strict=True):
            length = op.coordinates[b, tangent] - op.coordinates[a, tangent]
            for component in [0, 1]:
                speed = cfg.material.vp if component == normal else cfg.material.vs
                indices = [2 * a + component, 2 * b + component]
                answer[np.ix_(indices, indices)] += (
                    cfg.material.density * speed * length / 6 * np.array([[2, 1], [1, 2]])
                )
    return answer


@pytest.mark.parametrize("side", ["left", "right", "lower", "upper"])
def test_side_integrals_symmetry_and_positive_dissipation(side):
    cfg = config((side,))
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        C = dense(op.C)
        exact = hand_boundary_matrix(op)
        np.testing.assert_allclose(C, exact, atol=3e-15, rtol=2e-15)
        np.testing.assert_allclose(op.damping, C.sum(axis=1), atol=2e-15)
        np.testing.assert_allclose(C, C.T, atol=1e-15)
        assert np.min(np.linalg.eigvalsh(C)) >= -3e-15
        assert op.damping.min() == 0
        normal = 0 if side in ("left", "right") else 1
        length = 2 if normal == 0 else 3
        sums = op.damping.reshape(-1, 2).sum(axis=0)
        expected = cfg.material.density * length * np.array([cfg.material.vp, cfg.material.vs])
        np.testing.assert_allclose(sums[[normal, 1 - normal]], expected, atol=8e-15)
        print("side integrals", side, "errors", sums[[normal, 1 - normal]] - expected)
        for component in [normal, 1 - normal]:
            v = np.zeros(op.n)
            v[component::2] = 1
            assert v @ (op.damping * v) == pytest.approx(sums[component], abs=1e-14)
        rng = np.random.default_rng(502)
        for _ in range(10):
            v = rng.normal(size=op.n)
            assert v @ C @ v >= 0
            assert np.dot(op.damping * v, v) >= 0


@pytest.mark.parametrize(
    "change,factors",
    [
        (dict(density=6.9), (3, 3)),
        (dict(vp=6.4), (2, 1)),
        (dict(vs=2.16), (1, 1.2)),
        (dict(scale=2), (2, 2)),
    ],
)
def test_impedance_scaling(change, factors):
    with PlaneStrainOperators(config(), MPI.COMM_SELF) as base:
        reference = base.damping.reshape(-1, 2).copy()
    with PlaneStrainOperators(config(**change), MPI.COMM_SELF) as changed:
        np.testing.assert_allclose(changed.damping.reshape(-1, 2), reference * factors, atol=4e-15)


def test_corners_mixed_free_and_essential_conditions():
    cfg = config(("left", "right", "lower"))
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        np.testing.assert_allclose(dense(op.C), hand_boundary_matrix(op), atol=3e-15)
        x, z = op.coordinates[: op.n // 2].T
        interior_top = (abs(z - 4) < 1e-12) & (x > -1 + 1e-12) & (x < 2 - 1e-12)
        np.testing.assert_array_equal(op.damping.reshape(-1, 2)[interior_top], 0)
        corner = np.argmin((x - 2) ** 2 + (z - 2) ** 2)
        # h_x=h_z=0.5, each incident facet gives its own endpoint weight h/2.
        expected = 2.3 * 0.25 * np.array([3.2 + 1.8, 1.8 + 3.2])
        np.testing.assert_allclose(op.damping.reshape(-1, 2)[corner], expected, atol=2e-15)
    cfg = config(("right",), constraints=[dict(side="upper", components=["x"])])
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        assert np.any(op.damping[op.fixed] > 0)  # Contributions retained before projection.
        step = op.start(0.01, u0=np.ones(op.n), v0=np.ones(op.n), force0=np.ones(op.n))
        nxt, v, _, _ = step.evaluate(np.ones(op.n))
        np.testing.assert_array_equal(nxt[op.fixed], 0)
        np.testing.assert_array_equal(v[op.fixed], 0)


@pytest.mark.parametrize(
    "boundaries,constraints",
    [
        (dict(right="absorbing"), [dict(side="right", components=["x"])]),
        (dict(upper="absorbing"), [dict(side="upper")]),
        (dict(top="absorbing"), []),
        (dict(lower="pml"), []),
        (dict(left=float("nan")), []),
    ],
)
def test_invalid_boundary_configuration(boundaries, constraints):
    with pytest.raises(ValueError):
        config((), boundaries=boundaries, constraints=constraints)
    data = small_config().model_dump(mode="json", by_alias=True)
    with pytest.raises(ValueError):
        Simulation2D(data | dict(boundaries=boundaries, constraints=constraints), MPI.COMM_SELF)


def test_zero_damping_and_stability_unchanged():
    for fixed in [False, True]:
        constraints = [dict(side="left", components=["x"])] if fixed else []
        cfg = small_config(constraints=constraints)
        with Simulation2D(cfg, MPI.COMM_SELF) as sim:
            a = sim.run()
            assert sim.operators.C is None
            np.testing.assert_array_equal(sim.operators.damping, 0)
            dt = sim.operators.stable_dt
        data = cfg.model_dump(mode="json", by_alias=True)
        data["boundaries"] = {side: "free" for side in ["left", "right", "lower", "upper"]}
        with Simulation2D(data, MPI.COMM_SELF) as sim:
            b = sim.run()
            assert sim.operators.stable_dt == dt
        np.testing.assert_array_equal(a.displacement, b.displacement)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        data["boundaries"]["right"] = "absorbing"
        with Simulation2D(data, MPI.COMM_SELF) as sim:
            assert sim.operators.stable_dt == dt
            with pytest.raises(ValueError, match="spectral bound"):
                sim.operators.start(dt)
