import numpy as np
import pytest
from mpi4py import MPI
from pydantic import ValidationError

from seisfem.config2d import ForceSource2D, RickerWavelet
from seisfem.fem2d import PlaneStrainOperators
from seisfem.points2d import PointMap2D
from tests.plane_strain.helpers import configuration

POSITIONS = [
    (0, 0),
    (1, 1),
    (0.25, 0.25),
    (0.31, 0.43),
    (0.5, 0.5),
    (0.5 - 1e-12, 0.5 + 1e-12),
    (0, 0.37),
    (1, 0.62),
]


def affine(points):
    x, z = np.asarray(points).T
    return np.column_stack((0.7 + 1.3 * x - 2.1 * z, -1.2 + 0.4 * x + 3.2 * z))


def test_ricker_convention_and_extremes():
    wave = RickerWavelet(f0=12, amplitude=2.5e8, time_shift=0.125)
    assert wave(wave.time_shift) == wave.amplitude
    delta = np.linspace(0, 0.25, 101)
    np.testing.assert_allclose(wave(0.125 - delta), wave(0.125 + delta), atol=1e-7, rtol=1e-14)
    other = RickerWavelet(f0=12, amplitude=-5e8, time_shift=0.125)
    np.testing.assert_array_equal(other(delta), -2 * wave(delta))
    assert np.all(np.isfinite(wave(np.array([-1e308, 0, 1e308]))))
    extreme = RickerWavelet(f0=1e308, amplitude=1e308, time_shift=1e308)
    assert extreme(1e308) == 1e308
    assert extreme(-1e308) == 0
    with pytest.raises(ValueError, match="finite"):
        wave(np.nan)


@pytest.mark.parametrize(
    "field,values",
    [
        ("f0", [0, -1, np.nan, np.inf]),
        ("amplitude", [np.nan, np.inf, -np.inf]),
        ("time_shift", [np.nan, np.inf, -np.inf]),
    ],
)
def test_invalid_wavelet(field, values):
    for value in values:
        with pytest.raises(ValidationError):
            RickerWavelet.model_validate(dict(f0=12) | {field: value})


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (3, -4), (1e308, 1e308), (1e-308, 0)])
def test_direction(direction):
    cfg = ForceSource2D(position=(0.3, 0.7), direction=direction, wavelet=RickerWavelet(f0=1))
    assert np.linalg.norm(cfg.unit_direction) == pytest.approx(1, abs=2e-16)
    scale = max(abs(x) for x in direction)
    expected = np.array(direction) / scale
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(cfg.unit_direction, expected, atol=2e-16)


@pytest.mark.parametrize("direction", [(0, 0), (np.nan, 1), (1, np.inf), (1,), (1, 2, 3)])
def test_invalid_direction(direction):
    with pytest.raises(ValidationError):
        ForceSource2D(position=(0, 0), direction=direction, wavelet=RickerWavelet(f0=1))


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right", "right_left"])
def test_affine_receivers_and_source_functional(diagonal):
    with PlaneStrainOperators(configuration(n=4, diagonal=diagonal), MPI.COMM_SELF) as op:
        points = PointMap2D(op.V, POSITIONS)
        np.testing.assert_array_equal(points.ids, np.arange(len(POSITIONS)))
        values = affine(op.coordinates[: op.n // 2]).ravel()
        np.testing.assert_allclose(points.evaluate(values), affine(POSITIONS), atol=9e-16, rtol=0)
        # A separately prescribed velocity field uses precisely the same FE functional.
        np.testing.assert_allclose(
            points.evaluate(-3 * values), -3 * affine(POSITIONS), atol=4e-15, rtol=0
        )
        for direction in [(1, 0), (0, 1), (3, -4)]:
            d = np.array(direction) / np.linalg.norm(direction)
            for position in POSITIONS:
                source = PointMap2D(op.V, [position])
                load = source.unit_load(direction)
                expected = float(affine([position])[0] @ d)
                assert load @ values == pytest.approx(expected, abs=9e-16)
                np.testing.assert_allclose(load.reshape(-1, 2).sum(axis=0), d, atol=3e-16)
                assert np.count_nonzero(np.linalg.norm(load.reshape(-1, 2), axis=1)) <= 3
        interior = PointMap2D(op.V, [(0.31, 0.43)]).unit_load((1, 0))
        assert np.count_nonzero(interior) == 3  # Rules out nearest-node injection.
        empty = PointMap2D(op.V, [])
        assert empty.evaluate(values).shape == (0, 2)
        np.testing.assert_array_equal(empty.unit_load((1, 0)), 0)


@pytest.mark.parametrize(
    "positions", [[(-0.01, 0.2)], [(1.01, 0.2)], [(np.nan, 0)], [(0, np.inf)], [1, 2], [(1, 2, 3)]]
)
def test_invalid_points(positions):
    with PlaneStrainOperators(configuration(n=2), MPI.COMM_SELF) as op:
        with pytest.raises(ValueError):
            PointMap2D(op.V, positions)


def test_translated_nonsquare_mesh_affine_functional():
    from seisfem.config2d import PlaneStrainConfig

    cfg = PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-210.3, 40.2), upper=(892.7, 952.9), cells=(7, 5)),
            material=dict(density=2400, vp=3200, vs=1800),
        )
    )
    positions = [(-210.3, 40.2), (892.7, 952.9), (-210.3, 414.17), (0.123, 235.71)]
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        points = PointMap2D(op.V, positions)
        field = affine(op.coordinates).ravel()
        np.testing.assert_allclose(
            points.evaluate(field), affine(positions), atol=1e-12, rtol=2e-15
        )
        for i, position in enumerate(positions):
            source = PointMap2D(op.V, [position])
            load = source.unit_load((3, 4))
            assert load @ field == pytest.approx(affine(positions)[i] @ [0.6, 0.8], abs=2e-12)
