import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from pydantic import ValidationError

from seisfem.config2d import PlaneStrainConfig, Rectangle, ZeroDisplacement
from seisfem.fem2d import PlaneStrainOperators, strain, stress
from tests.plane_strain.helpers import configuration, dense, hand_matrices


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right", "right_left"])
@pytest.mark.parametrize("lam", [1.7, -0.7])
def test_hand_mass_stiffness_and_rigid_modes(diagonal, lam):
    with PlaneStrainOperators(configuration(diagonal=diagonal, lam=lam), MPI.COMM_SELF) as op:
        M, K = dense(op.M), dense(op.K)
        expected_m, expected_k = hand_matrices(op)
        np.testing.assert_allclose(M, expected_m, rtol=3e-14, atol=1e-16)
        np.testing.assert_allclose(K, expected_k, rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(M, M.T, atol=1e-16)
        np.testing.assert_allclose(K, K.T, atol=3e-14)
        np.testing.assert_allclose(op.mass, expected_m.sum(axis=1), rtol=3e-14)
        assert np.all(op.mass > 0)
        np.testing.assert_array_equal(M[0::2, 1::2], 0)
        np.testing.assert_allclose(op.mass.reshape(-1, 2).sum(axis=0), [2.3, 2.3], atol=3e-14)
        np.testing.assert_allclose(op.mass[0::2], op.mass[1::2], rtol=3e-14)
        x, z = op.coordinates.T
        for field in [
            np.tile([1.0, 0.0], len(x)),
            np.tile([0.0, 1.0], len(x)),
            np.column_stack((-z, x)).ravel(),
        ]:
            np.testing.assert_allclose(op.apply(field), 0, atol=3e-14)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.max(abs(eigenvalues[:3])) < 2e-14 * np.linalg.norm(K, 2)
        assert eigenvalues[3] > 0.01
        rng = np.random.default_rng(91)
        for _ in range(5):
            u = rng.normal(size=op.n)
            assert u @ K @ u > 0


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
@pytest.mark.parametrize(
    "gradient",
    [
        [[0.3, 0], [0, 0]],
        [[0, 0], [0, -0.2]],
        [[0, 0.7], [0, 0]],
        [[0.3, 0.7], [-0.2, -0.4]],
        [[0, -1], [1, 0]],
    ],
)
def test_affine_strain_stress_energy(diagonal, gradient):
    with PlaneStrainOperators(configuration(diagonal=diagonal), MPI.COMM_SELF) as op:
        G = np.array(gradient)
        values = (op.coordinates @ G.T + [0.13, -0.27]).ravel()
        op.field.x.array[:] = values
        eps = (G + G.T) / 2
        sigma = 1.7 * np.trace(eps) * np.eye(2) + 2 * 1.2 * eps
        space = fem.functionspace(op.mesh, ("DG", 0, (2, 2)))
        for actual, expected in [(strain(op.field), eps), (stress(op.field, 1.7, 1.2), sigma)]:
            result = fem.Function(space)
            result.interpolate(fem.Expression(actual, space.element.interpolation_points))
            np.testing.assert_allclose(
                result.x.array.reshape(-1, 2, 2),
                np.broadcast_to(expected, (len(result.x.array) // 4, 2, 2)),
                atol=3e-15,
            )
        # 3D energy with epsilon_yy=0, independent engineering-shear expression.
        expected_energy = 0.5 * 1.7 * (G[0, 0] + G[1, 1]) ** 2 + 1.2 * (
            G[0, 0] ** 2 + G[1, 1] ** 2 + 0.5 * (G[0, 1] + G[1, 0]) ** 2
        )
        assert 0.5 * values @ op.apply(values) == pytest.approx(expected_energy, abs=2e-14)


def test_component_constraints_and_positive_definite_subproblem():
    data = configuration(n=4).model_dump(mode="json", by_alias=True)
    data["constraints"] = [
        dict(side="left", components=["x"]),
        dict(side="lower", components=["z"]),
    ]
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        expected = np.zeros(op.n, bool)
        expected[0::2] = abs(op.coordinates[:, 0]) < 1e-14
        expected[1::2] = abs(op.coordinates[:, 1]) < 1e-14
        np.testing.assert_array_equal(op.fixed, expected)
        M, K = hand_matrices(op)
        np.testing.assert_allclose(dense(op.K), K, atol=3e-14)
        np.testing.assert_allclose(op.mass, M.sum(axis=1), rtol=3e-14)
        assert np.linalg.eigvalsh(K[np.ix_(~expected, ~expected)])[0] > 0.01


@pytest.mark.parametrize(
    "data",
    [
        dict(cells=(0, 2)),
        dict(cells=(True, 2)),
        dict(upper=(0, 1)),
        dict(upper=(float("inf"), 1)),
        dict(diagonal="crossed"),
    ],
)
def test_reject_unsupported_rectangle(data):
    with pytest.raises(ValidationError):
        Rectangle.model_validate(data)


def test_configuration_and_material_separation():
    with pytest.raises(ValidationError):
        ZeroDisplacement(side="left", components=())
    with pytest.raises(ValidationError):
        ZeroDisplacement(side="lower", components=("z", "z"))
    cfg = configuration()
    data = cfg.model_dump(mode="json", by_alias=True)
    data["material"] = dict(density=2.3, vp=np.sqrt(4.1 / 2.3), vs=np.sqrt(1.2 / 2.3))
    other = PlaneStrainConfig.model_validate(data)
    with (
        PlaneStrainOperators(cfg, MPI.COMM_SELF) as a,
        PlaneStrainOperators(other, MPI.COMM_SELF) as b,
    ):
        np.testing.assert_allclose(dense(a.K), dense(b.K), atol=2e-14)
    for key in ["source", "receivers", "absorbing", "layers", "degree"]:
        with pytest.raises(ValidationError):
            PlaneStrainConfig.model_validate({**data, key: 1})


def test_translated_rectangle_geometry_mass_and_energy():
    data = configuration().model_dump(mode="json", by_alias=True)
    data["domain"] = dict(lower=(-2, 1), upper=(1, 2.5), cells=(3, 2), diagonal="left")
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        assert op.mesh.topology.index_map(2).size_global == 12
        np.testing.assert_allclose(op.coordinates.min(axis=0), [-2, 1], atol=1e-14)
        np.testing.assert_allclose(op.coordinates.max(axis=0), [1, 2.5], atol=1e-14)
        np.testing.assert_allclose(
            op.mass.reshape(-1, 2).sum(axis=0), np.full(2, 2.3 * 4.5), atol=3e-14
        )
        M, K = hand_matrices(op)
        np.testing.assert_allclose(dense(op.M), M, rtol=3e-14, atol=1e-15)
        np.testing.assert_allclose(dense(op.K), K, rtol=3e-14, atol=3e-14)
        values = np.column_stack((0.3 * op.coordinates[:, 0], -0.2 * op.coordinates[:, 1])).ravel()
        density = 0.5 * 1.7 * (0.3 - 0.2) ** 2 + 1.2 * (0.3**2 + 0.2**2)
        assert 0.5 * values @ op.apply(values) == pytest.approx(4.5 * density, abs=2e-14)


def test_explicit_zero_load():
    import ufl

    with PlaneStrainOperators(configuration(), MPI.COMM_SELF) as op:
        load = op.assemble_load(ufl.as_vector([0.0, 0.0]))
        assert load.shape == (op.n,)
        np.testing.assert_array_equal(load, 0)


def test_constraints_against_collapsed_component_maps():
    # Independent scalar subspace maps and physical coordinates on a translated,
    # nonsquare rectangle; do not assume 2*node+component in the expected mask.
    data = configuration().model_dump(mode="json", by_alias=True)
    data["domain"] = dict(lower=(-2, 1), upper=(1, 2.5), cells=(3, 2), diagonal="right_left")
    data["constraints"] = [
        dict(side="right", components=["x"]),
        dict(side="upper", components=["z"]),
    ]
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        expected = np.zeros(op.n, dtype=bool)
        for component, endpoint in [(0, 1), (1, 2.5)]:
            scalar_space, maps = op.V.sub(component).collapse()
            (parent_map,) = maps
            coordinates = scalar_space.tabulate_dof_coordinates()
            selected = np.isclose(coordinates[:, component], endpoint, rtol=0, atol=1e-14)
            expected[np.asarray(parent_map)[selected]] = True
            marker = fem.Function(op.V)
            marker.sub(component).interpolate(lambda x: np.full((1, x.shape[1]), 3.7))
            np.testing.assert_allclose(marker.x.array[parent_map], 3.7)
            np.testing.assert_array_equal(np.flatnonzero(marker.x.array), np.sort(parent_map))
        np.testing.assert_array_equal(op.fixed, expected)
        step = op.start(0.001, u0=np.ones(op.n), v0=np.ones(op.n))
        np.testing.assert_array_equal(step.current[expected], 0)
        np.testing.assert_array_equal(step.current[~expected], 1)
