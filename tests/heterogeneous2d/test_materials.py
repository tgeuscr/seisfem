import copy

import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from seisfem import Simulation2D
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.materials2d import CellMaterials2D
from tests.experiments2d.helpers import small_config
from tests.heterogeneous2d.helpers import HIGH, LOW, config, independent_matrices, layers
from tests.plane_strain.helpers import dense


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right", "right_left"])
def test_assignment_mass_stiffness_and_energy(diagonal):
    cfg = config(domain=dict(lower=(-1, -2), upper=(2, 2), cells=(6, 8), diagonal=diagonal))
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        fields = op.material_fields
        vertices = op.coordinates[op.V.dofmap.list]
        expected = (vertices[:, :, 1].min(axis=1) >= -1e-14).astype(int)
        np.testing.assert_array_equal(fields.cell_layers, expected)
        for j, field in enumerate((fields.rho, fields.lam, fields.mu)):
            values = np.array([(m.material.density, *m.material.lame) for m in cfg.material.layers])
            np.testing.assert_array_equal(field.x.array[fields.cell_dofs], values[expected, j])
        M, K = independent_matrices(op)
        np.testing.assert_allclose(dense(op.M), M, rtol=3e-14, atol=2e-15)
        np.testing.assert_allclose(dense(op.K), K, rtol=3e-14, atol=1e-12)
        np.testing.assert_allclose(op.mass, M.sum(axis=1), rtol=3e-14, atol=1e-15)
        np.testing.assert_allclose(
            op.mass.reshape(-1, 2).sum(axis=0), 6 * (LOW["density"] + HIGH["density"]), rtol=3e-15
        )
        assert op.mass.min() > 0
        np.testing.assert_allclose(dense(op.K), dense(op.K).T, atol=1e-13)
        assert np.linalg.eigvalsh(dense(op.K)).min() > -2e-12
        v = np.random.default_rng(402).normal(size=op.n)
        assert v @ op.apply(v) > 0
        # Constant strain has an independently integrated piecewise energy.
        x, z = op.coordinates.T
        affine = np.column_stack((0.2 * x + 0.3 * z, -0.1 * x + 0.4 * z)).ravel()
        energy = 0.0
        for layer in cfg.material.layers:
            lam, mu = layer.material.lame
            energy += 6 * (lam * 0.6**2 + 2 * mu * (0.2**2 + 0.4**2 + 2 * 0.1**2))
        assert affine @ op.apply(affine) == pytest.approx(energy, rel=2e-14)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_homogeneous_limit_matrices_and_results(fixed, count):
    cfg = small_config(constraints=[dict(side="left", components=["x"])] if fixed else [])
    with Simulation2D(cfg, MPI.COMM_SELF) as sim:
        a = sim.run()
        M, K, mass, dt = (
            dense(sim.operators.M),
            dense(sim.operators.K),
            sim.operators.mass.copy(),
            sim.operators.stable_dt,
        )
    data = cfg.model_dump(mode="json", by_alias=True)
    mat = data["material"]
    data["material"] = dict(
        type="layered",
        layers=[dict(lower=i / count, upper=(i + 1) / count, material=mat) for i in range(count)],
    )
    with Simulation2D(data, MPI.COMM_SELF) as sim:
        b = sim.run()
        np.testing.assert_allclose(dense(sim.operators.M), M, rtol=3e-15, atol=2e-17)
        np.testing.assert_allclose(dense(sim.operators.K), K, rtol=3e-15, atol=3e-14)
        np.testing.assert_allclose(sim.operators.mass, mass, rtol=3e-15)
        assert sim.operators.stable_dt == pytest.approx(dt, rel=3e-15)
    for name in ["displacement", "velocity"]:
        error = np.linalg.norm(getattr(a, name) - getattr(b, name)) / np.linalg.norm(
            getattr(a, name)
        )
        print("homogeneous limit", fixed, count, name, error)
        assert error < 3e-13


def test_scaling_and_heterogeneous_spectral_bound():
    cfg = config()
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        M, K, dt = dense(op.M), dense(op.K), op.stable_dt
    data = cfg.model_dump(mode="json", by_alias=True)
    for layer in data["material"]["layers"]:
        layer["material"]["density"] *= 3
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        np.testing.assert_allclose(dense(op.M), 3 * M, rtol=3e-15, atol=3e-16)
        np.testing.assert_allclose(dense(op.K), 3 * K, rtol=3e-15, atol=3e-13)
        assert op.stable_dt == pytest.approx(dt, rel=3e-15)
    data = cfg.model_dump(mode="json", by_alias=True)
    for layer in data["material"]["layers"]:
        layer["material"]["vp"] *= 2
        layer["material"]["vs"] *= 2
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        np.testing.assert_allclose(dense(op.M), M, rtol=3e-15)
        np.testing.assert_allclose(dense(op.K), 4 * K, rtol=3e-15, atol=3e-13)
        assert op.stable_dt == pytest.approx(dt / 2, rel=3e-15)
    data["material"]["layers"][0]["material"] = LOW
    with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
        print("safe dt baseline / faster upper layer", dt, op.stable_dt)
        assert op.stable_dt < 0.7 * dt
        with pytest.raises(ValueError, match="spectral bound"):
            op.start(0.9 * dt)


def invalid_models():
    good = config().model_dump(mode="json", by_alias=True)
    cases = []
    for key, value in [
        ("density", 0),
        ("density", -1),
        ("vp", float("nan")),
        ("vs", float("inf")),
        ("vp", 1),
    ]:
        data = copy.deepcopy(good)
        data["material"]["layers"][0]["material"][key] = value
        cases.append(data)
    for index, key, value in [
        (0, "lower", -3),
        (1, "upper", 3),
        (1, "lower", 0.1),
        (1, "lower", -0.1),
    ]:
        data = copy.deepcopy(good)
        data["material"]["layers"][index][key] = value
        cases.append(data)
    data = copy.deepcopy(good)
    data["material"]["layers"].reverse()
    cases.append(data)
    data = copy.deepcopy(good)
    data["material"]["layers"][0]["upper"] = 0.1
    data["material"]["layers"][1]["lower"] = 0.1
    cases.append(data)
    cases.append(good | dict(boundaries=dict(right="absorbing")))
    cases.append(good | dict(material=dict(type="layered", layers=[])))
    return cases


@pytest.mark.parametrize("data", invalid_models())
def test_invalid_layers(data):
    with pytest.raises(ValueError):
        PlaneStrainConfig.model_validate(data)
    with pytest.raises(ValueError):
        Simulation2D(data | dict(time=dict(dt=0.001, duration=0.01)), MPI.COMM_SELF)


def test_vertex_guard_rejects_cut_cells_and_accepts_negative_lambda():
    cfg = config()
    wrong_mesh = mesh.create_rectangle(
        MPI.COMM_SELF, [[-1, -2], [2, 2]], [6, 3], cell_type=mesh.CellType.triangle
    )
    with pytest.raises(ValueError, match="cuts a cell"):
        CellMaterials2D(wrong_mesh, cfg)
    auxetic = dict(density=2.3, **{"lambda": -0.5}, mu=1.2)
    with PlaneStrainOperators(
        config(material=layers(-2, 0, 2, auxetic, auxetic)), MPI.COMM_SELF
    ) as op:
        assert np.all(op.material_fields.lam.x.array == -0.5)
        assert np.min(op.mass) > 0
