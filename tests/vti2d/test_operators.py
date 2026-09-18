import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import dense

from .helpers import ISO, RHO, VTI, C, config, record


def exact_matrices(op):
    """Independent physical-cell P1 engineering-strain integration."""
    mass, stiffness = np.zeros((op.n, op.n)), np.zeros((op.n, op.n))
    for cell in range(op.mesh.topology.index_map(2).size_local):
        nodes = op.V.dofmap.cell_dofs(cell)
        xy = op.coordinates[nodes]
        material = op.config.material
        if hasattr(material, "layers"):
            material = next(
                layer.material
                for layer in material.layers
                if layer.lower < xy[:, 1].mean() < layer.upper
            )
        if hasattr(material, "c11"):
            c11, c33, c13, c55 = material.c11, material.c33, material.c13, material.c55
        else:
            c11 = c33 = material.density * material.vp**2
            c55 = material.density * material.vs**2
            c13 = c11 - 2 * c55
        D = np.array([[c11, c13, 0], [c13, c33, 0], [0, 0, c55]])
        affine = np.column_stack((np.ones(3), xy))
        area = abs(np.linalg.det(affine)) / 2
        gradients = np.linalg.inv(affine)[1:, :].T
        B = np.zeros((3, 6))
        B[0, 0::2] = gradients[:, 0]
        B[1, 1::2] = gradients[:, 1]
        B[2, 0::2] = gradients[:, 1]
        B[2, 1::2] = gradients[:, 0]
        ids = (2 * nodes[:, None] + np.arange(2)).ravel()
        stiffness[np.ix_(ids, ids)] += area * B.T @ D @ B
        mass[np.ix_(ids, ids)] += (
            material.density * area / 12 * np.kron(np.ones((3, 3)) + np.eye(3), np.eye(2))
        )
    return mass, stiffness


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("diagonal", ["left", "right", "left_right", "right_left"])
def test_independent_assembly_energy_and_material_fields(layered, diagonal):
    cfg = config(
        layered, domain=dict(lower=(-1, -1), upper=(1, 1), cells=(4, 4), diagonal=diagonal)
    )
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        M, K = exact_matrices(op)
        error = float(abs(K - dense(op.K)).max() / abs(K).max())
        assert error < 8e-16
        np.testing.assert_allclose(dense(op.M), M, rtol=5e-15, atol=1e-13)
        np.testing.assert_allclose(dense(op.K), dense(op.K).T, rtol=1e-15, atol=1e-5)
        assert np.linalg.eigvalsh(dense(op.K)).min() > -1e-14 * abs(K).max()
        assert op.mass.min() > 0
        assert op.C is None
        assert np.all(op.damping == 0)
        if layered:
            f = op.material_fields
            centers = op.mesh.geometry.x[op.mesh.geometry.dofmaps[0], 1].mean(axis=1)
            np.testing.assert_array_equal(f.cell_layers, (centers > 0).astype(int))
            iso = (ISO["density"] * ISO["vp"] ** 2,) * 2 + (
                ISO["density"] * (ISO["vp"] ** 2 - 2 * ISO["vs"] ** 2),
                ISO["density"] * ISO["vs"] ** 2,
            )
            for field, lo, hi in zip(
                [f.rho, f.c11, f.c33, f.c13, f.c55], [ISO["density"], *iso], [RHO, *C], strict=True
            ):
                np.testing.assert_allclose(
                    field.x.array[f.cell_dofs], np.where(centers > 0, hi, lo), rtol=2e-15
                )
        record(f"assembly-{layered}-{diagonal}", dict(relative_stiffness_error=error))


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_isotropic_limit(layers):
    lam = ISO["density"] * (ISO["vp"] ** 2 - 2 * ISO["vs"] ** 2)
    mu = ISO["density"] * ISO["vs"] ** 2
    vti = dict(
        type="vti", density=ISO["density"], c11=lam + 2 * mu, c33=lam + 2 * mu, c13=lam, c55=mu
    )
    outputs = []
    for m in [ISO, vti]:
        material = (
            m
            if not layers
            else dict(
                type="layered",
                layers=[
                    dict(lower=-1 + 2 * i / layers, upper=-1 + 2 * (i + 1) / layers, material=m)
                    for i in range(layers)
                ],
            )
        )
        data = config(material=material).model_dump(mode="json", by_alias=True)
        data.update(
            time=dict(dt=1e-5, duration=0.002),
            source=dict(
                position=(0.13, -0.37),
                direction=(1, 2),
                wavelet=dict(f0=2000, amplitude=1e6, time_shift=0.0003),
            ),
            receivers=[dict(name="r", position=(0.21, 0.17))],
        )
        with Simulation2D(SimulationConfig2D.model_validate(data), MPI.COMM_SELF) as sim:
            result = sim.run()
            op = sim.operators
            u = np.sin(op.coordinates[:, 0:1] + 2 * op.coordinates[:, 1:2]) * np.array([1.0, -0.7])
            action = op.apply(u.ravel()).copy()
            diag = op.spectral_diagnostic()
            outputs.append(
                dict(
                    M=dense(op.M),
                    K=dense(op.K),
                    mass=op.mass.copy(),
                    dt=np.array(op.stable_dt),
                    action=action,
                    energy=np.array(0.5 * u.ravel() @ action),
                    eigen=np.array(diag.lambda_max),
                    u=result.displacement,
                    v=result.velocity,
                )
            )
    errors = {
        k: float(abs(a - outputs[1][k]).max() / max(abs(a).max(), 1e-30))
        for k, a in outputs[0].items()
    }
    assert max(errors.values()) < 2e-12
    record(f"isotropic-{layers}", errors)


@pytest.mark.parametrize("contrast", [1, 30])
def test_spectral_bound_and_source_free_stability(contrast):
    high = VTI | {key: VTI[key] * contrast for key in ["c11", "c33", "c13", "c55"]}
    cfg = config(
        material=dict(
            type="layered",
            layers=[dict(lower=-1, upper=0, material=VTI), dict(lower=0, upper=1, material=high)],
        ),
        # Remove rigid translation before a long-time energy-roundoff audit.
        constraints=[dict(side="left", components=["x"]), dict(side="lower", components=["z"])],
    )
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        K = dense(op.K)[np.ix_(~op.fixed, ~op.fixed)]
        m = op.mass[~op.fixed]
        largest = np.linalg.eigvalsh(K / np.sqrt(m[:, None] * m[None, :]))[-1]
        assert op.stable_dt <= 2 / np.sqrt(largest) * (1 + 1e-14)
        rng = np.random.default_rng(83)
        dt = 0.8 * op.stable_dt
        step = op.start(dt, u0=rng.normal(size=op.n) * 1e-4, v0=rng.normal(size=op.n))
        energies = []
        for _ in range(1000):
            nxt, _, _, energy = step.evaluate(np.zeros(op.n), energy=True)
            energies.append(energy)
            step.advance(nxt)
        drift = float(np.ptp(energies) / energies[0])
        assert min(energies) > 0
        assert drift < 2e-13
        with pytest.raises(ValueError, match="spectral bound"):
            op.start(1.01 * op.stable_dt)
        record(
            f"stability-{contrast}",
            dict(dt=dt, energy_drift=drift, bound_ratio=float(op.stable_dt * np.sqrt(largest) / 2)),
        )


def test_stiffness_scaling():
    dts = []
    for factor in [1, 4]:
        with PlaneStrainOperators(
            config(material=VTI | {k: VTI[k] * factor for k in ["c11", "c33", "c13", "c55"]}),
            MPI.COMM_SELF,
        ) as op:
            dts.append(op.stable_dt)
    assert abs(dts[0] / dts[1] - 2) < 1e-14
