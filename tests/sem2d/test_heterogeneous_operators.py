"""Independent cell-constant material, mass, stiffness and stability checks."""

import numpy as np
import pytest
from mpi4py import MPI

from seisfem import Simulation2D, SimulationConfig2D
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from tests.heterogeneous2d.helpers import HIGH, LOW, layers
from tests.plane_strain.helpers import dense

from .helpers import record
from .reference import layered_matrices


def config(p=4, material=None):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-1, -1), upper=(2, 1), cells=(2, 2)),
            material=material or layers(-1, 0, 1),
            discretization=dict(type="quad_gll", degree=p),
        )
    )


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("p", range(1, 7))
def test_layered_reference_and_stability(p):
    with PlaneStrainOperators(config(p), MPI.COMM_SELF) as op:
        assert op.sem_metadata["sum_factorization"]
        assert (
            op.mesh.ufl_domain()
            .ufl_coordinate_element()
            .basix_element.has_tensor_product_factorisation
        )
        f = op.material_fields
        assert f.rho.function_space.element.basix_element.has_tensor_product_factorisation
        cells = op.V.dofmap.list
        expected = (op.coordinates[cells, 1].min(axis=1) >= -1e-13).astype(int)
        np.testing.assert_array_equal(f.cell_layers, expected)
        for field, values in [
            (f.rho, [LOW["density"], HIGH["density"]]),
            (f.mu, [m["density"] * m["vs"] ** 2 for m in [LOW, HIGH]]),
        ]:
            np.testing.assert_array_equal(field.x.array[f.cell_dofs], np.array(values)[expected])
        rows = [
            (lo, hi, m["density"], m["vp"], m["vs"]) for lo, hi, m in [(-1, 0, LOW), (0, 1, HIGH)]
        ]
        M, K = dense(op.M), dense(op.K)
        a, b = layered_matrices(p, op.coordinates, cells, rows)
        errors = dict(M=relative(M, a), K=relative(K, b))
        assert errors["M"] < 3e-11
        assert errors["K"] < 3e-10
        x = np.random.default_rng(482).normal(size=op.n)
        errors["action"] = relative(op.apply(x), b @ x)
        assert errors["action"] < 3e-10
        off = np.max(abs(M - np.diag(np.diag(M)))) / np.max(op.mass)
        assert off < 1e-13
        total = 3 * (LOW["density"] + HIGH["density"])
        mass_error = float(np.max(abs(op.mass.reshape(-1, 2).sum(axis=0) / total - 1)))
        assert mass_error < 3e-14
        symmetry = float(np.max(abs(K - K.T)) / np.max(abs(K)))
        rigid = max(
            float(np.max(abs(K @ np.tile(d, op.n // 2))) / np.max(abs(K))) for d in [(1, 0), (0, 1)]
        )
        assert symmetry < 2e-14
        assert rigid < 3e-13
        ev = np.linalg.eigvalsh(K / np.sqrt(np.outer(op.mass, op.mass)))
        assert ev[0] > -1e-12 * ev[-1]
        critical = 2 / np.sqrt(ev[-1])
        dt_error = abs(op.spectral_diagnostic().critical_dt / critical - 1)
        assert dt_error < 3e-11
        assert op.stable_dt <= critical
        dt = 0.5 * op.stable_dt
        step = op.start(dt, u0=x, v0=np.zeros(op.n))
        energies = []
        for _ in range(200):
            nxt, _, _, energy = step.evaluate(np.zeros(op.n), energy=True)
            energies.append(energy)
            step.advance(nxt)
        drift = float(np.ptp(energies) / np.mean(energies))
        assert drift < 3e-13
        record(
            f"heterogeneous-operators-{p}",
            dict(
                degree=p,
                errors=errors,
                mass_offdiagonal_relative=float(off),
                integrated_mass_error=mass_error,
                symmetry=symmetry,
                rigid=rigid,
                minimum_eigenvalue=float(ev[0]),
                dtcrit=critical,
                safe_dt=op.stable_dt,
                dtcrit_error=dt_error,
                energy_drift=drift,
                sum_factorization=True,
            ),
        )


@pytest.mark.parametrize("p", [1, 2, 4, 6])
def test_homogeneous_limit(p):
    outputs = []
    for material in [LOW, layers(-1, 0, 1, LOW, LOW)]:
        data = config(p, material).model_dump(mode="json", by_alias=True)
        data.update(
            time=dict(dt=0.0005, duration=0.03),
            source=dict(
                position=(0.17, -0.27),
                direction=(3, 4),
                wavelet=dict(f0=12, amplitude=2, time_shift=0.01),
            ),
            receivers=[dict(name="above", position=(0.23, 0.37))],
        )
        with Simulation2D(SimulationConfig2D.model_validate(data), MPI.COMM_SELF) as sim:
            result = sim.run()
            op = sim.operators
            outputs.append(
                dict(
                    M=dense(op.M),
                    K=dense(op.K),
                    mass=op.mass.copy(),
                    dt=np.array(op.stable_dt),
                    u=result.displacement,
                    v=result.velocity,
                )
            )
    errors = {k: relative(outputs[1][k], a) for k, a in outputs[0].items()}
    assert max(errors.values()) < 3e-13
    record(f"heterogeneous-homogeneous-{p}", errors)


def test_alignment_guard_and_scope():
    with pytest.raises(ValueError, match="adjust the mesh or layer depths"):
        config(material=layers(-1, 0.1, 1))
    # Roundoff in an aligned depth is accepted; a physical cut is not.
    config(material=layers(-1, 1e-15, 1))
    for updates in [
        dict(boundaries=dict(left="pml")),
        dict(constraints=[dict(side="upper", components=["x"])]),
        dict(material=dict(type="vti", density=2, c11=18, c33=18, c13=9, c55=4.5)),
    ]:
        data = config().model_dump(mode="json", by_alias=True) | updates
        with pytest.raises(ValueError):
            PlaneStrainConfig.model_validate(data)
