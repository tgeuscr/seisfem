import basix
import numpy as np
import pytest
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from seisfem.timestepping import CentralDifference
from tests.plane_strain.helpers import dense

from .helpers import config, record
from .reference import gll, matrices


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
def test_collocation_and_independent_element(p):
    with PlaneStrainOperators(config(p), MPI.COMM_SELF) as op:
        element = op.V.element.basix_element
        assert element.has_tensor_product_factorisation
        points, weights = basix.make_quadrature(
            basix.CellType.quadrilateral, 2 * p - 1, basix.QuadratureType.gll
        )
        assert len(points) == (p + 1) ** 2
        values = element.tabulate(0, points)[0, :, :, 0]
        np.testing.assert_allclose(
            np.sort(values, axis=1), np.sort(np.eye(len(points)), axis=1), atol=2e-13
        )
        nodes, w = gll(p)
        np.testing.assert_allclose(np.sort(weights), np.sort(np.outer(w, w).ravel()), atol=1e-14)
        assert op.n == 2 * (2 * p + 1) * (3 * p + 1)
        M, K = dense(op.M), dense(op.K)
        hand_M, hand_K = np.zeros_like(M), np.zeros_like(K)
        for cell in range(op.mesh.topology.index_map(2).size_local):
            blocks = op.V.dofmap.cell_dofs(cell)
            a, b = matrices(p, op.coordinates[blocks])
            ids = (2 * blocks[:, None] + np.arange(2)).ravel()
            hand_M[np.ix_(ids, ids)] += a
            hand_K[np.ix_(ids, ids)] += b
        np.testing.assert_allclose(M, hand_M, rtol=3e-11, atol=3e-13)
        np.testing.assert_allclose(K, hand_K, rtol=3e-10, atol=3e-10)
        np.testing.assert_array_equal(op.mass, np.diag(M))
        off = np.max(abs(M - np.diag(np.diag(M))))
        assert off < 1e-13 * np.max(op.mass)
        np.testing.assert_allclose(op.mass.reshape(-1, 2).sum(axis=0), 3.0, rtol=3e-14)
        scale = np.max(abs(K))
        symmetry = np.max(abs(K - K.T)) / scale
        rigid = []
        for d in [(1, 0), (0, 1)]:
            u = np.tile(d, op.n // 2)
            rigid.append(float(np.max(abs(K @ u)) / scale))
        ev = np.linalg.eigvalsh(K)
        assert symmetry < 2e-14
        assert max(rigid) < 3e-13
        assert ev.min() > -1e-12 * ev.max()
        record(
            f"operators-{p}",
            dict(
                dofs=op.n,
                **op.sem_metadata,
                mass_total=op.mass.reshape(-1, 2).sum(axis=0).tolist(),
                symmetry=symmetry,
                rigid_residual=max(rigid),
                min_energy_eigenvalue=float(ev.min()),
                max_energy_eigenvalue=float(ev.max()),
                reference_K_error=float(np.max(abs(K - hand_K)) / scale),
            ),
        )


def test_q1_is_corner_quadrature_not_exact_gauss_q1():
    with PlaneStrainOperators(config(1, cells=(1, 1)), MPI.COMM_SELF) as op:
        blocks = op.V.dofmap.cell_dofs(0)
        ids = (2 * blocks[:, None] + np.arange(2)).ravel()
        a, k = matrices(1, op.coordinates[blocks])
        _, gauss = matrices(1, op.coordinates[blocks], gaussian=True)
        np.testing.assert_allclose(dense(op.K)[np.ix_(ids, ids)], k, atol=2e-14)
        assert np.linalg.norm(k - gauss) / np.linalg.norm(k) > 0.1
        assert np.count_nonzero(abs(a) > 1e-14) == op.n


@pytest.mark.parametrize("p", [1, 2, 4, 6])
def test_critical_eigenvalue_stability_and_energy(p):
    with PlaneStrainOperators(config(p, cells=(2, 2), extent=(1, 1)), MPI.COMM_SELF) as op:
        K = dense(op.K)
        m = op.mass
        ev, Q = np.linalg.eigh(K / np.sqrt(np.outer(m, m)))
        critical = 2 / np.sqrt(ev[-1])
        diagnostic = op.spectral_diagnostic()
        assert diagnostic.critical_dt == pytest.approx(critical, rel=3e-11)
        assert op.stable_dt <= critical
        mode = Q[:, -1] / np.sqrt(m)
        maxima = []
        for fraction in [0.95, 1.01]:
            step = CentralDifference(
                m,
                op.damping,
                op.fixed,
                fraction * critical,
                op.apply,
                mode,
                np.zeros(op.n),
                np.zeros(op.n),
            )
            maximum = 0
            for _ in range(120):
                nxt, *_ = step.evaluate(np.zeros(op.n))
                maximum = max(maximum, float(np.sqrt(m @ nxt**2)))
                step.advance(nxt)
            maxima.append(maximum)
        assert maxima[0] < 1.000001
        assert maxima[1] > 1e10
        rng = np.random.default_rng(72)
        dt = 0.5 * op.stable_dt
        step = op.start(dt, u0=rng.normal(size=op.n), v0=rng.normal(size=op.n))
        energy = []
        corrected = []
        physical = []
        for _ in range(200):
            nxt, v, _, e = step.evaluate(np.zeros(op.n), energy=True)
            ku = op.apply(step.current).copy()
            physical.append(float(0.5 * (m * v) @ v + 0.5 * step.current @ ku))
            corrected.append(float(physical[-1] - dt**2 / 8 * np.sum(ku**2 / m)))
            energy.append(e)
            step.advance(nxt)
        drift = float(np.ptp(energy) / np.mean(energy))
        assert drift < 3e-13
        np.testing.assert_allclose(corrected, energy, rtol=4e-13)
        record(
            f"stability-{p}",
            dict(
                degree=p,
                dofs=op.n,
                h=0.5,
                dtcrit=critical,
                safe_bound=op.stable_dt,
                spectral_residual=diagnostic.relative_residual,
                stable_max=maxima[0],
                unstable_max=maxima[1],
                conserved_energy_drift=drift,
                physical_integer_energy_variation=float(np.ptp(physical) / np.mean(physical)),
            ),
        )
