import numpy as np
import pytest
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators
from seisfem.timestepping import CentralDifference
from tests.plane_strain.helpers import configuration, dense, hand_matrices


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
@pytest.mark.parametrize("fixed,lam", [(False, 1.7), (True, 1.7), (True, -0.7)])
def test_spectral_bound_and_sparse_eigenvalue(diagonal, fixed, lam):
    with PlaneStrainOperators(
        configuration(n=4, diagonal=diagonal, fixed=fixed, lam=lam), MPI.COMM_SELF
    ) as op:
        M, K = hand_matrices(op)
        free = ~op.fixed
        m = M.sum(axis=1)[free]
        B = K[np.ix_(free, free)] / np.sqrt(np.outer(m, m))
        expected = np.linalg.eigvalsh(B)[-1]
        report = op.spectral_diagnostic()
        assert report.lambda_max == pytest.approx(expected, rel=3e-11)
        assert report.relative_residual < 1e-9
        assert report.lambda_upper_bound >= expected * (1 - 1e-14)
        assert report.lambda_upper_bound == pytest.approx(np.max(np.sum(abs(B), axis=1)), rel=3e-14)
        assert op.stable_dt <= 2 / np.sqrt(expected) * (1 + 1e-14)
        print(
            "spectrum",
            diagonal,
            fixed,
            lam,
            report.lambda_max,
            report.lambda_upper_bound,
            op.stable_dt,
            report.critical_dt,
        )
        # No artificial constraints or scaling may be left on the original matrix.
        np.testing.assert_allclose(dense(op.K), K, atol=3e-14)


@pytest.mark.parametrize("diagonal", ["left", "right"])
def test_highest_mode_stability_transition_and_integer_energy(diagonal):
    with PlaneStrainOperators(
        configuration(n=4, fixed=True, diagonal=diagonal), MPI.COMM_SELF
    ) as op:
        free = ~op.fixed
        K = dense(op.K)[np.ix_(free, free)]
        m = op.mass[free]
        values, vectors = np.linalg.eigh(K / np.sqrt(np.outer(m, m)))
        mode = vectors[:, -1] / np.sqrt(m)
        critical = 2 / np.sqrt(values[-1])
        maxima = []
        for fraction in [0.5, 0.999, 1.001]:
            dt = fraction * critical
            initial = np.zeros(op.n)
            initial[free] = mode
            step = CentralDifference(
                op.mass,
                np.zeros(op.n),
                op.fixed,
                dt,
                op.apply,
                initial,
                np.zeros(op.n),
                np.zeros(op.n),
            )
            maximum = 0.0
            for _ in range(300):
                nxt, _, _, _ = step.evaluate(np.zeros(op.n))
                maximum = max(maximum, np.sqrt(np.dot(op.mass, nxt**2)))
                step.advance(nxt)
            maxima.append(maximum)
        assert max(maxima[:2]) < 1.000001
        assert maxima[-1] > 1e8
        print("stability fractions .5/.999/1.001", diagonal, maxima)
        # Equality is marginal, with a double root -1: nonzero mode velocity grows.
        initial = np.zeros(op.n)
        velocity = np.zeros(op.n)
        velocity[free] = mode
        step = CentralDifference(
            op.mass, np.zeros(op.n), op.fixed, critical, op.apply, initial, velocity, np.zeros(op.n)
        )
        for _ in range(100):
            nxt, _, _, _ = step.evaluate(np.zeros(op.n))
            step.advance(nxt)
        assert np.sqrt(np.dot(op.mass, step.current**2)) == pytest.approx(100 * critical, rel=1e-9)


def test_start_constraints_energy_and_rigid_motion():
    with PlaneStrainOperators(configuration(n=4, fixed=True), MPI.COMM_SELF) as op:
        dt = 0.7 * op.stable_dt
        rng = np.random.default_rng(21)
        step = op.start(dt, rng.normal(size=op.n), rng.normal(size=op.n))
        np.testing.assert_array_equal(step.current[op.fixed], 0)
        energies = []
        for _ in range(300):
            nxt, _, _, energy = step.evaluate(np.zeros(op.n), energy=True)
            np.testing.assert_array_equal(nxt[op.fixed], 0)
            vh = (nxt - step.current) / dt
            um = (nxt + step.current) / 2
            independent = (
                0.5 * np.dot(op.mass * vh, vh)
                - dt**2 / 8 * vh @ op.apply(vh)
                + 0.5 * um @ op.apply(um)
            )
            assert energy == pytest.approx(independent, rel=2e-14)
            energies.append(energy)
            step.advance(nxt)
        assert np.ptp(energies) / np.mean(energies) < 2e-13
        with pytest.raises(ValueError, match="spectral bound"):
            op.start(op.stable_dt)
        with pytest.raises(ValueError, match="shape"):
            op.start(dt, np.zeros(3))
    with PlaneStrainOperators(configuration(n=3), MPI.COMM_SELF) as op:
        u0 = np.tile([0.3, -0.2], op.n // 2)
        v0 = np.tile([0.7, 0.4], op.n // 2)
        dt = 0.5 * op.stable_dt
        step = op.start(dt, u0, v0)
        for _ in range(200):
            nxt, _, _, _ = step.evaluate(np.zeros(op.n))
            step.advance(nxt)
        np.testing.assert_allclose(step.current, u0 + 200 * dt * v0, atol=3e-12)


def test_fully_constrained_and_two_free_components():
    for n in [1, 2]:
        data = configuration(n=n, fixed=True).model_dump(mode="json", by_alias=True)
        with PlaneStrainOperators(PlaneStrainConfig.model_validate(data), MPI.COMM_SELF) as op:
            report = op.spectral_diagnostic()
            assert report.lambda_max >= 0
            if n == 1:
                assert report.critical_dt == float("inf")
                step = op.start(0.1)
                nxt, _, _, _ = step.evaluate(np.ones(op.n))
                np.testing.assert_array_equal(nxt, 0)
