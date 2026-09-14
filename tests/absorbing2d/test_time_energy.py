import numpy as np
import pytest
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from seisfem.timestepping import CentralDifference
from tests.absorbing2d.test_operators import config


def exact(t):
    e = np.exp(-0.3 * t)
    a = 1.7 * t
    b = 0.4 * t
    q = e * np.cos(a) + 0.2 * np.sin(b)
    v = e * (-0.3 * np.cos(a) - 1.7 * np.sin(a)) + 0.08 * np.cos(b)
    acc = e * ((0.3**2 - 1.7**2) * np.cos(a) + 2 * 0.3 * 1.7 * np.sin(a)) - 0.032 * np.sin(b)
    return q, v, acc


@pytest.mark.parametrize("damping", [0.0, 0.8, 8.0, 80.0])
def test_forced_damped_temporal_order_and_startup(damping):
    mass, stiffness, final = 1.7, 3.2, 0.8

    def force(t):
        q, v, a = exact(t)
        return np.array([mass * a + damping * v + stiffness * q])

    errors = []
    for dt in [0.04, 0.02, 0.01, 0.005]:
        q0, v0, a0 = exact(0)
        step = CentralDifference(
            np.array([mass]),
            np.array([damping]),
            np.zeros(1, bool),
            dt,
            lambda u: stiffness * u,
            np.array([q0]),
            np.array([v0]),
            force(0),
        )
        assert step.previous[0] == pytest.approx(q0 - dt * v0 + 0.5 * dt**2 * a0, abs=2e-16)
        nxt, v, acc, _ = step.evaluate(force(0))
        assert nxt[0] == pytest.approx(q0 + dt * v0 + 0.5 * dt**2 * a0, abs=5e-16)
        assert v[0] == pytest.approx(v0, abs=4 * np.finfo(float).eps / dt)
        assert acc[0] == pytest.approx(a0, abs=2e-11)
        for n in range(round(final / dt)):
            nxt, _, _, _ = step.evaluate(force(n * dt))
            step.advance(nxt)
        _, v, _, _ = step.evaluate(force(final))
        errors.append([abs(step.current[0] - exact(final)[0]), abs(v[0] - exact(final)[1])])
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("temporal damping", damping, "errors", errors, "rates", rates)
    assert np.all((rates > 1.8) & (rates < 2.2))


def test_scalar_jury_stability_and_large_damping():
    # (1+alpha)r² + (beta-2)r + 1-alpha = 0.
    # Jury: |1-alpha|<=1+alpha, p(1)=beta>=0, p(-1)=4-beta>=0.
    for alpha in [0, 1e-3, 0.1, 1, 10, 1000]:
        for beta in [0.1, 1, 3.99, 4.01, 5]:
            roots = np.roots([1 + alpha, beta - 2, 1 - alpha])
            if beta < 4:
                assert np.max(abs(roots)) <= 1 + 2e-15
            else:
                assert np.max(abs(roots)) > 1
    # A large c*dt/m remains bounded without an additional damping CFL.
    mass, dt, stiffness = 1.0, 1.0, 3.0
    for damping in [0.0, 2.0, 2000.0]:
        step = CentralDifference(
            np.array([mass]),
            np.array([damping]),
            np.zeros(1, bool),
            dt,
            lambda u: stiffness * u,
            np.array([1.0]),
            np.zeros(1),
            np.zeros(1),
        )
        peak = 0.0
        for _ in range(2000):
            nxt, _, _, _ = step.evaluate(np.zeros(1))
            peak = max(peak, abs(nxt[0]))
            step.advance(nxt)
        assert peak <= 1.01


def test_vector_damping_exact_discrete_energy_work_balance():
    cfg = config(("right", "lower"), constraints=[dict(side="left")])
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        rng = np.random.default_rng(512)
        dt = 0.6 * op.stable_dt
        step = op.start(dt, u0=0.01 * rng.normal(size=op.n), v0=0.02 * rng.normal(size=op.n))
        half_v = (step.current - step.previous) / dt
        energy = 0.5 * np.dot(op.mass * half_v, half_v) + 0.5 * np.dot(
            step.current, op.apply(step.previous)
        )
        initial = energy
        work = 0.0
        residuals = []
        for _ in range(600):
            nxt, v, _, new_energy = step.evaluate(np.zeros(op.n), energy=True)
            loss = dt * np.dot(op.damping * v, v)
            assert loss >= 0
            residuals.append(new_energy - energy + loss)
            assert new_energy <= energy + 2e-15 * initial
            work += loss
            energy = new_energy
            step.advance(nxt)
        print(
            "energy max balance/initial",
            max(abs(np.array(residuals))) / initial,
            "cumulative residual/initial",
            abs(energy + work - initial) / initial,
            "remaining",
            energy / initial,
        )
        assert max(abs(np.array(residuals))) / initial < 2e-14
        assert abs(energy + work - initial) / initial < 2e-13
        assert energy / initial < 0.05
