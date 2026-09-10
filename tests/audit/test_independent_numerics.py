"""Audit oracles intentionally do not import production reference helpers."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
from pydantic import ValidationError

from seisfem import Simulation, SimulationConfig
from seisfem.config import Isotropic
from seisfem.points import PointMap
from seisfem.timestepping import CentralDifference


def config(cells=8, mode="P", densities=(1.3, 4.7), speeds=(2.1, 3.2), boundary=("free", "free")):
    return SimulationConfig.model_validate(
        dict(
            mode=mode,
            domain=dict(lower=-1, upper=1),
            mesh=dict(cells=cells),
            materials=dict(
                type="layered",
                layers=[
                    dict(
                        lower=-1,
                        upper=0,
                        material=dict(density=densities[0], vp=speeds[0], vs=speeds[0] / 2),
                    ),
                    dict(
                        lower=0,
                        upper=1,
                        material=dict(density=densities[1], vp=speeds[1], vs=speeds[1] / 2),
                    ),
                ],
            ),
            time=dict(dt=0.0001, duration=0.001),
            boundaries=dict(lower=boundary[0], upper=boundary[1]),
        )
    )


def hand_matrices(cells, densities, speeds):
    """Element integrals in ascending coordinate order, no FEM/production helpers."""
    h = 2 / cells
    mass = np.zeros(cells + 1)
    stiffness = np.zeros((cells + 1, cells + 1))
    for cell in range(cells):
        side = int(cell >= cells // 2)
        mass[cell : cell + 2] += densities[side] * h / 2
        stiffness[cell : cell + 2, cell : cell + 2] += (
            densities[side] * speeds[side] ** 2 / h * np.array([[1, -1], [-1, 1]])
        )
    return mass, stiffness


@pytest.mark.parametrize("rho,h", [(2.7, 0.3), (1e-6, 7.1)])
def test_single_element_consistent_and_lumped_mass(rho, h):
    # Public models require >=2 cells; exercise the underlying one-element integral directly.
    msh = mesh.create_interval(MPI.COMM_SELF, 1, [0, h])
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    matrix = petsc.assemble_matrix(fem.form(rho * u * v * ufl.dx))
    matrix.assemble()
    try:
        actual = matrix.getValues([0, 1], [0, 1])
        np.testing.assert_allclose(actual, rho * h / 6 * np.array([[2, 1], [1, 2]]), rtol=2e-14)
        np.testing.assert_allclose(actual.sum(axis=1), [rho * h / 2] * 2, rtol=2e-14)
    finally:
        matrix.destroy()


@pytest.mark.parametrize("mode", ["P", "S"])
@pytest.mark.parametrize(
    "densities,speeds",
    [((1.3, 1.3), (2.1, 2.1)), ((1e-4, 1e4), (2.1, 2.1)), ((1.3, 4.7), (0.03, 31.7))],
)
@pytest.mark.parametrize(
    "cells,boundary", [(2, ("free", "free")), (8, ("fixed", "free")), (16, ("fixed", "fixed"))]
)
def test_hand_operators_and_spectral_bound(mode, densities, speeds, cells, boundary):
    effective = np.array(speeds) / (2 if mode == "S" else 1)
    expected_m, expected_k = hand_matrices(cells, densities, effective)
    with Simulation(config(cells, mode, densities, speeds, boundary), comm=MPI.COMM_SELF) as sim:
        op = sim.operators
        order = np.argsort(op.V.tabulate_dof_coordinates()[: op.n, 0])
        actual_k = op.K.getValues(order.astype(PETSc.IntType), order.astype(PETSc.IntType))
        np.testing.assert_allclose(op.mass[order], expected_m, rtol=3e-14)
        np.testing.assert_allclose(actual_k, expected_k, rtol=3e-14, atol=1e-18)
        assert np.all(op.mass > 0)
        assert sum(op.mass) == pytest.approx(sum(densities), rel=3e-14)
        free = ~op.fixed[order]
        scaled = actual_k[np.ix_(free, free)] / np.sqrt(
            np.outer(expected_m[free], expected_m[free])
        )
        eigenvalues = np.linalg.eigvalsh(scaled)
        true_limit = 2 / np.sqrt(eigenvalues[-1])
        assert op.spectral_dt_bound <= true_limit * (1 + 1e-13)
        assert sim.config.h / max(effective) <= true_limit * (1 + 1e-13)
        # Probe the true stability transition with the highest eigenmode.
        values, vectors = np.linalg.eigh(scaled)
        initial = vectors[:, -1] / np.sqrt(expected_m[free])
        k = actual_k[np.ix_(free, free)]
        for fraction in [0.5, 0.999, 1.001]:
            dt = fraction * true_limit
            step = CentralDifference(
                expected_m[free],
                np.zeros(sum(free)),
                np.zeros(sum(free), bool),
                dt,
                lambda x: k @ x,
                initial,
                np.zeros(sum(free)),
                np.zeros(sum(free)),
            )
            maximum = 0
            for _ in range(300):
                nxt, _, _, _ = step.evaluate(np.zeros(sum(free)))
                step.advance(nxt)
                maximum = max(maximum, np.linalg.norm(step.current) / np.linalg.norm(initial))
            assert maximum < 1.000001 if fraction < 1 else maximum > 1e8


@pytest.mark.parametrize(
    "position",
    [
        -0.75,
        -0.749999999,
        -0.500000001,
        -0.371,
        0,
        np.nextafter(0.0, -1.0),
        np.nextafter(0.0, 1.0),
        0.629,
    ],
)
def test_point_load_each_weight_and_nonlinear_receiver(position):
    with Simulation(config(), comm=MPI.COMM_SELF) as sim:
        op = sim.operators
        z = op.V.tabulate_dof_coordinates()[: op.n, 0]
        order = np.argsort(z)
        nodes = z[order]
        # Hat function evaluation from globally ordered vertices, no selected-cell reuse.
        expected = np.maximum(1 - abs(nodes - position) / 0.25, 0)
        points = PointMap(op.V, [position, position, -1, 1], 0.25)
        load = PointMap(op.V, [position], 0.25).unit_load()
        np.testing.assert_allclose(load[order], expected, atol=5e-15, rtol=0)
        assert load.sum() == pytest.approx(1, abs=3e-15)
        assert load @ z == pytest.approx(position, abs=3e-15)
        values = np.sin(2.3 * z) + z**2
        expected_values = np.interp([position, position, -1, 1], nodes, values[order])
        np.testing.assert_allclose(points.evaluate(values), expected_values, atol=3e-15)
        op.field.x.array[: op.n] = values
        op.field.x.scatter_forward()
        np.testing.assert_allclose(
            points.evaluate(values), op.field.eval(points.points, points.cells).ravel(), atol=3e-15
        )


@pytest.mark.parametrize("mu,lam", [(1e-24, 2), (1, -0.6), (1, -2 / 3 + 1e-12), (1, 0), (1, 1e-14)])
def test_material_energy_and_conversion(mu, lam):
    rho = 2.7
    material = Isotropic.model_validate(dict(density=rho, mu=mu, **{"lambda": lam}))
    assert material.speed("P") == pytest.approx(np.sqrt((lam + 2 * mu) / rho))
    assert material.speed("S") == pytest.approx(np.sqrt(mu / rho))
    recovered = Isotropic(density=rho, vp=np.sqrt((lam + 2 * mu) / rho), vs=np.sqrt(mu / rho))
    np.testing.assert_allclose(recovered.lame, [lam, mu], rtol=3e-14, atol=1e-15)
    # Six independent strain coordinates: three normal, three engineering shear.
    elastic = np.zeros((6, 6))
    elastic[:3, :3] = lam
    elastic[np.arange(3), np.arange(3)] += 2 * mu
    elastic[3:, 3:] = mu * np.eye(3)
    if mu > 1e-12 * abs(lam):
        assert np.linalg.eigvalsh(elastic).min() > 0
    else:
        # Dense eigensolvers lose the tiny deviatoric eigenvalues to cancellation.
        assert mu > 0 and lam + 2 * mu / 3 > 0
    for bad in [-2 * mu / 3 - 1e-12 * max(mu, 1), -mu]:
        with pytest.raises(ValidationError):
            Isotropic.model_validate(dict(density=rho, mu=mu, **{"lambda": bad}))


@pytest.mark.parametrize("density,modulus", [(1e300, 1e-300), (1e-300, 1e300)])
def test_unrepresentable_derived_speed_rejected(density, modulus):
    with pytest.raises(ValidationError, match="speed"):
        Isotropic.model_validate(dict(density=density, mu=modulus, **{"lambda": modulus}))


def test_exact_forced_polynomial_and_final_centered_outputs():
    # q=(1+2t+3t², -2+t-t²) is exactly representable by centered differences.
    mass = np.array([1.7, 2.3])
    damping = np.array([0.6, 1.1])
    K = np.array([[4.0, -1.0], [-1.0, 3.0]])

    def u(t):
        return np.array([1 + 2 * t + 3 * t * t, -2 + t - t * t])

    def v(t):
        return np.array([2 + 6 * t, 1 - 2 * t])

    acceleration = np.array([6.0, -2.0])

    def force(t):
        return mass * acceleration + damping * v(t) + K @ u(t)

    dt = 0.017
    step = CentralDifference(
        mass, damping, np.zeros(2, bool), dt, lambda x: K @ x, u(0), v(0), force(0)
    )
    np.testing.assert_allclose(step.previous, u(-dt), atol=1e-15)
    for n in range(101):
        nxt, velocity, actual_a, _ = step.evaluate(force(n * dt))
        np.testing.assert_allclose(step.current, u(n * dt), atol=2e-12, rtol=0)
        np.testing.assert_allclose(velocity, v(n * dt), atol=4e-12, rtol=0)
        np.testing.assert_allclose(actual_a, acceleration, atol=2e-11, rtol=0)
        step.advance(nxt)


def test_oscillator_physical_energy_is_not_the_invariant():
    dt, omega = 0.15, 3.7
    step = CentralDifference(
        np.array([2.0]),
        np.zeros(1),
        np.zeros(1, bool),
        dt,
        lambda u: 2 * omega**2 * u,
        np.ones(1),
        np.array([0.4]),
        np.zeros(1),
    )
    reported, independent, physical = [], [], []
    for _ in range(200):
        nxt, v, _, energy = step.evaluate(np.zeros(1), energy=True)
        midpoint = (nxt + step.current) / 2
        half_v = (nxt - step.current) / dt
        independent.append(
            float(((1 - dt**2 * omega**2 / 4) * half_v**2 + omega**2 * midpoint**2)[0])
        )
        physical.append(float((v**2 + omega**2 * step.current**2)[0]))
        reported.append(energy)
        step.advance(nxt)
    np.testing.assert_allclose(reported, independent, rtol=3e-15)
    assert np.ptp(independent) / np.mean(independent) < 3e-14
    assert np.ptp(physical) / np.mean(physical) > 0.07
    print(
        "physical/invariant energy ranges",
        np.ptp(physical) / np.mean(physical),
        np.ptp(independent) / np.mean(independent),
    )


def test_forced_damped_energy_work_independent():
    m, c = np.array([1.3, 2.7]), np.array([0.4, 0.7])
    K = np.array([[5.0, -2.0], [-2.0, 3.0]])
    dt = 0.03
    step = CentralDifference(
        m,
        c,
        np.zeros(2, bool),
        dt,
        lambda u: K @ u,
        np.array([0.2, -0.3]),
        np.array([0.7, 0.1]),
        np.array([0.0, 0.2]),
    )

    def invariant(a, b):
        vh, uh = (b - a) / dt, (a + b) / 2
        return 0.5 * vh @ (np.diag(m) - dt**2 * K / 4) @ vh + 0.5 * uh @ K @ uh

    before = invariant(step.previous, step.current)
    for n in range(200):
        force = np.array([np.sin(n * dt), 0.2 * np.cos(2 * n * dt)])
        nxt, v, _, reported = step.evaluate(force, energy=True)
        after = invariant(step.current, nxt)
        assert after - before == pytest.approx(dt * (v @ force - (c * v) @ v), abs=2e-14)
        assert reported == pytest.approx(after, abs=2e-14)
        before = after
        step.advance(nxt)


@pytest.mark.parametrize("cells,mode_number", [(12, 2), (18, 3)])
def test_temporal_order_dense_eigenproblem(cells, mode_number):
    errors = []
    with Simulation(
        config(cells, densities=(1.3, 1.3), speeds=(2.1, 2.1), boundary=("fixed", "fixed")),
        comm=MPI.COMM_SELF,
    ) as sim:
        op = sim.operators
        indices = np.flatnonzero(~op.fixed).astype(PETSc.IntType)
        K = op.K.getValues(indices, indices)
        m = op.mass[indices]
        vals, vecs = np.linalg.eigh(K / np.sqrt(np.outer(m, m)))
        omega = np.sqrt(vals[mode_number - 1])
        assert omega == pytest.approx(
            2 * 2.1 / (2 / cells) * np.sin(mode_number * np.pi / (2 * cells)), rel=2e-14
        )
        initial = vecs[:, mode_number - 1] / np.sqrt(m)
        for steps in [100, 200, 400, 800]:
            dt, final = 0.371 / steps, 0.371
            step = CentralDifference(
                m,
                np.zeros_like(m),
                np.zeros(len(m), bool),
                dt,
                lambda u: K @ u,
                initial,
                0.63 * initial,
                np.zeros_like(m),
            )
            for _ in range(steps):
                nxt, _, _, _ = step.evaluate(np.zeros_like(m))
                step.advance(nxt)
            exact = initial * (np.cos(omega * final) + 0.63 / omega * np.sin(omega * final))
            errors.append(np.sqrt(np.dot(m, (step.current - exact) ** 2)))
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("dense temporal", cells, mode_number, errors, rates)
    assert np.all((rates > 1.95) & (rates < 2.05))


def test_spatial_order_independent_gauss_multimode():
    # Two smooth modes; evaluate the FE field at Gauss points via np.interp.
    errors = []
    dt, final = 0.00001, 0.173
    gauss, weights = np.polynomial.legendre.leggauss(12)
    for cells in [16, 32, 64, 128]:
        with Simulation(
            config(cells, densities=(1.3, 1.3), speeds=(2.1, 2.1), boundary=("fixed", "fixed")),
            comm=MPI.COMM_SELF,
        ) as sim:
            op = sim.operators
            z = op.V.tabulate_dof_coordinates()[: op.n, 0]
            initial = np.sin(np.pi * (z + 1) / 2) + 0.23 * np.sin(3 * np.pi * (z + 1) / 2)
            step = CentralDifference(
                op.mass, op.damping, op.fixed, dt, op.apply, initial, 0.37 * initial, np.zeros(op.n)
            )
            for _ in range(round(final / dt)):
                nxt, _, _, _ = step.evaluate(np.zeros(op.n))
                step.advance(nxt)
            h = 2 / cells
            x = -1 + h * (np.arange(cells)[:, None] + (gauss + 1) / 2)
            exact = np.zeros_like(x)
            for mode, amplitude in [(1, 1), (3, 0.23)]:
                omega = 2.1 * mode * np.pi / 2
                exact += (
                    amplitude
                    * np.sin(mode * np.pi * (x + 1) / 2)
                    * (np.cos(omega * final) + 0.37 / omega * np.sin(omega * final))
                )
            order = np.argsort(z)
            numerical = np.interp(x, z[order], step.current[order])
            errors.append(np.sqrt(np.sum((numerical - exact) ** 2 * weights * h / 2)))
            # Quantify temporal contamination against the exact matrix ODE solution.
            free = np.flatnonzero(~op.fixed).astype(PETSc.IntType)
            m = op.mass[free]
            eigenvalues, eigenvectors = np.linalg.eigh(
                op.K.getValues(free, free) / np.sqrt(np.outer(m, m))
            )
            frequencies = np.sqrt(eigenvalues)
            modal = eigenvectors.T @ (np.sqrt(m) * initial[free])
            semidiscrete = np.zeros(op.n)
            semidiscrete[free] = (
                eigenvectors
                @ (
                    modal
                    * (
                        np.cos(frequencies * final)
                        + 0.37 / frequencies * np.sin(frequencies * final)
                    )
                )
                / np.sqrt(m)
            )
            temporal_error = np.sqrt(np.dot(op.mass, (step.current - semidiscrete) ** 2))
            assert temporal_error < 1e-4 * errors[-1]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("independent spatial", errors, rates)
    assert np.all((rates > 1.95) & (rates < 2.05))


def test_public_source_timing_all_quantities_and_final_state():
    # Two unit cells, rho=3, cP=2: the only free DOF has D=3 and K=24.
    data = config(cells=2, densities=(3, 3), speeds=(2, 2), boundary=("fixed", "fixed")).model_dump(
        mode="json", by_alias=True
    )
    data.update(
        time=dict(dt=0.02, duration=0.2),
        source=dict(position=0, amplitude=-7, frequency=3, time_shift=0),
        receivers=[
            dict(name="left", position=-1),
            dict(name="center", position=0),
            dict(name="quarter", position=0.25),
            dict(name="right", position=1),
        ],
    )
    result = Simulation(SimulationConfig.model_validate(data), comm=MPI.COMM_SELF).run()
    dt = 0.02
    phase = np.pi * 3 * result.time
    force = -7 * (1 - 2 * phase**2) * np.exp(-(phase**2))
    q = result.displacement[:, 1, 0]
    v = result.velocity[:, 1, 0]
    a = result.acceleration[:, 1, 0]
    assert q[0] == 0 and abs(v[0]) < 1e-15
    assert a[0] == pytest.approx(-7 / 3, abs=1e-14)
    assert q[1] == pytest.approx(-7 / 3 * dt**2 / 2, abs=1e-16)
    # Includes the final acceleration: independently balance the nodal equation.
    np.testing.assert_allclose(3 * a + 24 * q, force, atol=5e-14, rtol=0)
    np.testing.assert_allclose(v[1:-1], (q[2:] - q[:-2]) / (2 * dt), atol=2e-15, rtol=0)
    np.testing.assert_allclose(a[1:-1], (q[2:] - 2 * q[1:-1] + q[:-2]) / dt**2, atol=2e-14, rtol=0)
    # Eliminate the auxiliary state from the last centered velocity.
    assert v[-1] == pytest.approx((q[-1] - q[-2]) / dt + dt * a[-1] / 2, abs=2e-15)
    for quantity in ["displacement", "velocity", "acceleration"]:
        values = getattr(result, quantity)
        np.testing.assert_array_equal(values[:, [0, 3]], 0)
        np.testing.assert_allclose(values[:, 2, 0], 0.75 * values[:, 1, 0], atol=2e-15, rtol=0)


def test_sufficient_bound_is_not_true_fixed_mode_limit():
    with Simulation(
        config(cells=2, densities=(3, 3), speeds=(2, 2), boundary=("fixed", "fixed")),
        comm=MPI.COMM_SELF,
    ) as sim:
        bound = sim.operators.spectral_dt_bound
        assert bound == pytest.approx(0.5)
        true_limit = 2 / np.sqrt(8)
        assert true_limit > 1.4 * bound
        for dt in [0.5 * bound, 0.999 * bound, 1.001 * bound]:
            step = CentralDifference(
                np.array([3.0]),
                np.zeros(1),
                np.zeros(1, bool),
                dt,
                lambda u: 24 * u,
                np.ones(1),
                np.zeros(1),
                np.zeros(1),
            )
            for _ in range(300):
                nxt, _, _, _ = step.evaluate(np.zeros(1))
                assert abs(nxt[0]) <= 1 + 1e-13
                step.advance(nxt)
        print("sufficient vs true fixed one-mode limits", bound, true_limit)


def test_damped_oscillator_against_continuous_solution():
    mass, damping, stiffness = 1.3, 0.7, 4.1
    gamma = damping / (2 * mass)
    frequency = np.sqrt(stiffness / mass - gamma**2)
    initial, velocity, final = 0.4, -0.6, 0.4
    exact = np.exp(-gamma * final) * (
        initial * np.cos(frequency * final)
        + (velocity + gamma * initial) / frequency * np.sin(frequency * final)
    )
    errors = []
    for dt in [0.02, 0.01, 0.005, 0.0025]:
        step = CentralDifference(
            np.array([mass]),
            np.array([damping]),
            np.zeros(1, bool),
            dt,
            lambda u: stiffness * u,
            np.array([initial]),
            np.array([velocity]),
            np.zeros(1),
        )
        for _ in range(round(final / dt)):
            nxt, _, _, _ = step.evaluate(np.zeros(1))
            step.advance(nxt)
        errors.append(abs(step.current[0] - exact))
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("damped oscillator", errors, rates)
    assert np.all((rates > 1.95) & (rates < 2.05))
