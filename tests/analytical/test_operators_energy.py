import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem import petsc

from seisfem import Simulation
from seisfem.config import SimulationConfig
from seisfem.points import PointMap
from seisfem.timestepping import CentralDifference

pytestmark = pytest.mark.analytical


def standing_config(cells=40, dt=0.0025, duration=0.5, boundary="fixed"):
    return SimulationConfig.model_validate(
        dict(
            domain=dict(upper=1),
            mesh=dict(cells=cells),
            materials=dict(type="homogeneous", material=dict(density=1, vp=2, vs=1)),
            time=dict(dt=dt, duration=duration),
            boundaries=dict(lower=boundary, upper=boundary),
        )
    )


def test_assembled_operators_and_source():
    data = standing_config(cells=4, dt=0.01).model_dump(mode="json", by_alias=True)
    data["materials"] = dict(
        type="layered",
        layers=[
            dict(lower=0, upper=0.5, material=dict(density=1, vp=2, vs=1)),
            dict(lower=0.5, upper=1, material=dict(density=3, vp=2, vs=1)),
        ],
    )
    with Simulation(SimulationConfig.model_validate(data)) as sim:
        op = sim.operators
        z = op.V.tabulate_dof_coordinates()[: op.n, 0]
        expected = np.where(z < 0.5, 0.25, np.where(z > 0.5, 0.75, 0.5))
        expected[(z == 0) | (z == 1)] /= 2
        np.testing.assert_allclose(op.mass, expected, rtol=1e-14)
        assert sum(op.mass) == pytest.approx(2)
        np.testing.assert_allclose(op.apply(np.ones(op.n)), 0, atol=1e-13)
        # Row sums of an independently assembled consistent mass must match.
        u, v = ufl.TrialFunction(op.V), ufl.TestFunction(op.V)
        matrix = petsc.assemble_matrix(fem.form(op.rho * u * v * ufl.dx))
        matrix.assemble()
        rows = matrix.createVecLeft()
        matrix.getRowSum(rows)
        np.testing.assert_allclose(rows.array, op.mass, rtol=1e-14)
        rows.destroy()
        matrix.destroy()
        # A linear field has integral A*(q_z)^2 = 8 in this two-layer model.
        assert z @ op.apply(z) == pytest.approx(8)
        points = PointMap(op.V, [0.137, 0.5], 0.25)
        load = points.unit_load()
        assert sum(load) == pytest.approx(2)
        assert load @ z == pytest.approx(0.637)
        np.testing.assert_allclose(points.evaluate(2 * z + 3), [3.274, 4], atol=1e-13)
        # Compare cached weights to the official Function.eval path.
        op.field.x.array[: op.n] = z**2
        op.field.x.scatter_forward()
        np.testing.assert_allclose(
            points.evaluate(z**2), op.field.eval(points.points, points.cells).ravel(), atol=1e-14
        )


def evolve(cells, dt, final=0.37, temporal_reference=False, velocity_factor=0.0):
    cfg = standing_config(cells, dt, final)
    with Simulation(cfg) as sim:
        op = sim.operators
        z = op.V.tabulate_dof_coordinates()[: op.n, 0]
        initial = np.sin(np.pi * z)
        zeros = np.zeros(op.n)
        stepper = CentralDifference(
            op.mass, op.damping, op.fixed, dt, op.apply, initial, velocity_factor * initial, zeros
        )
        energies = []
        for _ in range(cfg.time.steps):
            nxt, _, _, energy = stepper.evaluate(zeros, energy=True)
            energies.append(energy)
            stepper.advance(nxt)
        omega = 4 * cells * np.sin(np.pi / (2 * cells)) if temporal_reference else 2 * np.pi
        reference = initial * (
            np.cos(omega * final) + velocity_factor / omega * np.sin(omega * final)
        )
        error = np.sqrt(np.dot(op.mass, (stepper.current - reference) ** 2))
        if not temporal_reference:
            # Integrate the actual FE error, including interpolation between nodes.
            op.field.x.array[: op.n] = stepper.current
            op.field.x.scatter_forward()
            coordinate = ufl.SpatialCoordinate(op.mesh)[0]
            exact = ufl.sin(np.pi * coordinate) * (
                np.cos(omega * final) + velocity_factor / omega * np.sin(omega * final)
            )
            error_form = fem.form(
                (op.field - exact) ** 2 * ufl.dx(metadata={"quadrature_degree": 8})
            )
            error = np.sqrt(op.comm.allreduce(fem.assemble_scalar(error_form)))
        drift = np.ptp(energies) / np.mean(energies)
        return error, drift


def test_energy_invariant():
    error, drift = evolve(80, 0.001, final=2, velocity_factor=0.7)
    print("discrete energy relative range", drift)
    assert drift < 2e-11  # Rounding from thousands of updates, not discretization error.
    assert error < 0.002


def test_spatial_convergence():
    errors = [evolve(n, 0.00005)[0] for n in (20, 40, 80)]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("spatial errors/rates", errors, rates)
    assert np.all((rates > 1.95) & (rates < 2.05))


def test_temporal_convergence():
    errors = [
        evolve(20, dt, final=0.4, temporal_reference=True, velocity_factor=0.7)[0]
        for dt in (0.01, 0.005, 0.0025)
    ]
    rates = np.log2(np.array(errors[:-1]) / errors[1:])
    print("temporal errors/rates", errors, rates)
    assert np.all((rates > 1.95) & (rates < 2.05))


def test_impedance_energy_balance():
    cfg = standing_config(80, 0.001, 0.8, boundary="absorbing")
    with Simulation(cfg) as sim:
        op = sim.operators
        z = op.V.tabulate_dof_coordinates()[: op.n, 0]
        initial = np.exp(-(((z - 0.5) / 0.1) ** 2))
        zeros = np.zeros(op.n)
        stepper = CentralDifference(
            op.mass, op.damping, op.fixed, cfg.time.dt, op.apply, initial, zeros, zeros
        )
        previous_energy = None
        defects, energies = [], []
        for _ in range(cfg.time.steps):
            nxt, v, _, energy = stepper.evaluate(zeros, energy=True)
            if previous_energy is not None:
                defects.append(energy - previous_energy + cfg.time.dt * np.dot(op.damping * v, v))
            previous_energy = energy
            energies.append(energy)
            stepper.advance(nxt)
        print(
            "impedance energy residual, remaining fraction",
            max(abs(np.array(defects))),
            energies[-1] / energies[0],
        )
        assert max(abs(np.array(defects))) < 1e-10 * energies[0]
        assert energies[-1] < 1e-4 * energies[0]


def test_free_rigid_translation():
    with Simulation(standing_config(boundary="free")) as sim:
        op = sim.operators
        stepper = CentralDifference(
            op.mass,
            op.damping,
            op.fixed,
            0.0025,
            op.apply,
            np.ones(op.n),
            np.full(op.n, 0.2),
            np.zeros(op.n),
        )
        for _ in range(200):
            nxt, _, _, _ = stepper.evaluate(np.zeros(op.n))
            stepper.advance(nxt)
        np.testing.assert_allclose(stepper.current, 1.1, atol=1e-11)
