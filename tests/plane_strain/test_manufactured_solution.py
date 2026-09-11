import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import configuration, dense
from tests.plane_strain.manufactured import (
    INITIAL_RATE,
    OMEGA,
    analytic_force,
    errors,
    fields,
    initial_data,
    time_factor,
)


def test_expanded_force_against_independent_vector_identity():
    with PlaneStrainOperators(configuration(n=3, fixed=True), MPI.COMM_SELF) as op:
        exact = fields(op.mesh)
        # Constant isotropic Navier operator, independent of the production stress helper.
        force = (
            -2.3 * OMEGA**2 * exact
            - 1.2 * ufl.div(ufl.grad(exact))
            - (1.7 + 1.2) * ufl.grad(ufl.div(exact))
        )
        defect = force - analytic_force(op.mesh)
        value = fem.assemble_scalar(
            fem.form(ufl.inner(defect, defect) * ufl.dx(metadata={"quadrature_degree": 12}))
        )
        assert abs(value) < 1e-24
        assert np.linalg.norm(op.assemble_load(analytic_force(op.mesh))) > 1


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
def test_manufactured_spatial_convergence(diagonal):
    measurements = []
    final = 0.31
    # Fixed small timestep keeps temporal error well below interpolation/spatial error.
    dt = 0.000025
    for n in [8, 16, 32, 64, 128]:
        with PlaneStrainOperators(
            configuration(n=n, fixed=True, diagonal=diagonal), MPI.COMM_SELF
        ) as op:
            initial, velocity = initial_data(op)
            load = op.assemble_load(analytic_force(op.mesh))
            step = op.start(dt, initial, velocity, load)
            for i in range(round(final / dt)):
                nxt, _, _, _ = step.evaluate(load * time_factor(i * dt))
                step.advance(nxt)
            measurements.append(errors(op, step.current, final))
            if n == 128:
                # Repeat only the finest case at half dt to quantify temporal contamination.
                half = op.start(dt / 2, initial, velocity, load)
                for i in range(round(2 * final / dt)):
                    nxt, _, _, _ = half.evaluate(load * time_factor(i * dt / 2))
                    half.advance(nxt)
                temporal_change = np.sqrt(np.dot(op.mass, (half.current - step.current) ** 2))
                assert temporal_change < 0.001 * measurements[-1][0]
                print("spatial temporal-contamination check", diagonal, temporal_change)
    values = np.array(measurements)
    rates = np.log2(values[:-1] / values[1:])
    print("MMS spatial", diagonal, "L2/H1 errors", values.tolist(), "rates", rates.tolist())
    # N=8 is retained as a reported coarse-grid diagnostic. Require all three
    # rates from N=16 through 128 to meet the asymptotic P1 gates.
    assert np.all((rates[1:, 0] > 1.8) & (rates[1:, 0] < 2.2))
    assert np.all((rates[1:, 1] > 0.9) & (rates[1:, 1] < 1.1))


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
def test_manufactured_temporal_convergence(diagonal):
    with PlaneStrainOperators(
        configuration(n=12, fixed=True, diagonal=diagonal), MPI.COMM_SELF
    ) as op:
        initial, velocity = initial_data(op)
        load = op.assemble_load(analytic_force(op.mesh))
        free = ~op.fixed
        m = op.mass[free]
        K = dense(op.K)[np.ix_(free, free)]
        eigenvalues, Q = np.linalg.eigh(K / np.sqrt(np.outer(m, m)))
        frequencies = np.sqrt(eigenvalues)
        assert np.min(abs(eigenvalues - OMEGA**2)) > 1
        y0 = Q.T @ (np.sqrt(m) * initial[free])
        forcing = Q.T @ (load[free] / np.sqrt(m))
        particular = forcing / (eigenvalues - OMEGA**2)
        final = 0.31
        # Exact forced semidiscrete ODE solution; no discrete recurrence frequency.
        modal = particular * time_factor(final) + (y0 - particular) * (
            np.cos(frequencies * final) + INITIAL_RATE / frequencies * np.sin(frequencies * final)
        )
        exact = Q @ modal / np.sqrt(m)
        measured = []
        for count in [40, 80, 160, 320]:
            dt = final / count
            step = op.start(dt, initial, velocity, load)
            for i in range(count):
                nxt, _, _, _ = step.evaluate(load * time_factor(i * dt))
                step.advance(nxt)
            measured.append(np.sqrt(np.dot(m, (step.current[free] - exact) ** 2)))
        rates = np.log2(np.array(measured[:-1]) / measured[1:])
        print("MMS temporal", diagonal, "errors", measured, "rates", rates)
        assert np.all((rates > 1.8) & (rates < 2.2))


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
def test_manufactured_temporal_refinement_on_fine_mesh(diagonal):
    # Fine-grid Richardson check complements the exact modal ODE comparison.
    # Fixed spatial error cancels in successive differences; no continuum-order
    # claim is inferred from a temporal curve already at its spatial error floor.
    with PlaneStrainOperators(
        configuration(n=64, fixed=True, diagonal=diagonal), MPI.COMM_SELF
    ) as op:
        initial, velocity = initial_data(op)
        load = op.assemble_load(analytic_force(op.mesh))
        solutions = []
        for count in [400, 800, 1600, 3200]:
            dt = 0.31 / count
            step = op.start(dt, initial, velocity, load)
            for i in range(count):
                nxt, _, _, _ = step.evaluate(load * time_factor(i * dt))
                step.advance(nxt)
            solutions.append(step.current.copy())
        differences = [
            np.sqrt(np.dot(op.mass, (a - b) ** 2))
            for a, b in zip(solutions[:-1], solutions[1:], strict=True)
        ]
        rates = np.log2(np.array(differences[:-1]) / differences[1:])
        print(
            "MMS fine-grid temporal",
            diagonal,
            "successive differences",
            differences,
            "rates",
            rates,
        )
        assert np.all((rates > 1.8) & (rates < 2.2))
