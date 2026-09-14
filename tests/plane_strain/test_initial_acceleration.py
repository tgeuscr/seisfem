"""Distinguish interpolated initial data from elliptically compatible data."""

import json

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import configuration
from tests.plane_strain.manufactured import OMEGA, ErrorNorms, analytic_force, fields, initial_data


def ritz_projection(op, elastic_load):
    """Test-only elliptic projection: K_ff R_h U = a(U,phi_f)."""
    free = np.flatnonzero(~op.fixed).astype(PETSc.IntType)
    selection = PETSc.IS().createGeneral(free, comm=op.comm)
    matrix = op.K.createSubMatrix(selection, selection)
    rhs, solution = matrix.createVecRight(), matrix.createVecRight()
    solver = PETSc.KSP().create(op.comm)
    try:
        rhs.array[:] = elastic_load[free]
        solver.setOperators(matrix)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(rhs, solution)
        assert solver.getConvergedReason() > 0
        result = np.zeros(op.n)
        result[free] = solution.array_r
        return result
    finally:
        solver.destroy()
        solution.destroy()
        rhs.destroy()
        matrix.destroy()
        selection.destroy()


@pytest.mark.parametrize("diagonal", ["left", "right", "left_right"])
def test_initial_acceleration_and_compatible_ritz_data(diagonal):
    records = []
    for n in [8, 16, 32, 64]:
        with PlaneStrainOperators(
            configuration(n=n, fixed=True, diagonal=diagonal), MPI.COMM_SELF
        ) as op:
            initial, velocity = initial_data(op)
            force = op.assemble_load(analytic_force(op.mesh), quadrature_degree=12)
            continuum_mass_load = op.assemble_load(2.3 * fields(op.mesh), quadrature_degree=12)
            norm = ErrorNorms(op)
            acceleration = (force - op.apply(initial)) / op.mass
            acceleration[op.fixed] = 0
            interpolation_error = norm(initial, 1)[0]
            initial_acceleration_error = norm(acceleration, -(OMEGA**2))[0]
            # Since f = -rho*omega²*U - div sigma(U), adding back the
            # continuum inertia gives the independently expanded elastic load.
            ritz = ritz_projection(op, force + OMEGA**2 * continuum_mass_load)
            ritz_acceleration = (force - op.apply(ritz)) / op.mass
            ritz_acceleration[op.fixed] = 0
            expected = -(OMEGA**2) * continuum_mass_load / op.mass
            expected[op.fixed] = 0
            np.testing.assert_allclose(ritz_acceleration, expected, rtol=1e-8, atol=2e-10)
            # Verify start/evaluate implements this initialization, including nonzero v0.
            dt = 0.5 * op.stable_dt
            step = op.start(dt, ritz, velocity, force)
            _, _, actual_acceleration, _ = step.evaluate(force)
            np.testing.assert_allclose(actual_acceleration, expected, rtol=1e-8, atol=2e-10)
            records.append(
                [
                    interpolation_error,
                    initial_acceleration_error,
                    norm(ritz, 1)[0],
                    norm(ritz_acceleration, -(OMEGA**2))[0],
                ]
            )
    values = np.array(records)
    rates = np.log2(values[:-1] / values[1:])
    print(
        "MMS initial acceleration",
        json.dumps(
            dict(
                diagonal=diagonal,
                resolutions=[8, 16, 32, 64],
                columns=[
                    "interpolation_L2",
                    "interpolated_acceleration_L2",
                    "ritz_displacement_L2",
                    "ritz_acceleration_L2",
                ],
                errors=values.tolist(),
                rates=rates.tolist(),
            )
        ),
    )
    # The compatible projection supplies a convergent continuum initial acceleration.
    assert np.all((rates[1:, 3] > 1.8) & (rates[1:, 3] < 2.2))
    assert np.all((rates[1:, 2] > 1.8) & (rates[1:, 2] < 2.2))
    # Retain the counterexample as an explicit limitation, not an accuracy assertion.
    if diagonal == "left_right":
        assert values[-1, 1] > 0.8 * values[-2, 1]
        assert values[-1, 1] > values[-1, 3] * 10


def test_alternating_quadratic_stencil_explains_initial_acceleration():
    # An independent hand-assembled quadratic patch exposes the strong-form
    # inconsistency of D^-1 K I_h on alternating node stars, despite weak consistency.
    from tests.plane_strain.helpers import hand_matrices

    for n in [4, 8]:
        with PlaneStrainOperators(configuration(n=n, diagonal="left_right"), MPI.COMM_SELF) as op:
            mass, stiffness = hand_matrices(op)
            x, z = op.coordinates.T
            interpolant = np.column_stack((x**2, np.zeros_like(z))).ravel()
            interior = (x > 1e-10) & (x < 1 - 1e-10) & (z > 1e-10) & (z < 1 - 1e-10)
            strong = (stiffness @ interpolant / mass.sum(axis=1)).reshape(-1, 2)[interior]
            # For U=(x²,0), -div sigma(U)/rho=(-2*(lambda+2*mu)/rho,0).
            exact_x = -2 * (1.7 + 2 * 1.2) / 2.3
            ratios = strong[:, 0] / exact_x
            assert np.any(np.isclose(ratios, 0.75)) and np.any(np.isclose(ratios, 1.5))
            np.testing.assert_allclose(
                np.minimum(abs(ratios - 0.75), abs(ratios - 1.5)), 0, atol=2e-14
            )
            np.testing.assert_allclose(strong[:, 1], 0, atol=2e-13)
