import json

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
    ErrorNorms,
    analytic_force,
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
    # Integrate over a fixed physical interval, resolving mesh-scale transients.
    # The four observation times deliberately include the review counterexample.
    observations = [0.13, 0.2, 0.31, 0.37]
    sample_dt, final, dt = 0.00125, 0.4, 0.000025
    times = np.linspace(0, final, round(final / sample_dt) + 1)
    indices = [round(t / sample_dt) for t in observations]
    resolutions = [8, 16, 32, 64, 128]
    measurements, integrated, quadrature_changes, h1_series = [], [], [], []
    contamination = None
    for n in resolutions:
        with PlaneStrainOperators(
            configuration(n=n, fixed=True, diagonal=diagonal), MPI.COMM_SELF
        ) as op:
            initial, velocity = initial_data(op)
            load = op.assemble_load(analytic_force(op.mesh))
            norm = ErrorNorms(op)
            step = op.start(dt, initial, velocity, load)
            samples, states = [], []
            stride = round(sample_dt / dt)
            for i in range(round(final / dt) + 1):
                if i % stride == 0:
                    samples.append(norm(step.current, time_factor(i * dt)))
                    if n == resolutions[-1]:
                        states.append(step.current.copy())
                if i < round(final / dt):
                    nxt, _, _, _ = step.evaluate(load * time_factor(i * dt))
                    step.advance(nxt)
            samples = np.array(samples)
            h1_series.append(samples[:, 1].tolist())
            measurements.append(samples[indices])
            rms = np.sqrt(np.trapezoid(samples**2, times, axis=0) / final)
            integrated.append(rms)
            coarser = np.sqrt(np.trapezoid(samples[::2] ** 2, times[::2], axis=0) / final)
            quadrature_changes.append(abs(coarser / rms - 1))
            assert np.all(abs(coarser / rms - 1) < 0.001)
            if n == resolutions[-1]:
                # Compare FE *field differences*, in both L2 and H1, at half dt.
                half = op.start(dt / 2, initial, velocity, load)
                changes = []
                for i in range(round(2 * final / dt) + 1):
                    if i % (2 * stride) == 0:
                        changes.append(norm(half.current - states[i // (2 * stride)], 0))
                    if i < round(2 * final / dt):
                        nxt, _, _, _ = half.evaluate(load * time_factor(i * dt / 2))
                        half.advance(nxt)
                changes = np.array(changes)
                relative_points = changes[indices] / samples[indices]
                relative_rms = np.sqrt(np.trapezoid(changes**2, times, axis=0) / final) / rms
                assert np.all(relative_points < 0.001)
                assert np.all(relative_rms < 0.001)
                contamination = dict(
                    point_relative=relative_points.tolist(), rms_relative=relative_rms.tolist()
                )
    values, integrated = np.array(measurements), np.array(integrated)
    rates = np.log2(values[:-1] / values[1:])
    rms_rates = np.log2(integrated[:-1] / integrated[1:])
    h = 1 / np.array(resolutions)
    fitted = np.polyfit(np.log(h[1:]), np.log(integrated[1:]), 1)[0]
    point_fits = np.array(
        [np.polyfit(np.log(h[1:]), np.log(values[1:, j]), 1)[0] for j in range(len(indices))]
    )
    # L2 is still checked at t=.31 using the original three asymptotic gates.
    assert np.all((rates[1:, 2, 0] > 1.8) & (rates[1:, 2, 0] < 2.2))
    # H1 evidence is an integrated fit plus a bounded pointwise E_h/h envelope.
    # No individual pointwise pairwise H1 slope is required to be near one.
    assert 0.9 < fitted[1] < 1.1
    envelope = values[1:, :, 1] / h[1:, None]
    assert np.all(envelope.max(axis=0) / envelope.min(axis=0) < 1.5)
    print(
        "MMS spatial ensemble",
        json.dumps(
            dict(
                diagonal=diagonal,
                resolutions=resolutions,
                times=observations,
                integration_times=times.tolist(),
                h1_time_series=h1_series,
                point_errors=values.tolist(),
                point_rates=rates.tolist(),
                point_fits=point_fits.tolist(),
                rms_errors=integrated.tolist(),
                rms_rates=rms_rates.tolist(),
                rms_fit=fitted.tolist(),
                quadrature_relative_changes=np.array(quadrature_changes).tolist(),
                contamination=contamination,
            )
        ),
    )


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
