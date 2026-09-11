"""Independent smooth plane-strain solution and analytic forcing (unit square)."""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

OMEGA = 2.7
INITIAL_RATE = 0.37


def time_factor(time):
    return np.cos(OMEGA * time) + INITIAL_RATE / OMEGA * np.sin(OMEGA * time)


def spatial_values(x, z):
    return np.array(
        [np.sin(np.pi * x) * np.sin(2 * np.pi * z), 0.7 * np.sin(2 * np.pi * x) * np.sin(np.pi * z)]
    )


def fields(mesh):
    x, z = ufl.SpatialCoordinate(mesh)
    X = ufl.sin(np.pi * x) * ufl.sin(2 * np.pi * z)
    Z = 0.7 * ufl.sin(2 * np.pi * x) * ufl.sin(np.pi * z)
    return ufl.as_vector([X, Z])


def analytic_force(mesh, rho=2.3, lam=1.7, mu=1.2):
    """Expanded derivatives, not production stress/divergence applied to exact u."""
    x, z = ufl.SpatialCoordinate(mesh)
    X, Z = fields(mesh)
    force_x = ((lam + 2 * mu) * np.pi**2 + mu * (2 * np.pi) ** 2 - rho * OMEGA**2) * X - (
        lam + mu
    ) * 0.7 * 2 * np.pi**2 * ufl.cos(2 * np.pi * x) * ufl.cos(np.pi * z)
    force_z = (mu * (2 * np.pi) ** 2 + (lam + 2 * mu) * np.pi**2 - rho * OMEGA**2) * Z - (
        lam + mu
    ) * 2 * np.pi**2 * ufl.cos(np.pi * x) * ufl.cos(2 * np.pi * z)
    return ufl.as_vector([force_x, force_z])


def initial_data(op):
    x, z = op.coordinates[: op.n // 2].T
    initial = spatial_values(x, z).T.ravel()
    initial[op.fixed] = 0
    return initial, INITIAL_RATE * initial


def errors(op, owned, final):
    """Integrated FE L2 and H1 seminorm, using independent exact UFL fields."""
    op.field.x.array[: op.n] = owned
    op.field.x.scatter_forward()
    difference = op.field - float(time_factor(final)) * fields(op.mesh)
    dx = ufl.dx(metadata={"quadrature_degree": 12})
    quantities = [
        ufl.inner(difference, difference),
        ufl.inner(ufl.grad(difference), ufl.grad(difference)),
    ]
    return tuple(
        np.sqrt(op.comm.allreduce(fem.assemble_scalar(fem.form(q * dx)), op=MPI.SUM))
        for q in quantities
    )
