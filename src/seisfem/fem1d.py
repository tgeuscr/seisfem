"""DOLFINx 0.11 interval-P1 elastic operators, assembled once per experiment."""

from __future__ import annotations

import dolfinx
import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from .config import SimulationConfig
from .materials import properties_at


class IntervalOperators:
    """Own distributed spatial state; mass/damping arrays contain owned DOFs only.

    `K` is the unmodified symmetric stiffness. Homogeneous fixed constraints are
    enforced by time-step projection, not by inserting artificial eigenvalues.
    Call close collectively to destroy explicitly created PETSc resources.
    """

    def __init__(self, config: SimulationConfig, comm):
        if not dolfinx.__version__.startswith("0.11."):
            raise RuntimeError(f"DOLFINx 0.11 required, found {dolfinx.__version__}")
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("This verified runtime requires real double PETSc scalars")
        self.comm = comm
        self.closed = False
        self.mesh = mesh.create_interval(
            comm, config.mesh.cells, [config.domain.lower, config.domain.upper]
        )
        self.V = fem.functionspace(self.mesh, (config.fem.family, config.fem.degree))
        self.n = self.V.dofmap.index_map.size_local
        dg = fem.functionspace(self.mesh, ("DG", 0))
        z = dg.tabulate_dof_coordinates()[:, 0]
        self.rho, self.lam, self.mu = (
            fem.Function(dg, name=name) for name in ("density", "lambda", "mu")
        )
        for field, values in zip(
            (self.rho, self.lam, self.mu), properties_at(config, z), strict=True
        ):
            field.x.array[:] = values
            field.x.scatter_forward()
        self.modulus = fem.Function(dg, name="propagation_modulus")
        self.modulus.x.array[:] = (
            self.lam.x.array + 2 * self.mu.x.array if config.mode == "P" else self.mu.x.array
        )
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.mass_form = fem.form(self.rho * v * ufl.dx)
        mass = fem.assemble_vector(self.mass_form)
        mass.scatter_reverse(la.InsertMode.add)
        self.mass = mass.array[: self.n].copy()
        if not comm.allreduce(bool(np.all(self.mass > 0)), op=MPI.LAND):
            raise RuntimeError("Nonpositive lumped mass")
        self.stiffness_form = fem.form(self.modulus * u.dx(0) * v.dx(0) * ufl.dx)
        self.K = petsc.assemble_matrix(self.stiffness_form)
        self.K.assemble()
        self._work = self.K.createVecLeft()
        self.field = fem.Function(self.V, name="displacement")
        self.damping = np.zeros(self.n)
        self.fixed = np.zeros(self.n, dtype=bool)
        coords = self.V.tabulate_dof_coordinates()[: self.n, 0]
        # Geometric endpoint tolerance is scaled to h, not absolute elevation.
        for endpoint, kind, material in (
            (config.domain.lower, config.boundaries.lower, config.material_values[0]),
            (config.domain.upper, config.boundaries.upper, config.material_values[-1]),
        ):
            mask = np.abs(coords - endpoint) < config.h * 1e-8
            if kind == "fixed":
                self.fixed |= mask
            elif kind == "absorbing":
                self.damping[mask] = material.density * material.speed(config.mode)
        self.K.getDiagonal(self._work)
        local_ratio = np.max(self._work.array / self.mass, initial=0)
        ratio = comm.allreduce(float(local_ratio), op=MPI.MAX)
        self.spectral_dt_bound = np.sqrt(2 / ratio)
        if config.time.dt > config.time.safety * self.spectral_dt_bound * (1 + 1e-10):
            self.close()
            raise ValueError("Timestep violates assembled Gershgorin bound")

    def apply(self, owned: np.ndarray) -> np.ndarray:
        """Apply K collectively. Returned owned view is valid until the next call."""
        self.field.x.array[: self.n] = owned
        self.field.x.scatter_forward()
        self.K.mult(self.field.x.petsc_vec, self._work)
        return self._work.array_r

    def close(self) -> None:
        """Release explicitly owned PETSc objects collectively; safe to repeat."""
        if not self.closed:
            self._work.destroy()
            self.K.destroy()
            self.closed = True
