"""Homogeneous plane-strain vector P1 operators on affine rectangle triangles."""

import dolfinx
import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from .config2d import PlaneStrainConfig


def strain(u):
    """In-plane small strain; epsilon_yy is constrained to zero."""
    return ufl.sym(ufl.grad(u))


def stress(u, lam, mu):
    """In-plane block of 3D isotropic stress, with ordinary 3D Lamé parameters."""
    epsilon = strain(u)
    return lam * ufl.tr(epsilon) * ufl.Identity(2) + 2 * mu * epsilon


class PlaneStrainOperators:
    """Collective operator owner; scalar storage is [ux0,uz0,ux1,uz1,...].

    M and K are unmodified consistent matrices. `mass` contains positive lumped
    entries for owned scalar DOFs, including constrained ones. `fixed` identifies
    zero-displacement components for time-step projection; no artificial boundary
    diagonals are inserted. Geometry coordinates are per two-component node.
    """

    def __init__(self, config: PlaneStrainConfig, comm=MPI.COMM_WORLD):
        self.comm, self.config = comm, config
        self.M = self.K = self._work = None
        self.closed = False
        signatures = comm.allgather(config.model_dump_json(by_alias=True))
        if len(set(signatures)) != 1:
            raise ValueError("All ranks must use identical plane-strain configuration")
        if not dolfinx.__version__.startswith("0.11."):
            raise RuntimeError("Plane-strain kernel requires DOLFINx 0.11")
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("Plane-strain kernel requires real-double PETSc")
        try:
            self._assemble()
        except Exception:
            self.close()
            raise

    def _assemble(self):
        cfg = self.config
        domain = cfg.domain
        self.mesh = mesh.create_rectangle(
            self.comm,
            [domain.lower, domain.upper],
            domain.cells,
            cell_type=mesh.CellType.triangle,
            diagonal=getattr(mesh.DiagonalType, domain.diagonal),
        )
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))
        self.index_map = self.V.dofmap.index_map
        if self.V.dofmap.index_map_bs != 2 or self.V.dofmap.bs != 2:
            raise RuntimeError("Expected blocked P1 vector space with two components")
        self.n = 2 * self.index_map.size_local
        self.coordinates = self.V.tabulate_dof_coordinates()[:, :2].copy()
        self.field = fem.Function(self.V, name="displacement")
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        rho = cfg.material.density
        lam, mu = cfg.material.lame
        self.M = petsc.assemble_matrix(fem.form(rho * ufl.inner(u, v) * ufl.dx))
        self.M.assemble()
        self.K = petsc.assemble_matrix(fem.form(ufl.inner(strain(v), stress(u, lam, mu)) * ufl.dx))
        self.K.assemble()
        lumped = fem.assemble_vector(
            fem.form(rho * ufl.inner(ufl.as_vector((1.0, 1.0)), v) * ufl.dx)
        )
        lumped.scatter_reverse(la.InsertMode.add)
        self.mass = lumped.array[: self.n].copy()
        if not self.comm.allreduce(
            bool(np.all(np.isfinite(self.mass) & (self.mass > 0))), op=MPI.LAND
        ):
            raise RuntimeError("Lumped mass must be finite and positive")
        self.fixed = np.zeros(self.n, dtype=bool)
        sides = {
            "left": (0, domain.lower[0]),
            "right": (0, domain.upper[0]),
            "lower": (1, domain.lower[1]),
            "upper": (1, domain.upper[1]),
        }
        for constraint in cfg.constraints:
            axis, value = sides[constraint.side]
            tolerance = (domain.upper[axis] - domain.lower[axis]) / domain.cells[axis] * 1e-8
            facets = mesh.locate_entities_boundary(
                self.mesh, 1, lambda x, a=axis, b=value, tol=tolerance: abs(x[a] - b) < tol
            )
            blocks = fem.locate_dofs_topological(self.V, 1, facets)
            for component in constraint.components:
                dofs = 2 * blocks + (0 if component == "x" else 1)
                self.fixed[dofs[dofs < self.n]] = True
        self._work = self.K.createVecLeft()

    def apply(self, owned):
        """Collective K action; returned owned view is invalidated by the next call."""
        self.field.x.array[: self.n] = owned
        self.field.x.scatter_forward()
        self.K.mult(self.field.x.petsc_vec, self._work)
        return self._work.array_r

    def assemble_load(self, body_force, quadrature_degree=8):
        """Integrate a smooth UFL body force [N/m³], returning owned weak-load entries.

        Compile/assemble once for separable time factors, then scale the returned
        array in time. This is a volume load, not a discrete point-source API.
        """
        v = ufl.TestFunction(self.V)
        form = fem.form(
            ufl.inner(v, body_force) * ufl.dx(metadata={"quadrature_degree": quadrature_degree})
        )
        load = fem.assemble_vector(form)
        load.scatter_reverse(la.InsertMode.add)
        return load.array[: self.n].copy()

    def close(self):
        """Destroy explicit PETSc resources collectively; repeat calls are harmless."""
        if not self.closed:
            for resource in (self._work, self.K, self.M):
                if resource is not None:
                    resource.destroy()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
