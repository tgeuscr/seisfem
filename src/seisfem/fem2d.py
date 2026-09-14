"""Isotropic plane-strain vector P1 operators on affine rectangle triangles."""

from contextlib import contextmanager
from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

from .boundaries2d import assemble_boundary_damping
from .config import Layered
from .config2d import PlaneStrainConfig
from .materials2d import CellMaterials2D
from .timestepping import CentralDifference


def strain(u):
    """In-plane small strain; epsilon_yy is constrained to zero."""
    return ufl.sym(ufl.grad(u))


def stress(u, lam, mu):
    """In-plane block of 3D isotropic stress, with ordinary 3D Lamé parameters."""
    epsilon = strain(u)
    return lam * ufl.tr(epsilon) * ufl.Identity(2) + 2 * mu * epsilon


@dataclass(frozen=True)
class SpectralDiagnostic:
    """Sparse eigenpair estimate plus a separate, sufficient algebraic upper bound."""

    lambda_max: float
    relative_residual: float
    lambda_upper_bound: float

    @property
    def critical_dt(self):
        """Estimated marginal threshold; arbitrary data require strictly smaller dt."""
        return 2 / np.sqrt(self.lambda_max) if self.lambda_max > 0 else float("inf")


class PlaneStrainOperators:
    """Collective operator owner; scalar storage is [ux0,uz0,ux1,uz1,...].

    M and K are unmodified consistent matrices. `mass` contains positive lumped
    entries for owned scalar DOFs, including constrained ones. `fixed` identifies
    zero-displacement components for time-step projection; no artificial boundary
    diagonals are inserted. Geometry coordinates are per two-component node.
    C is the consistent boundary impedance matrix (None without absorbers);
    damping is its nonnegative row-sum lumping on owned scalar DOFs.
    """

    def __init__(self, config: PlaneStrainConfig, comm=MPI.COMM_WORLD):
        self.comm, self.config = comm, config
        self.M = self.K = self.C = self._work = None
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
        self.material_fields = None
        if isinstance(cfg.material, Layered):
            self.material_fields = CellMaterials2D(self.mesh, cfg)
            rho = self.material_fields.rho
            lam, mu = self.material_fields.lam, self.material_fields.mu
        else:
            # Retain the validated homogeneous forms and numerical execution path.
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
        self.C, self.damping = assemble_boundary_damping(self.V, cfg)
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
        integrand = ufl.inner(v, body_force)
        # UFL removes the test argument (and implicit domain) from an exact zero.
        # Such a rank-zero form cannot be passed to assemble_vector.
        if isinstance(integrand, ufl.constantvalue.Zero):
            return np.zeros(self.n, dtype=PETSc.ScalarType)
        form = fem.form(
            integrand * ufl.dx(domain=self.mesh, metadata={"quadrature_degree": quadrature_degree})
        )
        load = fem.assemble_vector(form)
        load.scatter_reverse(la.InsertMode.add)
        return load.array[: self.n].copy()

    @contextmanager
    def _scaled_free_operator(self):
        """Own a temporary D_f^-1/2 K_ff D_f^-1/2, with constraints eliminated."""
        first, _ = self.index_map.local_range
        indices = (2 * first + np.flatnonzero(~self.fixed)).astype(PETSc.IntType)
        selection = PETSc.IS().createGeneral(indices, comm=self.comm)
        matrix = scaling = None
        try:
            matrix = self.K.createSubMatrix(selection, selection)
            scaling = matrix.createVecLeft()
            scaling.array[:] = 1 / np.sqrt(self.mass[~self.fixed])
            matrix.diagonalScale(scaling, scaling)
            yield matrix
        finally:
            if scaling is not None:
                scaling.destroy()
            if matrix is not None:
                matrix.destroy()
            selection.destroy()

    def _eigenvalue_bound(self, matrix):
        # Absolute row sums of a symmetric matrix bound every eigenvalue in
        # magnitude. Vector-elastic off-diagonal signs play no role in this proof.
        offsets, _, entries = matrix.getValuesCSR()
        local_max = max(
            (np.sum(abs(entries[a:b])) for a, b in zip(offsets[:-1], offsets[1:], strict=True)),
            default=0.0,
        )
        return self.comm.allreduce(float(local_max), op=MPI.MAX)

    @property
    def stable_dt(self):
        """Sufficient assembled spectral bound without a safety factor [s]."""
        with self._scaled_free_operator() as matrix:
            bound = self._eigenvalue_bound(matrix)
        return 2 / np.sqrt(bound) if bound > 0 else float("inf")

    def spectral_diagnostic(self):
        """Estimate the largest free eigenvalue with SLEPc Krylov-Schur.

        SLEPc is optional for assembly/time stepping and supplied by the existing
        binary lock. The converged Ritz value is a diagnostic, not a certified
        upper bound; start() uses the independent absolute-row-sum bound.
        """
        from slepc4py import SLEPc

        with self._scaled_free_operator() as matrix:
            bound = self._eigenvalue_bound(matrix)
            size = matrix.getSize()[0]
            if size <= 1:
                return SpectralDiagnostic(bound, 0.0, bound)
            solver = SLEPc.EPS().create(self.comm)
            try:
                solver.setOperators(matrix)
                solver.setProblemType(SLEPc.EPS.ProblemType.HEP)
                solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
                solver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
                solver.setDimensions(1, min(30, size))
                solver.setTolerances(1e-11, 2000)
                solver.solve()
                if solver.getConvergedReason() <= 0 or solver.getConverged() < 1:
                    raise RuntimeError("Largest-eigenvalue iteration did not converge")
                value = float(solver.getEigenvalue(0).real)
                residual = solver.computeError(0, SLEPc.EPS.ErrorType.RELATIVE)
                if value <= 0 or not np.isfinite(value) or residual > 1e-9:
                    raise RuntimeError("Invalid largest eigenpair or excessive residual")
                return SpectralDiagnostic(value, float(residual), bound)
            finally:
                solver.destroy()

    def start(self, dt, u0=None, v0=None, force0=None, safety=0.9):
        """Create the unchanged central-difference integrator with verified bounds.

        Call collectively with the same dt/safety, and owned scalar initial arrays.
        The caller supplies F(t_n), records states, and accepts updates with advance.
        No 2D Simulation, point source, receiver or field-output frontend is implied.
        """
        settings = self.comm.allgather((dt, safety))
        if any(setting != settings[0] for setting in settings):
            raise ValueError("All ranks must use identical dt and safety")
        if not np.isfinite(dt) or dt <= 0 or not 0 < safety < 1:
            raise ValueError("Require finite dt>0 and 0<safety<1")
        arrays, local_errors = [], []
        for name, value in zip(("u0", "v0", "force0"), (u0, v0, force0), strict=True):
            try:
                array = np.zeros(self.n) if value is None else np.asarray(value, dtype=float)
                if array.shape != (self.n,):
                    raise ValueError(
                        f"expected owned scalar-DOF shape {(self.n,)}, got {array.shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise ValueError("entries must be finite")
                arrays.append(array)
            except Exception as error:
                # Ordinary argument errors must not let one rank skip the collective.
                local_errors.append(f"{name}: {type(error).__name__}: {error}")
        errors = self.comm.allgather(local_errors)
        failures = [
            f"rank {rank}: {message}"
            for rank, messages in enumerate(errors)
            for message in messages
        ]
        if failures:
            raise ValueError("Invalid initial data; " + "; ".join(failures))
        if dt > safety * self.stable_dt:
            raise ValueError("dt exceeds the assembled plane-strain spectral bound with safety")
        return CentralDifference(self.mass, self.damping, self.fixed, dt, self.apply, *arrays)

    def close(self):
        """Destroy explicit PETSc resources collectively; repeat calls are harmless."""
        if not self.closed:
            for resource in (self._work, self.C, self.K, self.M):
                if resource is not None:
                    resource.destroy()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
