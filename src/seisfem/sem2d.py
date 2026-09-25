"""Assembled continuous GLL spectral elements on structured affine rectangles.

Only spatial assembly differs from PlaneStrainOperators. Tensor-product DOF
ordering is retained; the PETSc matrix action is deliberately the reference path.
"""

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI

from .config import Layered
from .fem2d import strain, stress
from .materials2d import CellMaterials2D


def gll_element(degree):
    """Scalar Basix tensor-product ordering wrapped as a blocked x/z vector."""
    scalar = basix.create_tp_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        degree,
        basix.LagrangeVariant.gll_warped,
    )
    return basix.ufl.blocked_element(basix.ufl.wrap_element(scalar), shape=(2,))


def _rectangle(comm, domain):
    """Structured affine mesh with tensor-product Q1 coordinate tabulation.

    Rank zero supplies global input nodes/cells; create_mesh partitions them.
    Coordinate DOFs follow the Basix element's ordering, not cyclic corners.
    """
    coordinate = basix.create_tp_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        1,
        basix.LagrangeVariant.gll_warped,
    )
    if comm.rank == 0:
        nx, nz = domain.cells
        # Match create_rectangle's coordinate arithmetic, including the endpoints.
        x = domain.lower[0] + np.arange(nx + 1) * ((domain.upper[0] - domain.lower[0]) / nx)
        z = domain.lower[1] + np.arange(nz + 1) * ((domain.upper[1] - domain.lower[1]) / nz)
        xx, zz = np.meshgrid(x, z)
        coordinates = np.column_stack((xx.ravel(), zz.ravel()))
        ix, iz = np.meshgrid(np.arange(nx), np.arange(nz))
        lower_left = (iz * (nx + 1) + ix).ravel()
        corners = coordinate.points.astype(np.int64)
        offsets = corners[:, 0] + (nx + 1) * corners[:, 1]
        cells = lower_left[:, None] + offsets
    else:
        coordinates = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 4), dtype=np.int64)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, 2)
    return mesh.create_mesh(comm, cells, coordinate, coordinates, partitioner=partitioner)


def assemble_gll(op):
    """Populate the existing operator contract, without mass row-sum lumping."""
    cfg, comm = op.config, op.comm
    degree = cfg.discretization.degree
    op.mesh = _rectangle(comm, cfg.domain)
    op.V = fem.functionspace(op.mesh, gll_element(degree))
    op.index_map = op.V.dofmap.index_map
    if op.V.dofmap.bs != 2 or op.V.dofmap.index_map_bs != 2:
        raise RuntimeError("Expected blocked GLL space with two components")
    op.n = 2 * op.index_map.size_local
    op.coordinates = op.V.tabulate_dof_coordinates()[:, :2].copy()
    op.field = fem.Function(op.V, name="displacement")
    op.material_fields = None
    u, v = ufl.TrialFunction(op.V), ufl.TestFunction(op.V)
    dx = ufl.Measure(
        "dx",
        domain=op.mesh,
        metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * degree - 1},
    )
    if isinstance(cfg.material, Layered):
        # Tensor-product DG0 keeps coefficient tables factorable too. Material
        # assignment/containment and ghost updates use the shared layered path.
        cell_element = basix.create_tp_element(
            basix.ElementFamily.P,
            basix.CellType.quadrilateral,
            0,
            basix.LagrangeVariant.gll_warped,
            discontinuous=True,
        )
        space = fem.functionspace(op.mesh, basix.ufl.wrap_element(cell_element))
        op.material_fields = CellMaterials2D(op.mesh, cfg, space=space)
        rho = op.material_fields.rho
        lam, mu = op.material_fields.lam, op.material_fields.mu
    else:
        rho = cfg.material.density
        lam, mu = cfg.material.lame
    options = {"sum_factorization": True}
    op.M = petsc.assemble_matrix(
        fem.form(rho * ufl.inner(u, v) * dx, form_compiler_options=options)
    )
    op.M.assemble()
    op.K = petsc.assemble_matrix(
        fem.form(
            ufl.inner(strain(v), stress(u, lam, mu)) * dx,
            form_compiler_options=options,
        )
    )
    op.K.assemble()
    diagonal = op.M.getDiagonal()
    try:
        op.mass = diagonal.array_r.copy()
    finally:
        diagonal.destroy()
    offsets, columns, entries = op.M.getValuesCSR()
    first, _ = op.M.getOwnershipRange()
    rows = np.repeat(np.arange(op.n) + first, np.diff(offsets))
    off = comm.allreduce(float(np.max(abs(entries[rows != columns]), initial=0)), op=MPI.MAX)
    scale = comm.allreduce(float(np.max(abs(op.mass), initial=0)), op=MPI.MAX)
    if not comm.allreduce(bool(np.all(np.isfinite(op.mass) & (op.mass > 0))), op=MPI.LAND):
        raise RuntimeError("GLL mass diagonal must be finite and positive")
    if off > 1e-12 * scale:
        raise RuntimeError("GLL mass is not diagonal: diagnose basis/quadrature mismatch")
    op.sem_metadata = dict(
        degree=degree,
        quadrature_degree=2 * degree - 1,
        mass_offdiagonal_absolute=off,
        mass_offdiagonal_relative=off / scale,
        tensor_product_ordering=True,
        coordinate_tensor_product=True,
        sum_factorization=True,
    )
    op.fixed = np.zeros(op.n, dtype=bool)
    op.damping = np.zeros(op.n)
    op._work = op.K.createVecLeft()
