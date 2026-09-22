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

from .fem2d import strain, stress


def gll_element(degree):
    """Scalar Basix tensor-product ordering wrapped as a blocked x/z vector."""
    scalar = basix.create_tp_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        degree,
        basix.LagrangeVariant.gll_warped,
    )
    return basix.ufl.blocked_element(basix.ufl.wrap_element(scalar), shape=(2,))


def assemble_gll(op):
    """Populate the existing operator contract, without mass row-sum lumping."""
    cfg, comm = op.config, op.comm
    degree = cfg.discretization.degree
    op.mesh = mesh.create_rectangle(
        comm,
        [cfg.domain.lower, cfg.domain.upper],
        cfg.domain.cells,
        cell_type=mesh.CellType.quadrilateral,
    )
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
    rho = cfg.material.density
    lam, mu = cfg.material.lame
    op.M = petsc.assemble_matrix(fem.form(rho * ufl.inner(u, v) * dx))
    op.M.assemble()
    op.K = petsc.assemble_matrix(fem.form(ufl.inner(strain(v), stress(u, lam, mu)) * dx))
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
        sum_factorization=False,
    )
    op.fixed = np.zeros(op.n, dtype=bool)
    op.damping = np.zeros(op.n)
    op._work = op.K.createVecLeft()
