"""Same-form compiler equivalence and physical equivalence to the old mesh."""

import basix
import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc
from mpi4py import MPI

from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators, strain, stress
from seisfem.sem2d import gll_element
from tests.plane_strain.helpers import dense

from .helpers import config, record


def assemble(form, factorized=False):
    matrix = petsc.assemble_matrix(
        fem.form(form, form_compiler_options={"sum_factorization": factorized})
    )
    matrix.assemble()
    try:
        return dense(matrix)
    finally:
        matrix.destroy()


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def physical_order(x):
    keys = np.round(x, 12)
    return np.lexsort((keys[:, 1], keys[:, 0]))


def physical_cells(msh):
    cells = []
    for cell in range(msh.topology.index_map(2).size_local):
        points = msh.geometry.x[msh.geometry.dofmaps[0][cell], :2]
        cells.append(points[physical_order(points)].ravel())
    cells = np.array(cells)
    # A physical corner signature is independent of cell and geometry numbering.
    keys = np.round(cells, 12)
    return cells[np.lexsort(tuple(keys[:, j] for j in reversed(range(keys.shape[1]))))]


@pytest.mark.parametrize("p", [2, 4, 6])
def test_tensor_coordinates_and_factorized_assembly(p):
    data = config(p).model_dump(mode="json", by_alias=True)
    data["domain"] = dict(lower=(-0.7, -1.2), upper=(1.6, 0.5), cells=(3, 2))
    cfg = PlaneStrainConfig.model_validate(data)
    with PlaneStrainOperators(cfg, MPI.COMM_SELF) as op:
        coordinate = op.mesh.ufl_domain().ufl_coordinate_element().basix_element
        assert coordinate.has_tensor_product_factorisation
        assert coordinate.degree == 1
        assert op.sem_metadata["sum_factorization"]
        old_mesh = mesh.create_rectangle(
            MPI.COMM_SELF,
            [cfg.domain.lower, cfg.domain.upper],
            cfg.domain.cells,
            cell_type=mesh.CellType.quadrilateral,
        )
        assert op.mesh.topology.index_map(2).size_global == 6
        corners, old_corners = physical_cells(op.mesh), physical_cells(old_mesh)
        np.testing.assert_array_equal(corners, old_corners)
        dx = ufl.Measure(
            "dx",
            domain=op.mesh,
            metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * p - 1},
        )
        scalar = basix.ufl.wrap_element(
            basix.create_tp_element(
                basix.ElementFamily.P,
                basix.CellType.quadrilateral,
                p,
                basix.LagrangeVariant.gll_warped,
            )
        )
        Q = fem.functionspace(op.mesh, scalar)
        a, b = ufl.TrialFunction(Q), ufl.TestFunction(Q)
        errors = {}
        for name, form in [
            ("scalar_mass", a * b * dx),
            ("scalar_stiffness", ufl.inner(ufl.grad(a), ufl.grad(b)) * dx),
        ]:
            errors[name] = relative(assemble(form, True), assemble(form, False))
            assert errors[name] <= 1e-12
        u, v = ufl.TrialFunction(op.V), ufl.TestFunction(op.V)
        lam, mu = cfg.material.lame
        reference_M = assemble(cfg.material.density * ufl.inner(u, v) * dx)
        reference_K = assemble(ufl.inner(strain(v), stress(u, lam, mu)) * dx)
        M, K = dense(op.M), dense(op.K)
        errors["mass"] = relative(M, reference_M)
        errors["stiffness"] = relative(K, reference_K)
        assert max(errors.values()) <= 1e-12

        old_V = fem.functionspace(old_mesh, gll_element(p))
        a, b = ufl.TrialFunction(old_V), ufl.TestFunction(old_V)
        old_dx = ufl.Measure("dx", domain=old_mesh, metadata=dx.metadata())
        old_M = assemble(cfg.material.density * ufl.inner(a, b) * old_dx)
        old_K = assemble(ufl.inner(strain(b), stress(a, lam, mu)) * old_dx)
        old_xy = old_V.tabulate_dof_coordinates()[:, :2]
        order, old_order = physical_order(op.coordinates), physical_order(old_xy)
        np.testing.assert_allclose(op.coordinates[order], old_xy[old_order], rtol=0, atol=3e-14)
        ids = (2 * order[:, None] + np.arange(2)).ravel()
        old_ids = (2 * old_order[:, None] + np.arange(2)).ravel()
        legacy_errors = dict(
            mass=relative(M[np.ix_(ids, ids)], old_M[np.ix_(old_ids, old_ids)]),
            stiffness=relative(K[np.ix_(ids, ids)], old_K[np.ix_(old_ids, old_ids)]),
        )
        assert max(legacy_errors.values()) <= 1e-12
        mass = np.diag(old_M)
        eig = np.linalg.eigvalsh(old_K / np.sqrt(np.outer(mass, mass)))
        old_dtcrit = 2 / np.sqrt(eig[-1])
        old_bound = 2 / np.sqrt(np.max(np.sum(abs(old_K / np.sqrt(np.outer(mass, mass))), axis=1)))
        dt_error = abs(op.spectral_diagnostic().critical_dt / old_dtcrit - 1)
        bound_error = abs(op.stable_dt / old_bound - 1)
        assert max(dt_error, bound_error) < 1e-12
        off = float(np.max(abs(M - np.diag(np.diag(M)))))
        assert off < 1e-13 * np.max(op.mass)
        mass_error = abs(float(np.sum(op.mass) / np.trace(old_M) - 1))
        assert mass_error < 3e-14
        symmetry = float(np.max(abs(K - K.T)) / np.max(abs(K)))
        rigid = max(
            float(np.max(abs(K @ np.tile(d, op.n // 2))) / np.max(abs(K))) for d in [(1, 0), (0, 1)]
        )
        assert symmetry < 2e-14
        assert rigid < 3e-13
        field = np.column_stack(
            (np.sin(op.coordinates[:, 0]), np.cos(op.coordinates[:, 1]))
        ).ravel()
        action_error = relative(K @ field, reference_K @ field)
        assert action_error < 1e-12
        record(
            f"factorization-{p}",
            dict(
                degree=p,
                coordinate_tensor_product=True,
                same_mesh_errors=errors,
                legacy_mesh_errors=legacy_errors,
                corner_coordinate_error=float(np.max(abs(corners - old_corners))),
                mass_offdiagonal_absolute=off,
                mass_offdiagonal_relative=off / np.max(op.mass),
                integrated_mass_relative_error=mass_error,
                symmetry=symmetry,
                rigid_residual=rigid,
                stiffness_action_relative_error=action_error,
                dtcrit_relative_difference=dt_error,
                safe_dt_relative_difference=bound_error,
            ),
        )
