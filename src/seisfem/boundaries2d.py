"""FE boundary impedance on the supported affine, axis-aligned rectangle mesh."""

import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc
from mpi4py import MPI


def assemble_boundary_damping(V, config, material_fields=None, *, collocated_degree=None):
    """Return C and owned damping: P1 row sums or the collocated SEM diagonal.

    sigma*n = -B*v, B=rho*(Vs*I + (Vp-Vs)*n⊗n), on selected exterior facets.
    Entries of C_L have units kg/(m*s), i.e. line force divided by velocity.
    On these axis-aligned sides B is diagonal and positive. Scalar P1 trace
    functions are nonnegative and sum to one, so row sums are nonnegative.
    No such row-sum positivity claim is made for general rotated/curved meshes.

    Layered operators supply their synchronized DG0 rho/lambda/mu fields.
    On exterior ds facets these have the unique adjacent-cell trace, giving
    Zs=sqrt(rho*mu), Zp=sqrt(rho*(lambda+2*mu)). No internal facet term is added.

    The caller owns the returned PETSc matrix; the no-absorber path returns
    None and exact zeros without any facet search or extra matrix assembly.
    SEM supplies collocated_degree: GLL edge quadrature then makes C diagonal
    on these axis-aligned sides. Extract that diagonal without row-sum lumping.
    """
    msh, domain = V.mesh, config.domain
    count = V.dofmap.index_map_bs * V.dofmap.index_map.size_local
    if not config.boundaries.absorbing_sides:
        return None, np.zeros(count)
    sides = {
        "left": (0, domain.lower[0]),
        "right": (0, domain.upper[0]),
        "lower": (1, domain.lower[1]),
        "upper": (1, domain.upper[1]),
    }
    marked = []
    if collocated_degree is not None:
        msh.topology.create_connectivity(1, 2)
        exterior = mesh.exterior_facet_indices(msh.topology)
        geometry = mesh.entities_to_geometry(msh, 1, exterior)
        facet_coordinates = msh.geometry.x[geometry]
    for side in config.boundaries.absorbing_sides:
        axis, value = sides[side]
        tolerance = (domain.upper[axis] - domain.lower[axis]) / domain.cells[axis] * 1e-8
        if collocated_degree is None:
            marked.append(
                mesh.locate_entities_boundary(
                    msh, 1, lambda x, a=axis, b=value, tol=tolerance: abs(x[a] - b) < tol
                )
            )
        else:
            # DOLFINx 0.11's vertex locator does not consistently respect the
            # TP coordinate DOF ordering. Map exterior entities to geometry
            # explicitly; require every physical facet vertex on the side.
            on_side = np.all(abs(facet_coordinates[:, :, axis] - value) < tolerance, axis=1)
            marked.append(exterior[on_side])
    # Only owned exterior facets are located. Sorted, unique facet tags avoid
    # duplicating any integration domain; shared corner DOFs still receive both
    # adjacent facet integrals through ordinary reverse-add assembly.
    facets = np.unique(np.concatenate(marked)).astype(np.int32)
    tags = mesh.meshtags(msh, 1, facets, np.ones(len(facets), dtype=np.int32))
    msh.topology.create_connectivity(1, 2)
    if collocated_degree is None:
        ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)(1)
    else:
        ds = ufl.Measure(
            "ds",
            domain=msh,
            subdomain_data=tags,
            metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * collocated_degree - 1},
        )(1)
    normal = ufl.FacetNormal(msh)
    if material_fields is None:
        # Preserve the validated homogeneous expression and assembly path.
        rho = config.material.density
        vp, vs = config.material.speed("P"), config.material.speed("S")
        impedance = rho * (vs * ufl.Identity(2) + (vp - vs) * ufl.outer(normal, normal))
    else:
        rho, lam, mu = material_fields.rho, material_fields.lam, material_fields.mu
        zs = ufl.sqrt(rho * mu)
        zp = ufl.sqrt(rho * (lam + 2 * mu))
        impedance = zs * ufl.Identity(2) + (zp - zs) * ufl.outer(normal, normal)
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    matrix = None
    try:
        matrix = petsc.assemble_matrix(fem.form(ufl.inner(w, impedance * u) * ds))
        matrix.assemble()
        if collocated_degree is not None:
            diagonal_vector = matrix.getDiagonal()
            try:
                diagonal = diagonal_vector.array_r.copy()
            finally:
                diagonal_vector.destroy()
            offsets, columns, entries = matrix.getValuesCSR()
            first, _ = matrix.getOwnershipRange()
            rows = np.repeat(np.arange(count) + first, np.diff(offsets))
            off = msh.comm.allreduce(
                float(np.max(abs(entries[rows != columns]), initial=0)), op=MPI.MAX
            )
            scale = msh.comm.allreduce(float(np.max(abs(diagonal), initial=0)), op=MPI.MAX)
            if off > 1e-12 * scale:
                raise RuntimeError("GLL boundary damping is not diagonal; check edge quadrature")
            if not msh.comm.allreduce(
                bool(np.all(np.isfinite(diagonal) & (diagonal >= 0))), op=MPI.LAND
            ):
                raise RuntimeError("Boundary damping must be finite and nonnegative")
            return matrix, diagonal
        # Sum all trial basis functions: the vector field (1,1). This integrates
        # the row sum directly, and is equivalent to C @ ones, before constraints.
        lumped = fem.assemble_vector(
            fem.form(ufl.inner(w, impedance * ufl.as_vector((1.0, 1.0))) * ds)
        )
        lumped.scatter_reverse(la.InsertMode.add)
        diagonal = lumped.array[:count].copy()
        if not msh.comm.allreduce(
            bool(np.all(np.isfinite(diagonal) & (diagonal >= 0))), op=MPI.LAND
        ):
            raise RuntimeError("Boundary damping must be finite and nonnegative")
        return matrix, diagonal
    except Exception:
        if matrix is not None:
            matrix.destroy()
        raise
