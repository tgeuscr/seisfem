"""FE boundary impedance on the supported affine, axis-aligned rectangle mesh."""

import numpy as np
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc
from mpi4py import MPI


def assemble_boundary_damping(V, config):
    """Collectively return consistent C and owned row-sum diagonal C_L.

    sigma*n = -B*v, B=rho*(Vs*I + (Vp-Vs)*n⊗n), on selected exterior facets.
    Entries of C_L have units kg/(m*s), i.e. line force divided by velocity.
    On these axis-aligned sides B is diagonal and positive. Scalar P1 trace
    functions are nonnegative and sum to one, so row sums are nonnegative.
    No such row-sum positivity claim is made for general rotated/curved meshes.

    The caller owns the returned PETSc matrix; the no-absorber path returns
    None and exact zeros without any facet search or extra matrix assembly.
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
    for side in config.boundaries.absorbing_sides:
        axis, value = sides[side]
        tolerance = (domain.upper[axis] - domain.lower[axis]) / domain.cells[axis] * 1e-8
        marked.append(
            mesh.locate_entities_boundary(
                msh, 1, lambda x, a=axis, b=value, tol=tolerance: abs(x[a] - b) < tol
            )
        )
    # Only owned exterior facets are located. Sorted, unique facet tags avoid
    # duplicating any integration domain; shared corner DOFs still receive both
    # adjacent facet integrals through ordinary reverse-add assembly.
    facets = np.unique(np.concatenate(marked)).astype(np.int32)
    tags = mesh.meshtags(msh, 1, facets, np.ones(len(facets), dtype=np.int32))
    msh.topology.create_connectivity(1, 2)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)(1)
    normal = ufl.FacetNormal(msh)
    rho = config.material.density
    vp, vs = config.material.speed("P"), config.material.speed("S")
    impedance = rho * (vs * ufl.Identity(2) + (vp - vs) * ufl.outer(normal, normal))
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    matrix = None
    try:
        matrix = petsc.assemble_matrix(fem.form(ufl.inner(w, impedance * u) * ds))
        matrix.assemble()
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
