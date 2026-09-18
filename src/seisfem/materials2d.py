"""Mesh-aligned horizontal isotropic or VTI material fields."""

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .vti import stiffnesses


class CellMaterials2D:
    """DG0 rho and moduli with one layer ID per local cell, including ghosts.

    Isotropic models retain rho/lambda/mu. Models containing VTI use
    rho/C11/C33/C13/C55, canonicalizing their isotropic layers only on that path.

    The caller supplies the generated affine rectangle and a validated layered
    PlaneStrainConfig. Configuration checks grid alignment; a separate vertex
    containment check below prevents silently assigning a cut cell by its center.
    Interface facets have two material traces, one from each adjacent cell.
    Displacement DOFs shared at the interface are not assigned a single material.
    """

    def __init__(self, msh, config):
        layers = config.material.layers
        dg = fem.functionspace(msh, ("DG", 0))
        if config.has_vti:
            self.rho, self.c11, self.c33, self.c13, self.c55 = (
                fem.Function(dg, name=name) for name in ("density", "c11", "c33", "c13", "c55")
            )
            fields = (self.rho, self.c11, self.c33, self.c13, self.c55)
        else:
            self.rho, self.lam, self.mu = (
                fem.Function(dg, name=name) for name in ("density", "lambda", "mu")
            )
            fields = (self.rho, self.lam, self.mu)
        self.cell_dofs = dg.dofmap.list[:, 0]
        z = msh.geometry.x[msh.geometry.dofmaps[0], 1]
        centers = z.mean(axis=1)
        self.cell_layers = np.searchsorted([layer.upper for layer in layers[:-1]], centers)
        lower = np.array([layer.lower for layer in layers])[self.cell_layers]
        upper = np.array([layer.upper for layer in layers])[self.cell_layers]
        scale = max(1.0, abs(config.domain.lower[1]), abs(config.domain.upper[1]))
        tolerance = 64 * np.finfo(float).eps * scale
        valid = np.all((z.min(axis=1) >= lower - tolerance) & (z.max(axis=1) <= upper + tolerance))
        if not msh.comm.allreduce(bool(valid), op=MPI.LAND):
            raise ValueError("Material interface cuts a cell; require mesh-aligned layers")
        if config.has_vti:
            values = np.array(
                [(layer.material.density, *stiffnesses(layer.material)) for layer in layers]
            )
        else:
            values = np.array([(layer.material.density, *layer.material.lame) for layer in layers])
        for i, field in enumerate(fields):
            field.x.array[self.cell_dofs] = values[self.cell_layers, i]
            field.x.scatter_forward()
