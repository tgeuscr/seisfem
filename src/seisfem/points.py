"""Cached distributed point evaluation for the explicitly supported scalar P1 space."""

from collections.abc import Sequence

import numpy as np
from dolfinx import fem, geometry, la
from mpi4py import MPI


class PointMap:
    """Elect one owned-cell rank per point, then cache cell DOFs and basis values.

    Coordinates/IDs are initially replicated. Only elected ranks store traces;
    no receiver collective occurs at each sample. This setup is O(ranks*points)
    and must be replaced by routed point ownership for very large receiver sets.
    """

    def __init__(self, V: fem.FunctionSpace, positions: Sequence[float], h: float):
        msh, comm = V.mesh, V.mesh.comm
        self.V = V
        points = np.zeros((len(positions), 3), dtype=msh.geometry.x.dtype)
        points[:, 0] = positions
        cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)
        tree = geometry.bb_tree(msh, msh.topology.dim, entities=cells, padding=h * 1e-10)
        candidates = geometry.compute_collisions_points(tree, points)
        collisions = geometry.compute_colliding_cells(msh, candidates, points)
        chosen = np.array(
            [
                collisions.links(i)[0] if len(collisions.links(i)) else -1
                for i in range(len(points))
            ],
            dtype=np.int32,
        )
        election = np.where(chosen >= 0, comm.rank, comm.size).astype(np.int32)
        owners = np.empty_like(election)
        comm.Allreduce(election, owners, op=MPI.MIN)
        if np.any(owners == comm.size):
            raise ValueError("Point localization failed for an in-domain point")
        self.ids = np.flatnonzero(owners == comm.rank)
        self.cells = chosen[self.ids]
        self.points = points[self.ids]
        self.dofs = np.array([V.dofmap.cell_dofs(c) for c in self.cells], dtype=np.int32)
        self.dofs = self.dofs.reshape(-1, 2)
        z = V.tabulate_dof_coordinates()[:, 0][self.dofs]
        fraction = (self.points[:, 0] - z[:, 0]) / (z[:, 1] - z[:, 0])
        self.weights = np.column_stack((1 - fraction, fraction))
        if np.any(self.weights < -1e-8):
            raise RuntimeError("Point basis weights outside cell")
        self._sample = fem.Function(V)

    def evaluate(self, owned: np.ndarray) -> np.ndarray:
        """Collectively refresh ghosts and return values in this rank's point-ID order."""
        n = self.V.dofmap.index_map.size_local
        self._sample.x.array[:n] = owned
        self._sample.x.scatter_forward()
        return np.sum(self.weights * self._sample.x.array[self.dofs], axis=1)

    def unit_load(self) -> np.ndarray:
        """Assemble sum phi_i(z_s), each source inserted on exactly one cell/rank."""
        load = fem.Function(self.V)
        np.add.at(load.x.array, self.dofs.ravel(), self.weights.ravel())
        load.x.scatter_reverse(la.InsertMode.add)
        return load.x.array[: self.V.dofmap.index_map.size_local].copy()
