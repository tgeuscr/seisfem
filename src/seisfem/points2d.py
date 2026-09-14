"""Cached point functionals for affine triangular vector P1 elements."""

import numpy as np
from dolfinx import fem, geometry, la

from ._collective import collective_input, identical_input
from .config2d import unit_direction


class PointMap2D:
    """Elect one owned triangle per point using a partition-independent cell key.

    The key is the sorted physical vertex coordinates of a triangle, then rank
    as a final tie breaker. Only elected ranks store interpolation supports.
    Setup replicates O(points*ranks) candidate keys, never the complete mesh.
    Methods taking owned field arrays are collective because they refresh ghosts.
    """

    def __init__(self, V, positions):
        self.V = V
        msh, comm = V.mesh, V.mesh.comm

        def validate():
            if V.dofmap.bs != 2 or V.dofmap.index_map_bs != 2 or V.element.space_dimension != 6:
                raise ValueError("PointMap2D requires triangular vector P1 with two components")
            points = np.asarray(positions, dtype=float)
            if points.shape == (0,):
                points = points.reshape(0, 2)
            if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
                raise ValueError("Point coordinates must have finite shape (point, 2)")
            return points

        positions = collective_input(comm, validate, "point coordinates/space")
        identical_input(comm, positions.tolist(), "point coordinates")
        self.positions = positions.copy()
        xyz = np.zeros((len(positions), 3), dtype=msh.geometry.x.dtype)
        xyz[:, :2] = positions
        coordinates = V.tabulate_dof_coordinates()[:, :2]
        cells = np.arange(msh.topology.index_map(2).size_local, dtype=np.int32)

        def candidates():
            tree = geometry.bb_tree(msh, 2, entities=cells)
            hits = geometry.compute_colliding_cells(
                msh, geometry.compute_collisions_points(tree, xyz), xyz
            )
            choices = []
            for i in range(len(positions)):
                options = []
                for cell in hits.links(i):
                    # Use stored mesh geometry, not mapped DOF coordinates: the
                    # latter can differ by roundoff with cell permutation/rank.
                    vertices = msh.geometry.x[msh.geometry.dofmaps[0][cell], :2]
                    key = tuple(sorted(tuple(vertex) for vertex in vertices))
                    options.append((key, comm.rank, int(cell)))
                choices.append(min(options) if options else None)
            return choices

        choices = collective_input(comm, candidates, "point localization")
        gathered = comm.allgather(choices)
        winners = []
        for i in range(len(positions)):
            options = [rank_choices[i] for rank_choices in gathered if rank_choices[i] is not None]
            if not options:
                raise ValueError(f"Point {i} at {positions[i].tolist()} is outside the mesh")
            winners.append(min(options))
        self.owners = np.array([winner[1] for winner in winners], dtype=np.int32)
        self.ids = np.flatnonzero(self.owners == comm.rank)
        self.cells = np.array([winners[i][2] for i in self.ids], dtype=np.int32)
        self.cell_keys = tuple(winner[0] for winner in winners)

        def supports():
            nodes = np.array([V.dofmap.cell_dofs(c) for c in self.cells], dtype=np.int32)
            nodes = nodes.reshape(-1, 3)
            vertices = coordinates[nodes]
            # Barycentric coordinates in the nodal reference triangle:
            # x = X0 + xi*(X1-X0) + eta*(X2-X0).
            jacobian = np.stack(
                (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=2
            )
            reference = np.linalg.solve(
                jacobian, (positions[self.ids] - vertices[:, 0])[..., None]
            )[..., 0]
            weights = np.column_stack((1 - reference.sum(axis=1), reference))
            if np.any(weights < -1e-10) or np.any(weights > 1 + 1e-10):
                raise ValueError("Localized point has invalid P1 weights")
            return nodes, reference, weights

        self.nodes, self.reference_coordinates, self.weights = collective_input(
            comm, supports, "point interpolation support"
        )
        self.dofs = 2 * self.nodes[:, :, None] + np.arange(2)
        blocks = V.dofmap.index_map.local_to_global(self.nodes.ravel()).reshape(-1, 3)
        self.global_dofs = 2 * blocks[:, :, None] + np.arange(2)
        self._sample = fem.Function(V)
        self.n = 2 * V.dofmap.index_map.size_local

    def evaluate(self, owned):
        """Refresh field ghosts; return (local receiver, x/z) in increasing ID order.

        owned is a finite scalar-DOF array from the integrator, shape (self.n,).
        This hot path trusts the prevalidated runtime array layout.
        """
        self._sample.x.array[: self.n] = owned
        self._sample.x.scatter_forward()
        return np.sum(self.weights[:, :, None] * self._sample.x.array[self.dofs], axis=1)

    def unit_load(self, direction):
        """Sum normalized d·v(xs) over these points, inserted exactly once each.

        Return owned weak-load entries. Reverse-add sends ghost contributions to
        DOF owners once at setup. No division by mass, area, or mesh size occurs.
        """
        comm = self.V.mesh.comm
        direction = collective_input(comm, lambda: unit_direction(direction), "source direction")
        identical_input(comm, direction, "source direction")
        load = fem.Function(self.V)
        values = self.weights[:, :, None] * np.asarray(direction)
        np.add.at(load.x.array, self.dofs.ravel(), values.ravel())
        load.x.scatter_reverse(la.InsertMode.add)
        return load.x.array[: self.n].copy()
