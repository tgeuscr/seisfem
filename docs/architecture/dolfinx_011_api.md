# DOLFINx 0.11 API audit

Checked 2026-09-10 against the installed 0.11.0 package signatures/source and the
versioned official **0.11.0.post0** documentation (not `main`, which advertises
0.12 development). Runtime rejects other minor versions and non-real-double
PETSc builds. The docs publication suffix differs from the installed version.

* [fem](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.html):
  `functionspace(mesh, ("Lagrange", 1))` and `( "DG", 0)` use current element
  metadata; vector Basix elements are not yet needed. Compile with `fem.form`.
  Linear assembly returns a DOLFINx vector; explicitly reverse-add ghosts.
  `Function.eval` takes three-coordinate points and local cell indices. Tests
  compare it to cached interval basis weights.
* [fem.petsc](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html):
  explicitly import the PETSc submodule; finalize `assemble_matrix` with
  `Mat.assemble()`. Explicitly created PETSc matrix/work vector are destroyed
  collectively. Function-owned cached PETSc wrappers remain owned by DOLFINx.
* [mesh](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.mesh.html):
  `create_interval(comm, nx, endpoints)` creates a distributed mesh. No root-only
  Python mesh or degree-of-freedom gather is used. Geometry slot 0 represents z.
* [geometry](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.geometry.html):
  `bb_tree` is restricted to owned cells, followed by `compute_collisions_points`
  and `compute_colliding_cells`. Elect one rank once, then cache local cell DOFs.
  `determine_point_ownership` was reviewed: it offers a routed future path, but
  its source/destination ordering requires explicit ID handling. The simpler
  tested election does not claim many-receiver scalability.
* [io](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.io.html):
  XDMF writes cellwise constants directly; other Functions are interpolated to
  mesh geometry nodes. This is suitable for our affine CG1 field, but would lose
  higher-order field information. VTX supports higher-order Lagrange fields and
  mesh reuse; its function collection must share mesh/element type. Hence do not
  combine DG0 materials and CG1 displacement in one VTX writer. No VTX execution
  is claimed here; the initial output path is tested XDMF/HDF5.

No legacy `dolfin`, `VectorFunctionSpace`, old quadrature representation flags,
or high-level linear-solver objects appear in the explicit numerical path.
