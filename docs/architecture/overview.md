# Architecture and critical design review

This milestone is a 1D **reduction of 3D solid elasticity**, with independent P
and S experiments. It is not a rod model or acoustic pressure solver.

The requirements are correct physics, reproducible experiments, a single backend,
and quantitative verification. Design choices: immutable Pydantic configuration,
aligned uniform interval meshes, CG1/DG0, central differences, distributional FE
point loading, and partitioned receiver output. Degree is explicit but only 1 is
accepted. Unsupported physics is rejected, never silently approximated.

Dependency direction: CLI / Python / future Qt -> SimulationConfig -> Simulation
lifecycle -> spatial operators, time integration, cached points, results and output.

`config.py` contains serializable values only. `materials.py` converts material
parameters and maps coordinates independently of the constitutive reduction.
`fem1d.py` owns mesh, UFL forms, diagonal mass, PETSc stiffness and endpoint
constraints. `timestepping.py` consumes local arrays and an operator callable;
it knows nothing of UFL, CLI or output. `points.py` localizes once on owned cells,
elects one owner and caches P1 basis weights. `simulation.py` composes these pieces.
`results.py` names axes and supports optional Xarray. `output.py` writes distributed
snapshots and rank-local receiver shards. No GUI dependencies enter the backend.

## Corrections to the proposed design

* A scalar solve represents **one polarization**, P or S. Plane strain in 2D
  requires vector displacement with 3D Lamé coefficients, not a dimension switch.
* Interfaces must coincide with mesh vertices. Midpoint assignment across an
  unresolved interface would change the scientific model with refinement.
* Mass lumping is an element-specific method, not a generic switch.
* A point load in this planar reduction is force **per transverse area** (Pa),
  not the total force in N of a 3D point source.
* Stability validation uses a proven interval-P1 bound and a safety margin.
  Wavelength warnings are separate empirical accuracy advice.
* MPI results contain locally owned receivers, including empty shards. No full
  field is gathered. Replicated receiver coordinates and one-time rank election
  are an explicit initial scaling limit.

## Lifecycle

`Simulation(config, comm=None)` defaults to MPI.COMM_WORLD. `build()` creates
inspectable `operators`; `run()` releases resources if it built them itself.
Use `with Simulation(config) as sim:` to inspect operators and repeat runs before
collective context exit. Every rank must enter build/run/close with the same config.
Callbacks execute on all ranks; frontend callbacks may filter rank 0. They must
not raise asymmetrically or invoke mismatched collectives.

## Transfer to 2D/3D

Config/runtime separation, time functions, explicit integrator, metadata, events
and named results transfer. Material conversion transfers; spatial assignment
needs mesh tags or intersection checks. Interval mesh creation, P1 point weights,
scalar modulus, endpoint impedance and the CFL proof do not transfer. Replace
these with vector forms, basis evaluation/ownership, normal/tangential boundary
operators and an element-specific spectral bound. Higher order needs a new
validated lumping method before enabling a degree.

## Initial numerical risk review

Hazards: confusing constrained compression with rod extension; invalid high-order
lumped masses; wrong displacement reflection sign; source strength proportional
to h; duplicate point loads at MPI boundaries; stale ghosts; wrong initial
acceleration; comparing ordinary energy with the conserved staggered invariant;
using a Ricker displacement peak for force travel time (velocity is delayed Ricker).
These determine the tests, rather than being deferred to optimization.
