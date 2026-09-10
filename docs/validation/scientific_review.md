# End-of-milestone scientific review

This is a verified interval-P1 laboratory, not yet a production 3D seismic code.
The [measured report](results.md) specifies evidence and resolutions. Verification
means agreement with mathematical reference problems; no field-data validation
or claim of geological predictive accuracy is made.

This is the original milestone review. The subsequent
[adversarial audit](adversarial_audit.md) supplies independent matrix/spectral,
interface, source-timing, partition-ownership and full field-value readback
evidence, with its own findings and remaining limits. In particular, its readback
tests supersede the field-value gap listed below; other unverified regimes remain.

## Implemented

A single DOLFINx 0.11 backend runs homogeneous/layered 1D P or S from immutable
YAML/Python configuration. It constructs a distributed mesh, CG1 space and DG0
rho/lambda/mu, assembles lumped mass and PETSc stiffness once, advances explicit
central differences, injects a normalized FE point force, and samples displacement,
velocity and acceleration. Fixed, natural free and 1D impedance endpoints work.
CLI validate/run/inspect/mesh/info, optional Xarray conversion, frontend-neutral
events, rank-filtered logging, collective strided XDMF/HDF5 fields, receiver shards
and a completion manifest are implemented. Six ADRs explain consequential choices.

## Verified

Actual FE mass/stiffness identities, source total strength/first moment, cached
receiver values versus DOLFINx evaluation, material conversion and invalid config
rejection. Quantitative homogeneous P/S travel times and all three receiver
quantities; two-layer direct/reflected/transmitted signed velocity amplitudes;
refinement of a point-source waveform; smooth standing-wave L2 spatial convergence;
second-order time convergence against an exact semidiscrete mode; the staggered
energy invariant including nonzero initial velocity; free rigid translation;
fixed/free reflection signs; 1D impedance residual reflection and its discrete
energy-loss identity. Serial, two-rank and four-rank traces/energy agree, with
interface source insertion and empty receiver shards. CLI and Python traces match
exactly in the tested case. XDMF mesh readback and snapshot time strides are checked.

## Not yet verified

No 2D/3D fields, oblique incidence, mode conversion, radiation patterns, stress
receivers or manufactured vector forcing. No high-order or nonuniform-grid
convergence, extreme density/velocity contrast sweep, broad-band long-distance
accuracy survey, multi-node fabric or many-rank scaling. XDMF output is exercised
but there is no independent full field-value readback test. VTX was investigated
from official APIs, not executed. A fresh headless lock is tested separately from
the maintainable YAML environment; the latter is not an independently certified
solver resolution across platforms. Plotting is diagnostic; it is not a test gate.

## Numerical assumptions

Small-strain, isotropic, lossless **solid** elasticity, rho>0, mu>0 and positive
3D bulk modulus. One polarization, infinite uniform transverse extent, per-unit
area energetics and force amplitude in Pa. Positive-up z is explicit. Interfaces
are welded, horizontal and aligned to uniform affine interval vertices. Initial
data are zero for public configured runs. Fixed dt, P1, real double PETSc and
central differences are enforced. Ricker forcing starts at t=0 with a tiny
nonzero tail; f_max is a bandwidth of interest, not strict band limitation.

## Technical debt

Receiver setup replicates all positions and elects ranks with an allreduce; only
traces are distributed. Histories are buffered until completion, then written as
one NPZ per rank. The time loop allocates several owned arrays per step and performs
three ghost updates at each receiver sample, even when requested quantities are
sparse. Build/run/close is collective but unexpected asymmetric callbacks or
binary I/O exceptions can still hang an MPI job; there is no fault recovery.
Source identity hashes available package files and records Git status, but does
not archive them. License, authors and permanent citation identifiers are pending.
There is no CI service configured or remote repository; local pytest/Ruff/pre-commit
are the executable quality gates.

## Scientific risks

A stable model can still have unacceptable accumulated phase error: the original
S-wave mesh passed the wavelength screen yet failed the waveform comparison.
The tests were refined, not relaxed. The scalar interface signs and source units
must not be copied blindly into traction, moment-tensor or 3D total-force code.
High-order lumping and multi-dimensional impedance cannot inherit current proofs.
Point singularities, near-interface receivers and unresolved interfaces require
separate analysis. Free/fixed and absorbing boundaries have very different
physical meanings; future GUI controls must retain the schema distinctions.

## Performance risks

PETSc stiffness action and owned diagonal updates distribute correctly, but a
Python-assembled sparse matrix and memory-bound vector temporaries may dominate
at millions of DOFs. Receiver setup is O(ranks*receivers); buffered output memory
is O(samples*local receivers*quantities). NPZ file counts and collective energy
reductions can burden large jobs. XDMF is appropriate for present low-order fields,
not a general high-order output path. No timing here establishes strong scaling.

## Architecture assessment

Config/runtime separation, material conversion independent of layering, spatial
forms independent of time integration, named results and frontend-neutral events
survived end-to-end testing. Keep them. Keep the interval-specific module visibly
specific. Do not turn `IntervalOperators` into a dimension-flag manager: a vector
elastic operator should expose its own tested mass, stiffness and constraints.
The callable explicit update contract is adequate for another explicit scheme;
implicit methods need a solver contract. Replace cached P1 weights with a tested
Basix/UFL point-evaluation mechanism and routed point IDs before large vector runs.
Do not add GUI widgets, generic factories or inheritance solely for future features.

## Concrete next milestone: 2D plane-strain P-SV

1. Add a discriminated rectangular-domain configuration and a vector formulation
   with u=(u_x,u_z), epsilon=sym(grad u), and sigma=lambda tr(epsilon) I+2mu epsilon.
   Retain 3D lambda, including the nonzero sigma_yy implied by plane strain.
   Gate: constant strain patches, rigid translation/rotation nullspaces, matrix
   symmetry, nonnegative stiffness and analytic affine-field strain energy.
2. Choose one affine triangular P1 mesh family. Derive positive componentwise
   mass lumping and an assembled symmetric mass-scaled spectral bound. The scalar
   interval off-diagonal sign proof does **not** apply to vector elasticity.
   Gate: exact total/component masses, positive diagonal entries and sufficient
   bound versus independently computed eigenvalues on small meshes, with density
   contrast, element skewness and constrained boundaries.
3. Add a smooth manufactured vector solution with nonzero forcing, both Lamé
   terms, exact boundary data and nonzero initial velocity. Gate: L2 slopes
   1.8–2.2 and H1 slopes 0.9–1.1 across at least three asymptotic refinements;
   isolated temporal slopes 1.8–2.2; load-work/discrete-energy balance with a
   tolerance justified from accumulated roundoff.
4. Add distributed vector point-force insertion and receiver IDs using current
   DOLFINx ownership/Basix evaluation. Define 2D source units per out-of-plane
   length explicitly. Gate: test-space action, partition of unity/linear moments,
   source/receiver points at vertices/facets/rank boundaries; serial versus 2/4
   ranks for all components. Extend the planar 1D invariant subspace across a
   rectangular strip and compare with the verified 1D P and S solutions.
5. Verify homogeneous radiation before adding absorbers: force orientations
   along each axis, P/S arrival slopes within 1% on a converged mesh, symmetry
   errors decreasing with h, longitudinal/transverse projections and angular
   radiation amplitudes against a specified 2D Green function. Use distances and
   windows that exclude boundaries; distinguish cylindrical from spherical
   spreading and 2D Green-function tails.
6. Add free top and first-order impedance sides/bottom using normal/tangential
   projectors. Boundary damping is a surface operator, not merely independent
   endpoint entries. Select/verify boundary quadrature and local block inversion
   before claiming explicitness. Gate: energy decay; normal-incidence reflected
   amplitude below 1% at a documented bandwidth; quantify oblique residuals
   instead of calling the boundary exact. Test a free-surface reflection/mode
   conversion problem against analytical coefficients.
7. Add an aligned horizontal interface. Gate: normal incidence agrees with 1D
   R/T and travel times for both polarizations; oblique P/S conversion satisfies
   an independently evaluated Zoeppritz system and energy-flux balance. Specify
   below-critical incidence first, then evanescent/critical cases separately.
8. Benchmark routed receiver setup, PETSc matvec, ghost traffic and chunked trace
   output on a multi-node job. Compare XDMF and VTX at fixed output stride and
   verify field values after readback. Gate: no full-field gather, unique receiver
   ownership, bounded memory per rank and reproducible manifests on every frontend.

Only after these gates should the 3D path reuse the vector formulation and
verified element-specific algorithms. New tetrahedral/hexahedral discretizations
will need their own mass, dispersion, boundary and scaling evidence.
