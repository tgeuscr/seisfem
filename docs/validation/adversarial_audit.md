# Adversarial audit of the v0.1.0 1D laboratory

Date: 2026-09-10. Assessment: **TRUSTWORTHY AS A 1D REFERENCE LABORATORY**, within
the assumptions below, including the small validation correction on this branch.
No fundamental physical, assembly, integration or distributed numerical defect
was demonstrated. This judgment is based on independent checks, not test count.
The repository is ready to **begin** the vector plane-strain milestone; none of
the scalar element, boundary or CFL algorithms is thereby verified in 2D.

## A. Baseline and scope

Branch: `audit/adversarial-1d`. Original commit/tag:
`b4c4045d0d75e3d2b3789cf37a28fb2b08f43535` (`v0.1.0`). Working tree initially clean.
No baseline history, tag or commit was changed. All 12 source modules, all original
test files and helpers, the analytical example, README, theory/numerics/user notes,
scientific review, measured report and raw results, six ADRs, packaging metadata,
license/citation placeholders and both environment locks were inspected before edits.
Generated figure/measurement provenance was checked against its script and report.
Ignored caches and previous output directories were not treated as source evidence.

`python -m pytest -ra -s`: **33 passed in 9.00 s**, including the existing two-
and four-rank subprocesses. Raw output: [adversarial_baseline.txt](adversarial_baseline.txt).
Active Python: `/home/m/packages/anaconda3/envs/fenicsx0.11/bin/python`, 3.14.6;
DOLFINx/Basix 0.11.0, UFL 2026.1.0, NumPy 2.5.3, real-double PETSc 3.25.5,
MPICH 5.0.1. `python -m pip check`: no broken requirements.

The conda lock contains explicit build URLs and checksums; the pip supplement has
exact versions without artifact hashes. `environment.yml` is a solver intent,
not the exact lock. This audit used the installed matching numerical stack; it
**did not recreate the locked environment**, independently certify its full
transitive closure, or verify every remote artifact checksum. The historical
fresh-environment claim remains supported by the previous run record, not this run.

The supplied YAML/CLI example and `analyze_layered.py` were rerun in a new temporary
output directory. [The new measurements](adversarial_example.json) reproduce the
reported peaks. Existing example outputs and the historical figure were preserved.

## B. Independent methodology and derivations

### Continuum mechanics

Starting from balance of momentum, tension-positive stress and positive-up z,
`rho u_tt = div(sigma)+f`, with `epsilon_ij=(u_i,j+u_j,i)/2` and
`sigma_ij=lambda epsilon_kk delta_ij+2mu epsilon_ij`:

* Longitudinal `u=(0,0,w(z,t))`: only epsilon_zz=w_z is nonzero. Stress is
  diag(lambda w_z, lambda w_z, (lambda+2mu)w_z). The z equation is
  `rho w_tt - partial_z[(lambda+2mu)w_z] = f_z`.
* Transverse `u=(s(z,t),0,0)`: epsilon_xz=epsilon_zx=s_z/2, trace zero.
  sigma_xz=sigma_zx=mu s_z. The x equation is
  `rho s_tt - partial_z[mu s_z] = f_x`.

Thus A=lambda+2mu or mu, respectively. Young's modulus and bulk modulus alone
would both be wrong in the longitudinal operator. Source inspection finds neither
substitution. The strain energy density is
`mu epsilon_dev:epsilon_dev + (lambda+2mu/3)(tr epsilon)^2/2`.
Stable 3D solids require rho>0, mu>0 and lambda+2mu/3>0. Requiring lambda>0
would be wrong. Both polarizations have kinetic/potential energy per transverse
area `integral(rho q_t^2+A q_z^2)/2 dz` and speed sqrt(A/rho).

Conversion follows from A_P=rho Vp² and A_S=rho Vs²:
mu=rho Vs² and lambda=rho(Vp²-2Vs²). Positive bulk is equivalent to
Vp²>4Vs²/3. Tests include lambda=0, lambda nearly zero, admissible negative
lambda, bulk only 1e-12 above its boundary, and mu=1e-24 Pa. No fluid limit is
claimed. At near-machine degeneracy the sign of a computed bulk modulus is
roundoff-sensitive. A dense 6x6 eigensolver also loses tiny shear eigenvalues
when lambda/mu is enormous; the audit uses the analytic volumetric/deviatoric
split there instead of calling eigensolver roundoff a material defect.

### Weak form, interfaces and local matrices

Integrating `-(A q_z)_z v` gives `integral A q_z v_z - [A q_z v]`.
Outward traction is t=n A q_z, n=-1 at the lower endpoint and +1 at the upper.
The resulting positive stiffness agrees with `fem1d.py`. Material functions
multiply the derivative product; UFL does not differentiate DG0 properties.
CG continuity plus cancellation of internal boundary terms gives a welded
interface: q and A q_z continuous in the weak continuum problem. This is not
pointwise equality of the piecewise-constant FE stress at every transient step.
Free endpoints have zero natural traction. Projected zero Dirichlet DOFs evolve
the free principal subproblem of the unmodified stiffness matrix.

With xi in [0,1], phi=(1-xi,xi), direct polynomial integration gives

```
M_e = rho*h/6 [[2,1],[1,2]]
D_e = rho*h/2 diag(1,1)
K_e = A/h [[1,-1],[-1,1]].
```

Two cells give nodal masses `(rho_left*h/2,
(rho_left+rho_right)*h/2, rho_right*h/2)`. An interface node receives **both**
cell contributions. Full matrices and every owned mass entry were checked
against a separately written cell loop, including rho contrast 1e8 and speed
contrast above 1e3, both modes, endpoints and MPI partitions. A one-element
integral test exercises DOLFINx directly because the public schema requires at
least two cells; it is not misrepresented as a one-cell production run.

### Time stepping, energy and stability

For diagonal D,C, substitution of centered acceleration and velocity into the
matrix ODE gives

```
(D+kC/2) U[n+1] = 2D U[n] - (D-kC/2) U[n-1] + k²(F[n]-K U[n]).
a0 = D^-1(F0-K U0-C V0), U[-1]=U0-k V0+k² a0/2.
```

This matches the code, including nonzero initial velocity and constraint
projection. A forced quadratic two-DOF solution checks initialization and all
centered outputs independently, including nonzero diagonal damping and coupled
stiffness. An analytic damped oscillator checks physical exponential decay and
second-order accuracy. A separate public two-cell problem has D=3, K=24 at its
only free node: initial acceleration, first displacement, nodal force balance
at **every** sample, interpolation, and final auxiliary-state velocity are checked
with locally computed forcing. This detects frontend-loop source-time mistakes
that an integrator-only test cannot detect.

Dot the recurrence with `(U[n+1]-U[n-1])/2`. Symmetry of K telescopes the cross
potential, giving

```
E[n+1/2] - E[n-1/2] = k v[n]^T F[n] - k v[n]^T C v[n]
E[n+1/2] = 1/2 v_half^T (D-k²K/4) v_half + 1/2 u_mid^T K u_mid.
```

The second expression is used as the audit oracle; production uses the
cross-time expression. They are algebraically equivalent, as an identity test
should be, but are computed from different terms. The oscillator experiment
separately shows ordinary integer-time energy varying **8.009%**, while the
staggered invariant varies **3.61e-15** relatively. Thus invariant preservation
is evidence of correct integration algebra, not exact continuum energetics.
The forced/damped work test verifies both work signs, and the physical damped
oscillator supplies evidence beyond the discrete identity.

P1 interval K has zero row sums and nonpositive off-diagonals. Gershgorin yields
lambda_max(D^-1 K)<=2 max(K_ii/D_ii), hence
`k < sqrt(2/max(K_ii/D_ii))`. On uniform h the ratio is bounded by
`2 c_max²/h²`, including density jumps. Principal restriction to free DOFs
cannot increase the maximum generalized eigenvalue. Centered diagonal damping
has nonnegative loss in the same energy identity. These arguments match both
configuration and assembled bounds; they are specifically interval-P1 arguments.

Dense symmetric eigensolutions of `D^-1/2 K D^-1/2` confirm the bound across
18 mode/material/constraint/resolution combinations. Highest-mode stepping
remains bounded at 0.5 and 0.999 of the **true** spectral limit and grows by more
than 1e8 within 300 steps at 1.001. In a fixed two-cell example the sufficient
bound is 0.5 s and the true limit is 0.70710678 s: 1.001 times the sufficient
bound remains stable. Exceeding a sufficient bound does **not** prove instability.
Experiments above configured safety deliberately use the low-level integrator;
the public configuration continues to reject them.

### Propagation, source, reflection and absorption

An outgoing increasing-z wave satisfies A q_z=-Z q_t; a decreasing-z wave has
A q_z=+Z q_t. At either outer boundary t=n A q_z=-Z q_t, giving positive damping
and outward energy flux Z q_t². For an incident wave crossing an interface,
velocity continuity gives 1+R=T, and internal stress continuity gives
Z1(1-R)=Z2 T. Solving these two equations yields
`R=(Z1-Z2)/(Z1+Z2)`, `T=2Z1/(Z1+Z2)` for displacement/velocity, and
`R²+(Z2/Z1)T²=1`. Fixed/free reflected displacement and velocity have signs -/+.
Stress-reflection coefficients use a different convention. No depth sign
conversion or pressure convention appears in the production path.

The point load is the weak functional B r(t) v(zs). Partition of unity and
linear reproduction imply sum F_i=B r(t) and sum z_i F_i=zs B r(t).
The source magnitude is Pa, since this planar reduction is per transverse area.
Direct hat-function tests check **each weight**, not just the sum and moment.
Receiver expectations use sorted-node interpolation and a changing nonlinear
nodal field, plus the separate DOLFINx Function.eval path. Using the same PointMap
for force and receiver reciprocity would not, by itself, be independent evidence.

The homogeneous causal Green function gives velocity B/(2Z) r(t-distance/c).
An interface-coincident force sends equal interface velocities B/(Z1+Z2) r(t)
into the two sides. Both formulas are tested without production wavelet/speed
helpers. The public simulation starts forcing at t=0; infinite Ricker tails are
not causal initial data. Pulse comparisons here lie after the causal front,
with shifts suppressing the startup tail. No smooth-source convergence claim
is made at the point singularity.

The new contrasts use (rho,Vp,Vs)=(1730,1870,930) and (2670,3210,1420), then
reverse them. R is ±0.45194735 for P and ±0.40414421 for S; maximum velocity
waveform error is **0.599%**, including reflected/transmitted paths. The original
P/S pair had the same impedance ratio for both modes; the audit pair does not.
The near-equal P case has R=-4.99975e-5 and reflected error **5.74e-6 of incident
amplitude**. That absolute normalization remains meaningful near zero reflection;
it does not establish high relative accuracy of a vanishing reflected phase.
The interface-test source is z=1000.37 m. Phase windows are ±0.09 s at 8 Hz.
Direct and interface-reflected phases are
separated by at least 0.436 s. Finite-boundary paths were checked: returning
main-pulse windows are separated by explicit test assertions; small discrete
absorber tails remain. An initially tried source at 800.37 m allowed a lower
endpoint return into the reverse-contrast reflection window. That audit fixture
was corrected before the final run, even though its small residual had passed
the amplitude gate. This was a flaw in the new audit reference geometry, not
a production-code defect.

Nodal, near-left, interior and near-right source fractions give comparable
waveforms. Across h=4,2,1 m their errors are approximately 1.74%, 0.429%, 0.104%.
The total FE force is exactly mesh-independent to roundoff; the propagated
waveform is mesh-independent only asymptotically. Tests distinguish these claims.

Both absorbing endpoints were tested at 6/12 Hz, P/S, CFL 0.2/0.6, h=4/2/1 m
and rho=2370. Reflection falls roughly fourfold per halving of h. Finest-grid
residuals range from **0.00640% to 0.1022%** of incident amplitude; opposite
endpoints agree to roundoff-scale error. A separate rho=1130, fixed h=2 m,
12 Hz sweep gives 0.10255%, 0.06837%, 0.02963% at CFL 0.2,0.6,0.85.
Reducing dt alone does not necessarily reduce discrete reflection: spatial and
temporal dispersion can cancel. This supports an exact **continuous** 1D
impedance with a discretization-dependent residual, not an exact discrete ABC.

### Spatial and temporal convergence

The original spatial test really integrates the FE field, with UFL quadrature
degree 8, rather than comparing nodes alone. Its three levels are consistent
with second order; no evidence of cherry-picked failure removal was found in
its code, although historical choices cannot be reconstructed from tests alone.

The independent spatial test uses two modes, four refinements N=16/32/64/128,
final time 0.173 s, dt=1e-5 s, and 12-point NumPy Gauss-Legendre integration of
piecewise-linear interpolation. Rates are **1.97247, 1.99311, 1.99828**.
Comparison to the exact matrix ODE solution bounds temporal contamination below
1e-4 of the spatial error on every level. The reference uses no production UFL
expression, and includes between-node error. No heterogeneous smooth-solution
claim is manufactured across incompatible interface fluxes.

Separate second and third modes on N=12/18 use dense generalized eigensolutions
as temporal references, checking the original closed-form frequency independently.
Four timestep levels give rates **2.00000–2.00038**. The continuous damped
oscillator gives **2.00002–2.00024**. Existing 1.95–2.05 order tolerances were
retained; no baseline tolerance was changed.

## C. Claim-to-evidence matrix

Paths below are relative to the repository. Baseline operator tests are in
`tests/analytical/test_operators_energy.py` (O); baseline pulse tests are in
`tests/analytical/test_pulses.py` (P); frontend/MPI tests are in
`tests/integration/test_frontends_mpi.py` (I). New tests are in
`tests/audit/test_independent_numerics.py` (N), `test_independent_waves.py` (W),
and `test_distributed_io.py` with `mpi_probe.py` (D).

| Claim | Production | Baseline executable evidence | Independent audit evidence/reference | Independence and remaining common-mode risk |
|---|---|---|---|---|
| P speed | config.Isotropic, fem1d, simulation | P: homogeneous_speed_and_amplitude | W: interface_both_contrast_directions; local retarded pulses | Prescribed Vp independent; peak timing alone only sample resolution |
| S speed | same, S modulus | P: homogeneous_speed_and_amplitude | W: S contrast cases and interface force | Prescribed Vs independent; no vector S polarization test |
| P amplitude | PointMap.unit_load, simulation | P: homogeneous_speed_and_amplitude | W: negative force, fractions, two outgoing impedances | B/(2 rho c) derived locally; propagation dispersion remains |
| S amplitude | same, mu | P: homogeneous_speed_and_amplitude | W: S contrasts and interface force | New pair has different P/S impedance ratios |
| Source normalization | points.unit_load | O: assembled_operators_and_source | N: point_load_each_weight; D: actual partition vertices | Independent hats and moments; not only source/receiver reciprocity |
| Source position independence | points, fem1d mass | P: source_and_waveform_refinement at one source | W: four fractions at three meshes; N/D each weight | Exact weak normalization versus asymptotic wave accuracy explicitly separated |
| Receiver interpolation | points.evaluate | O: linear field and Function.eval | N/D: nonlinear fields, duplicated coordinates, repeated ghost refresh | np.interp and Function.eval; underlying DOLFINx space shared |
| Spatial convergence | fem1d, timestepping | O: spatial_convergence, FE UFL integral | N: spatial_order_independent_gauss_multimode | Independent quadrature/exact continuum modes; temporal error quantified |
| Temporal convergence | timestepping | O: temporal_convergence, formula omega_h | N: temporal_order_dense_eigenproblem; damped oscillator | Matrix eigensolve/continuous ODE, not recurrence-derived frequency |
| Discrete energy | timestepping.evaluate | O: energy_invariant | N: oscillator_physical_energy_is_not_the_invariant | Identity oracle deliberately equivalent algebra; physical-energy counterexample separate |
| Damping energy balance | damping, timestepping | O: impedance_energy_balance | N: forced_damped_energy_work_independent; damped oscillator | Work and midpoint invariant local; identity alone cannot prove damping physics |
| Fixed reflection | fem1d.fixed | P: endpoint_reflection, upper P | N: constrained tiny-system values; mechanics derivation | Baseline waveform independently signed; lower/S reflection waveform not separately swept |
| Free reflection | natural weak boundary | P: endpoint_reflection; O: free_rigid_translation | N: hand matrices/free spectra | Positive sign derived independently; lower/S waveform gap as above |
| Impedance absorption | fem1d.damping | P: endpoint_reflection at one upper endpoint | W: absorber_both_endpoints_refinement and timestep_sweep | Both directions and several resolutions; continuous exactness not discrete exactness |
| Interface R | DG0/modulus, CG1 stiffness | P: layered_interface | W: both contrast signs, different P/S ratios, near-equal limit | Two-equation traction solve; independent local wavelet |
| Interface T | same | P: layered_interface | W: both signs and source-at-interface | Velocity/displacement convention, weighted flux check; no stress receivers |
| CLI/Python equivalence | cli, Simulation | I: cli_api_output_and_snapshots | D: direct configuration, YAML/Python, YAML/CLI and inspect | Same backend is intentional; equivalence does not independently verify physics |
| Serial/MPI equivalence | ghost assembly/action/sampling | I: mpi_matches_serial, 2/4 ranks | D: actual_partition_edges, 1/2/3/4 ranks, all quantities | Rank-local arrays matched to serial; separate hand action/mass guards shared error |
| Unique receiver ownership | PointMap election | I: counts and empty shards | D worker: discovered shared vertices, duplicates, sparse and empty maps | Owned-cell connectivity proves actual shared containment; exact count by ID |
| Material-interface behavior | materials.properties_at, DG0, mass/K | O: jump masses; unit: unresolved layer rejection | N: hand matrix contrast sweep; W: interface source; D: material readback | Cell integrals independent of nodal assignment; FE flux is weak, not pointwise exact |
| Source timing/final sampling | simulation loop, stepper | P: pulse tests; I: final time | N: public_source_timing_all_quantities_and_final_state | Local nodal force balance at every sample detects time-shift mutation directly |
| Stability bound | config.stable_dt, fem1d bound | config rejection; no baseline dense spectra | N: dense spectra and high-mode instability; sufficient-vs-true test | Independent eigensolver; no transfer to vector off-diagonal structure |
| I/O values and metadata | output, results | I: mesh count, times, trace shards | D: external h5dump geometry/topology/DG0/CG1 values, interruption, identity fixtures | Independent read path; shared HDF5 library, not a second format implementation |

The plotting reference imports production `Isotropic.speed` and `ricker`; it is
**not an independent validation gate**. The original pulse tests import `ricker`
but use local c, Z and coefficient formulas. Their displacement/acceleration
references and the wavelet's FFT/root tests provide some independent protection.
The original consistent-mass row-sum comparison uses another UFL form and is an
identity check; its manually expected nodal masses are the stronger independent
part. The audit preserves useful scientific duplication instead of factoring
expected formulas into production helpers.

## D. Findings, severity and disposition

| ID | Severity | Evidence and consequence | Disposition |
|---|---|---|---|
| F1 | Minor | Finite positive Lamé/density inputs can yield zero or infinite computed squared speeds. rho=1e300, lambda=mu=1e-300 gives zero speed; the reciprocal scale gives infinity. The former can trigger division by zero in configuration stability. These are extreme, non-geophysical inputs, not evidence against reported waves. | Demonstrated two failing tests before patch; reject nonfinite/nonpositive derived squared speeds in material validation. |
| F2 | Observation | Baseline interface sweep has one impedance ratio, absorber sweep one endpoint, and MPI tests name partition candidates without asserting actual shared incidence. | New contrast, both-endpoint, and discovered-partition tests close those specific gaps. No claim that all contrasts/decompositions are covered. |
| F3 | Observation | Source/receiver maps share weights; pulse/plot references share production wavelet code, and plot speed conversion also shares production. | Independent hats, nonlinear receiver oracle, local wavelets, dense matrices and local traction solve added. Existing useful identity tests retained. |
| F4 | Minor | `git_dirty=False` when Git is unavailable is ambiguous if read as proof of cleanliness. Clean/dirty/detached/no-Git fixture demonstrates this behavior. | Document that dirty status is meaningful only with available Git identity; no metadata schema change. Source digest identifies available Python files, not a full archived build. |
| F5 | Observation | At fixed h the measured absorber residual increases as dt decreases; a monotone-dt absorption claim would be false. | Clarify dispersion cancellation and continuous-versus-discrete exactness. Existing theory was broadly correct. |
| F6 | Moderate, documented restriction | Asymmetric progress callbacks or binary I/O exceptions can strand other ranks in collectives. No general MPI fault recovery exists. Source inspection confirms collectives follow callbacks and some binary I/O is outside Python error propagation. | Existing restriction retained and made explicit here. Symmetric serial interruption tested; rank-asymmetric fault injection not attempted. This restricts safe API use, not the verified numerical experiments. |
| F7 | Minor | Offline example analysis has a fixed ±0.07 s peak window, overwrites duplicate IDs in a dictionary, and does not enforce a complete manifest before reading shards. | Document example-only analysis assumptions. No plot is treated as a scientific acceptance test; no new analysis framework added. |

No Critical or Major finding was established. F6 was already acknowledged in
the original review; severity indicates the consequence of violating the
collective API contract, not a newly discovered failure of normal MPI execution.

### Defect protocol for F1

Correct numerical behavior requires positive finite *representable* squared
speeds in addition to the mathematical solid positivity conditions. The new
`test_unrepresentable_derived_speed_rejected` failed twice before modifying
production: [failure record](adversarial_defect_before.txt). The correction is
five lines in `config.Isotropic.physical`; ordinary moduli, valid negative
lambda and small positive shear remain accepted. The focused test and full
then-current suite passed after patch (**74 tests**). The final larger suite
is recorded below. No mass, stiffness, source, receiver or stepping formula changed.

## E. New tests, test quality and mutation evidence

The new files group mathematical identities, independent numerical verification
and distributed integration rather than implying that each parameter is a new
scientific claim:

* N: one-element integration (2 cases), hand mass/stiffness and spectral tests
  (18), individual source weights/receiver values (8), admissible material edges
  (5), unrepresentable-speed rejection (2), forced polynomial, energy distinction,
  forced/damped work, dense temporal modes (2), independent spatial convergence,
  public source/final output, sufficient-versus-true limit, damped oscillator.
* W: both-sign P/S interfaces (4), interface force (2), near-equal impedance,
  source fractions/refinement (3), two-endpoint absorber sweeps (4), fixed-mesh
  timestep sweep.
* D: actual partitions and complete field readback (1/2/3/4 ranks), interrupted
  output protection, clean/dirty/detached/unavailable source identity.

These are **65 additional collected cases** (44 N, 15 W, 6 D); the MPI worker
contains multiple assertions on genuinely distributed state. Parametrization is
coverage of different inputs, not multiplication of independent physical proofs.
No tests were added solely for formatting or other reversible cosmetic changes.

Three actual mutations were made only in temporary copies of the package,
loaded through PYTHONPATH. The working production source was never mutated.
The transformations and subprocess results are retained in
[adversarial_mutations.json](adversarial_mutations.json) and
[raw mutation failures](adversarial_mutation_runs.txt):

| Mutation | Original tests | New test | Interpretation |
|---|---|---|---|
| Evaluate `force(time + cfg.time.dt)` instead of `force(time)` in Simulation | 7/8 pulse tests pass; source refinement fails | Public two-cell source-timing test fails immediately | A single successful pulse trace cannot certify time indexing; the baseline suite does have a refinement backstop. |
| Negate the applied source amplitude | Both homogeneous P/S cases fail | Public two-cell test fails | Signed source amplitude has real baseline protection. |
| Omit mass `scatter_reverse(add)` | Both original 2/4-rank comparisons fail | Actual-partition hand-mass test fails | MPI tests exercise mass ghost contributions; not merely replicated serial computations. |

Additional mutation reasoning (not executed): altering an endpoint mass or
omitting boundary stiffness is caught by full hand matrices; choosing one layer's
density at an interface is caught by its two-sided mass expectation; changing
receiver weights is caught by individual hats and nonlinear nodal interpolation;
duplicate receiver gathering is caught by ID counts; stale ghosts are caught by
three changing-field samples; an S bound using Vp is caught by the existing S
layered run and independent mode-dependent bounds. No exhaustive mutation score
or framework-level coverage claim is made.

Tolerance review: exact matrix, weight and energy checks use roughly 1e-14–1e-13
scale allowances, widened only where many updates/subtractions accumulate.
MPI trace comparison in the new short run uses 3e-12 relative and 2e-12 absolute;
the original long-run 2e-10 scale tolerance is conservative but detects ghost
omission. Baseline pulse peak tolerances (1–2.5%) are much larger than observed
peak errors, while waveform gates are tighter in practice; they should not be
read as achieved precision. The one-step mutation demonstrates why direct timing
checks were added instead of tightening all continuum pulse tolerances.
Near-zero reflected waves use incident-normalized absolute error. No new test
claims exact travel speed from a peak landing on a time sample. No baseline
threshold was relaxed. During audit-test development, a PETSc integer-index type
mismatch and an ill-conditioned dense material eigenvalue oracle were corrected
in the audit code; neither was a production defect.

## F. MPI, I/O, frontends and reproducibility details

The new 12-cell mesh has actual shared vertices `{0}` at 2 ranks,
`{-1/3,1/3}` at 3, and `{-1/2,0,1/2}` at 4. The worker discovers these from
owned-cell connectivity, verifies lowest-containing-rank election, exact global
ID counts, local nodal masses, stiffness action, source total/moment, interface
and off-node points, duplicated coordinates, and changing ghost values. At exact
vertices, either adjacent cell gives the same CG1 value/load; lowest-rank choice
is therefore harmless. Infinitesimal interface offsets remain limited by double
precision and geometric containment tolerances. These are scalar continuous
receivers, not one-sided stress/strain probes.

A one-point map proves ranks with no local receiver; an empty map exercises all
ranks without receivers. Original 4-rank tests also write actual empty NPZ shards.
An additional manual 2-cell/4-rank run completed with empty cells/receivers on
some ranks (all result shapes `(11,0,1)`). No multi-node or large-rank inference
is drawn. The MPI comparisons are launched from a single pytest process using
the matching launcher. Timeouts bound the subprocess tests; the worker aborts its
test communicator on assertion failure to avoid orphaned collective waits.

The source code uses owned arrays, forward refresh before stiffness/sampling,
one reverse accumulation for assembled mass/load, and global max for the bound.
No numerical rank-0 special case was found. Metadata/config filesystem actions
are rank-filtered with error exchange; XDMF construction/write/close and PETSc
destruction are collective. Incompatible configuration is allgather-checked.
Callbacks must be present consistently and must not raise asymmetrically or
invoke different collectives. Binary library faults remain outside fault recovery.

External `h5dump` reads XML-referenced mesh geometry/topology, all four DG0
material arrays, and every strided CG1 displacement dataset. Values are compared
by coordinates to expected materials and nonzero nodal receiver histories,
including the final off-stride snapshot. This is a data-value check, not a
file-existence check. HDF5 is still the reader/writer library on both paths;
this does not independently certify HDF5 itself. The locked HDF5 package supplies
`h5dump`; missing it is a failed prerequisite, not a silently skipped readback.
Tests compare direct Python, YAML/Python and YAML/CLI traces bit-for-bit in serial,
then compare all three quantities against MPI, and inspect normalized CLI defaults.

A controlled serial callback exception leaves status `running`, closes snapshots,
releases operators and writes no completed receiver shards. Reusing that directory
raises without changing any bytes. `running` cannot distinguish an active run from
an interrupted run; there is no restart or transactional archive guarantee.
Final completion is written after shards, but abrupt process/node/filesystem
failure is not simulated. Metadata tests inspect physical config, package/PETSc
versions, scalar type, mesh/DOFs, identities, times, names and coordinate ordering.
FEM degree, dt, material/source/receiver definitions are in normalized `config`.

A miniature temporary Git repository tests clean, dirty, detached and no-Git
states without changing the real branch. The source hash covers package Python
files and names; it excludes dependencies, binary artifacts, examples, tests and
other possible future assets. Commits plus a dirty flag do not archive a dirty
diff. Dependency locks and metadata support reproduction only when the actual
checkout, dependency artifacts and inputs are preserved.

The geometry ownership API was cross-checked against the versioned
[DOLFINx 0.11 geometry documentation](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.geometry.html):
explicit owned-cell entities matter because an unrestricted bounding tree can
include ghosts. Runtime behavior was then checked independently from connectivity.
The local 0.11 API and actual HDF readback, rather than unexecuted documentation,
are the evidence for output correctness.

## G. Architecture classification and remaining uncertainties

| Classification | Components | Audit judgment |
|---|---|---|
| Safe to retain | Configuration/runtime separation; source-time function; Progress events; central-difference owned-array contract | Dimension-independent concepts. Integrator requires positive diagonal mass/damping, fixed-zero constraints, symmetric stiffness and consistent collective calls. |
| Retain with modification | Isotropic conversion, Simulation lifecycle, result axes, metadata/output | Conversion is 3D-isotropic already. Simulation hard-codes interval backend, scalar P/S and zero initial data. Results have a one-element component axis and scalar component name; units/source schema/output representability need dimensional review. |
| 1D-specific | fem1d, properties_at layering, PointMap weights, interval domain/mesh config, scalar source position and Pa amplitude, endpoint BCs, stable_dt | Keep explicitly isolated. `(N,3)` padded search points do not make interval interpolation multidimensional. Geometry slot 0 is physical z only by this contract. |
| Architectural hazard if copied | Scalar propagation modulus; off-diagonal-sign CFL proof; diagonal endpoint impedance; row-sum strategy treated as generic; displacement/velocity R/T used for traction | A vector formulation needs strain tensors, coupled components, new spectral bounds, boundary normal/tangential operators and element-specific lumping evidence. Current docstrings/ADRs mostly already warn about these. |

No speculative refactoring was justified. Remaining limits include: no 2D/3D
vector patches or mode conversion, oblique incidence, stress receivers, imported
or nonuniform meshes, unaligned interfaces, extreme-scale floating-point accuracy
guarantee, physical attenuation, higher-order mass, multi-node MPI, many-receiver
scaling, asynchronous callback safety, or fault-tolerant output. Fixed/free
reflected waveforms were not newly swept at the lower endpoint or for S; the
existing upper-P tests, matrix constraints and mechanics supply narrower evidence.
Density/velocity contrast matrices were tested far beyond the pulse material
range, but a long-duration physical waveform sweep at those extreme contrasts
was not performed. Floating-point underflow/overflow in every possible config,
source or accumulated state is not comprehensively guarded by the small F1 fix.

The source/interface tests verify weak loads and observed waves; they do not
establish smooth error rates at singular forcing. Integer-time physical energy
and exact phase speed are not certified by the discrete energy identity. The
spatial/temporal tests are homogeneous smooth verification problems, not empirical
geological validation. The license and authorship placeholders remain unchanged.

## H–J. Trust decision, 2D readiness and diff summary

**TRUSTWORTHY AS A 1D REFERENCE LABORATORY** for small-strain isotropic stable
solids, one scalar P or S polarization, positive-up coordinates, uniform affine
P1 intervals, aligned welded layers, fixed timestep within the supported bound,
real-double arithmetic at sensible physical scales, the tested point-force
interpretation, and collective MPI use with well-behaved callbacks. Numerical
resolution must still be verified for each scientific experiment. No validation
against observations or accuracy guarantee for arbitrary input is implied.

**Ready to begin 2D plane strain**, keeping this 1D branch as a reference and
retaining the original scientific-review gates for vector mechanics, component
mass, strain-energy patches, independent spectra, manufactured convergence and
distributed vector sampling. This audit implements no 2D or new solver features.

Diff categories:

* Tests: four new audit Python files; 65 new collected cases plus MPI worker.
  Existing tests and their thresholds are unchanged.
* Production fixes: five-line material-validation guard for demonstrated
  unrepresentable squared speeds. No numerical algorithm changes.
* Documentation: this audit, raw baseline/failure/final/mutation records, example
  rerun measurements, and targeted links/clarifications in existing notes.
* Refactoring: none.

Final verification: **98 passed in 93.72 s**, including all 33 original cases and
65 audit cases. After strengthening the readback checks to require every material
attribute, exact interval connectivity and duplicate-safe ID counting, all six
I/O/MPI cases passed again in 5.15 s. Ruff check and format-check pass on all
source/tests/examples; `git diff --check` passes. Commands and results are recorded
in [adversarial_test_run.txt](adversarial_test_run.txt).

All audit edits remain uncommitted on `audit/adversarial-1d`; HEAD and `v0.1.0`
still identify the original baseline commit. No feature implementation,
performance refactor, branch switch, reset, squash, merge or tag rewrite occurred.
