# First narrow 2D plane-strain milestone

Date: 2026-09-11. Branch: `feature/2d-plane-strain-core`.
Base: `main` / `v0.1.1`, commit
`43f3dd70c1875a7dafe37910fc98e7c490e9b5dd`. The baseline branch/tag are preserved.
This is a **verified 2D vector-elastic numerical foundation**, not a complete
2D seismic Simulation/CLI workflow. It is ready for a second independent audit.

## Baseline and implementation

The original suite passed **98 tests in 87.81 s** before implementation:
[baseline output](plane_strain_baseline.txt). The new dedicated suite passed
**57 tests in 98.35 s**: [numerical measurements](plane_strain_test_run.txt).
The final combined run and code checks are recorded below.

Runtime: Linux, Python 3.14.6, DOLFINx/Basix 0.11.0, UFL 2026.1.0,
NumPy 2.5.3, real-double PETSc 3.25.5, SLEPc/slepc4py 3.25.1, MPICH 5.0.1.
The installed environment was used; the repository binary lock was not recreated.
SLEPc is already in that lock. No SciPy or new pip dependency was introduced.

`config2d.py` adds only rectangular mesh, homogeneous material and zero-component
boundary configuration. `fem2d.py` owns the vector P1 space, consistent M, positive
lumped D, symmetric K, smooth volume load integration, component masks, assembled
spectral bounds and optional sparse eigenvalue diagnostics. Its `start()` method
returns the existing `CentralDifference` without modifying that class.

The weak stiffness is `integral sym(grad(v)):sigma(u) dOmega`, with ordinary 3D
lambda and mu under epsilon_yy=0. Geometry and component axes are `(x,z)`;
sigma_yy=lambda*(epsilon_xx+epsilon_zz) is nonzero in general. The triangle mass
is `rho*area/12 * [[2,1,1],[1,2,1],[1,1,2]] tensor I_2`, with positive row sums
`rho*area/3` for each node and component. K and M retain original entries at fixed
DOFs; stepping projects fixed components to zero and eigenproblems eliminate
the constrained scalar DOFs. The [developer notes](../development/plane_strain_core.md)
give full derivations, API usage, ownership, force units and the manufactured PDE.

Existing 1D production modules and tests are unchanged. Top-level Simulation,
CLI, PointMap and output remain 1D. No 2D point source, receiver interpolation,
absorber, layered material, radiation/geology example, output system, GUI,
performance refactor, adaptive mesh, local stepping, higher order or 3D was added.
The pre-existing user `.gitignore` edit and untracked `scripts/` remain excluded
from milestone commits.

## Files and verification coverage

| Files added/modified | Purpose and independent evidence |
|---|---|
| `src/seisfem/config2d.py` (new) | Dimension-specific immutable configuration, existing Isotropic material conversion reused unchanged |
| `src/seisfem/fem2d.py` (new) | Vector elasticity, mass, constraints, smooth loading, spectral diagnostics and bounded integration setup |
| `tests/plane_strain/test_operators.py` | 31 cases: full hand triangle M/K matrices, four diagonals, admissible negative lambda, rigid translations/rotation, affine strain/stress/energy, component constraints, rectangle geometry/aspect ratio, invalid configuration |
| `tests/plane_strain/test_spectral_dynamics.py` | 14 cases: sparse/dense spectra, sufficient bound, below/above/equal critical dt, fixed-component projection, energy, rigid motion, zero/two-free-DOF cases, componentwise free spectrum |
| `tests/plane_strain/test_manufactured_solution.py` | 10 cases: expanded forcing versus independent Navier identity, three spatial families, exact forced semidiscrete temporal convergence, fine-grid temporal Richardson checks |
| `tests/plane_strain/test_mpi.py`, `mpi_worker.py` | Two subprocess cases: serial versus 2/4 ranks, comparing coordinate-ordered masses, loads, stiffness action, displacement/velocity/acceleration, constraints, sparse eigenvalue and sufficient bound |
| `tests/plane_strain/helpers.py`, `manufactured.py`, `__init__.py` | Independent B-matrix oracle, manufactured field/expanded force, integrated error norms and package marker |
| `docs/development/plane_strain_plan.md`, `plane_strain_core.md` | Pre-coding plan and developer contract |
| This report and `docs/validation/plane_strain_*.txt` | Measured evidence, baseline and final commands |
| `README.md` | Link to the narrow 2D kernel, while preserving the 1D frontend contract |

Tests compare entire assembled matrices against an independent triangle B-matrix
loop using actual cell connectivity and direct prescribed Lamé values. They do
not merely compare two UFL mass identities. The unconstrained stiffness has
exactly three numerical null modes: two translations and one affine in-plane
rotation. The next eigenvalue is positive. Roller-style component constraints
remove rigid motion and produce a positive free principal subproblem. Affine
fields include compression in each direction, pure engineering shear, mixed
strain and rotation; their directly evaluated stress/strain and continuum
energy agree to approximately 1e-14 absolute tolerance on these scaled problems.

The full test matrix includes `left`, `right`, `left_right` and `right_left`
triangulations for algebraic verification; the convergence study uses the first
three. A translated nonsquare rectangle separately checks coordinate extents,
triangle count, mass/area and affine energy. These checks do not certify arbitrary
imported, curved or highly skewed mesh families.

## Measured spatial convergence

Material: rho=2.3, lambda=1.7, mu=1.2. Unit square with homogeneous fixed displacement
on every side. Both components of the smooth manufactured solution are nonzero;
normal, shear and Lamé coupling terms all participate. Final time is 0.31 s.
The final study uses **dt=0.000025 s** at every spatial resolution. Volume loads
use quadrature degree 8; integrated FE L2 displacement and H1 seminorm errors use
degree 12. These norms include the field between nodes.

| N per direction | Left/right L2 | Left/right H1 seminorm | Alternating L2 | Alternating H1 seminorm |
|---:|---:|---:|---:|---:|
| 8 | 4.08304e-2 | 9.55487e-1 | 3.07268e-2 | 9.32478e-1 |
| 16 | 1.09313e-2 | 4.80990e-1 | 8.07504e-3 | 5.20926e-1 |
| 32 | 2.79385e-3 | 2.40757e-1 | 1.87082e-3 | 2.45375e-1 |
| 64 | 7.02749e-4 | 1.20403e-1 | 4.60625e-4 | 1.19702e-1 |
| 128 | 1.75967e-4 | 6.02043e-2 | 1.18060e-4 | 6.23213e-2 |

Left and right errors agree to rounding in this reflection-symmetric problem;
the alternating diagonal provides additional evidence with different behavior.
All raw values for each family are retained in the run log.

| Refinement | Left/right L2 rate | Left/right H1 rate | Alternating L2 rate | Alternating H1 rate |
|---|---:|---:|---:|---:|
| 8 → 16, coarse diagnostic | 1.90118 | 0.99023 | 1.92796 | **0.83999** |
| 16 → 32 | 1.96813 | 0.99843 | 2.10980 | 1.08609 |
| 32 → 64 | 1.99117 | 0.99971 | 2.02201 | 1.03553 |
| 64 → 128 | 1.99770 | 0.99993 | 1.96407 | 0.94165 |

Acceptance uses all three rates from N=16 through 128: L2 in [1.8,2.2], H1 in
[.9,1.1]. N=8 is retained and explicitly not described as asymptotic for every
family. The alternating H1 rates oscillate around one; no claim of a monotone
approach to the limiting rate is made. P1 convergence is measured across three
finer refinements, not inferred from one favorable mesh.

Repeating N=128 at half dt changes the integrated FE L2 displacement by
**1.78e-11** (left), **1.79e-11** (right), and **1.21e-8** (alternating). The
largest change is **0.0103%** of the corresponding spatial L2 error, below the
0.1% threshold. Both sides of this final comparison use the same unweighted
integrated FE norm. The earlier raw log contains an initial lumped-mass-norm
screen; the appended final spatial rerun supersedes that contamination metric.

## Measured temporal convergence

At fixed N=12, the reference is the **exact forced semidiscrete matrix ODE**,
obtained by diagonalizing D^-1/2 K_ff D^-1/2 and solving each forced oscillator.
Its frequencies come from the matrix eigenproblem, not the central-difference
recurrence. This isolates time discretization without a continuum spatial error
floor. Final time is 0.31 s, with 40,80,160,320 steps.

| Family | Lumped-mass displacement errors | Consecutive rates |
|---|---|---|
| left | 7.26135e-6, 1.81402e-6, 4.53424e-7, 1.13351e-7 | 2.00105, 2.00026, 2.00007 |
| right | 7.26135e-6, 1.81402e-6, 4.53424e-7, 1.13351e-7 | 2.00105, 2.00026, 2.00007 |
| alternating | 3.36642e-4, 8.40415e-5, 2.10021e-5, 5.24998e-6 | 2.00204, 2.00057, 2.00015 |

A separate N=64 study uses 400,800,1600,3200 steps. Successive solution differences
cancel the fixed spatial error. Richardson rates are **2.00002,2.00021** (left),
**2.00002,2.00024** (right), and **2.03570,2.00922** (alternating). These complement
the exact temporal oracle; they do not claim absolute continuum accuracy from
a self-comparison alone.

## Measured spectral behavior

The sufficient bound is `2/sqrt(L)`, where L is the maximum absolute row sum of
the symmetric mass-scaled **free** stiffness. This bound is independent of the
signs of vector-elastic off-diagonal entries. Krylov-Schur estimates of lambda_max
agree with independent dense hand-matrix eigensolutions within 3e-11 relative;
reported relative eigenpair residuals must be below 1e-9. The Ritz estimate and
the sufficient upper bound are kept distinct.

Representative N=4 results, rho=2.3, lambda=1.7, mu=1.2:

| Mesh / constraints | lambda_max | Row-sum upper bound L | Sufficient dt | Estimated critical dt |
|---|---:|---:|---:|---:|
| left/right, natural | 239.22360 | 269.93264 | 0.1217313 | 0.1293088 |
| alternating, natural | 171.63390 | 217.54086 | 0.1356000 | 0.1526611 |
| left/right, fully fixed sides | 184.95321 | 228.17391 | 0.1324027 | 0.1470615 |
| alternating, fully fixed sides | 145.15076 | 179.96665 | 0.1490850 | 0.1660047 |

Negative lambda and selected component constraints are also tested. No artificial
Dirichlet eigenvalues appear because constrained DOFs are removed for spectral
analysis, while the original matrices remain unchanged for inspection/actions.

Exciting the highest free mode on left and right meshes and running 300 updates:

* dt=0.5 dt_critical: maximum mass-normalized displacement approximately 1.
* dt=0.999 dt_critical: maximum approximately 0.9999962.
* dt=1.001 dt_critical: maximum approximately **2.24565e11**.

At exact equality a nonzero highest-mode initial velocity produces the expected
linear growth (100*dt_critical in mass norm after 100 steps). Therefore equality
is documented as marginal; the public start helper always applies a factor
strictly below one to the sufficient bound. The over-limit tests deliberately
invoke the existing low-level integrator to demonstrate instability.

## Findings, surprises and limits

1. Alternating diagonals expose a coarse-grid H1 rate of 0.84 and larger transient
   temporal error than the single-diagonal families. The coarse point was retained,
   N=128 was added, and the timestep was reduced. An initial conservative screen
   compared a density-weighted lumped norm against the spatial L2 norm and rejected
   dt=5e-5. The final check consistently integrates the FE L2 norm on both sides;
   the smaller dt=2.5e-5 was retained. No convergence-order gate was widened.
2. Mapped P1 DOF coordinates at a nominal zero boundary can be about 5.4e-18 rather
   than exactly zero. The independent constraint test initially used exact equality;
   it was corrected to a 1e-14 geometric tolerance. Production facet localization
   already used a cell-scaled tolerance. A separate corrective test commit preserves
   that development history.
3. The stability endpoint is marginal, not a generally safe equality. The test with
   nonzero highest-mode velocity makes this explicit.
4. The unchanged integrator's scalar-array interface correctly handles vector DOFs
   once block ownership and component masks are handled in the operator. There was
   no need to redesign 1D APIs or introduce a dimension-switching Simulation.

Remaining risks: arbitrary meshes, high-order elements, near-incompressibility,
extreme material/coordinate conditioning, long-distance wave accuracy, general
nonhomogeneous/moving boundaries, all real seismic sources/receivers/absorbers,
and multidimensional radiation are unverified. The sparse eigenvalue calculation
is diagnostic and requires convergence; a residual is not by itself a guarantee
that an unobserved larger eigenvalue does not exist. The independent row-sum bound
is the default timestep guard. Geometry and fields remain small in MPI tests;
no many-rank/multi-node scaling or asymmetric-failure recovery is claimed. Both
consistent M and K are retained for verification, without a memory optimization
claim. The inherited package version string remains unchanged; identify this
milestone by its feature-branch commit, not by a new release tag.

## Reproduce and review

From the repository root in the same DOLFINx/PETSc/MPICH environment:

```bash
# Entire reference laboratory plus new 2D foundation (pytest itself is serial).
python -m pytest -ra -s

# New numerical measurements only.
python -m pytest tests/plane_strain -q -s

# Explicit original-1D regression selection.
python -m pytest tests/unit tests/analytical tests/integration tests/audit -q

# Distributed checks launch their own 1/2/4-rank subprocesses.
python -m pytest tests/plane_strain/test_mpi.py -q

# A standalone 2D four-rank smoke artifact (no production output frontend).
mpiexec -n 4 python -m tests.plane_strain.mpi_worker /tmp/seisfem-2d-smoke.npz

ruff check src tests examples
ruff format --check src tests examples
pre-commit run --all-files
git diff --check
```

The operator/rigid/mass/spectral tests execute before MMS in implementation history;
pytest's file ordering is not a dependency. The full suite also executes the
original 1D serial/2/3/4-rank audit checks. No tests depend on a pre-existing result
file or use regression snapshots as the analytic oracle.

Final combined regression: **155 passed in 189.05 s**, including all 98 original
1D tests and 57 new 2D cases. After correcting the contamination comparison to
use like FE L2 norms, its three spatial cases passed again in **92.39 s** with
unchanged convergence errors/rates. Ruff check, Ruff format-check, pre-commit
`--all-files`, and `git diff --check` all pass. The
[full regression record](plane_strain_full_regression.txt) includes those commands
and the final spatial rerun. No original 1D production file or test changed.

The feature commits separate the pre-coding plan, operator kernel/algebraic
verification, mapped-coordinate test correction, spectral dynamics, manufactured
and MPI verification, the final norm-consistency correction, and documentation.
No merge to main or tag change has been performed. Only the user's pre-existing
.gitignore edit and untracked scripts are left outside the milestone commits.

## Independent-review corrective pass

The review at `7cb5e4e` found no critical or demonstrated major defect for valid
numerical inputs. Two functional defects were reproduced before correction:
explicit zero-vector load assembly raised `ValueError: This integral is missing
an integration domain`; the two-rank nonnumeric-initial-data regression timed out
after 45 seconds. Adding only an explicit integration domain still failed because
UFL had removed the test argument (a rank-zero form).

`start()` now catches ordinary local array conversion, shape and finite-value
errors before collectively gathering diagnostics. Every rank raises the same
`ValueError`, identifying the originating rank, argument, exception type and
message. No rank proceeds to spectral setup when any initial array is invalid.
This covers deterministic initial-argument failures, not general MPI recovery.
The load assembler returns a correctly sized owned zero vector when UFL simplifies
the integrand to exact zero, and attaches an explicit domain to other load forms.

Permanent tests cover asymmetric conversion/shape/nonfinite errors in each of
`u0`, `v0`, and `force0` on 2/4 ranks; ranks owning no free DOFs; a fully constrained
mesh with a rank owning no DOFs at four ranks; serial/distributed explicit zero
loading; and independently collapsed component maps on a translated nonsquare
rectangle. Existing serial-versus-MPI field comparisons remain in place.
The focused operator, spectral and MPI selection passed **51 tests in 4.56 s**.
