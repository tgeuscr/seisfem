# First narrow 2D plane-strain milestone

Date: 2026-09-11. Branch: `feature/2d-plane-strain-core`.
Base: `main` / `v0.1.1`, commit
`43f3dd70c1875a7dafe37910fc98e7c490e9b5dd`. The baseline branch/tag are preserved.
This is a **verified 2D vector-elastic numerical foundation**, not a complete
2D seismic Simulation/CLI workflow. The independent review at `7cb5e4e` is complete; this corrective pass awaits final audit.

## Baseline and implementation

The original suite passed **98 tests in 87.81 s** before implementation:
[baseline output](plane_strain_baseline.txt). The pre-review dedicated suite passed
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

## Original implementation files and verification coverage

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

## Original t=.31 spatial measurements (retained for comparison)

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

The original acceptance gated all three N=16→128 pairwise H1 rates in [.9,1.1].
The independent review showed this pointwise gate was observation-time sensitive,
including at t=.2 on the same problem. That H1 methodology is superseded below;
the original L2 gate at t=.31 is retained. These historical numbers alone do not
establish arbitrary-time pairwise H1 rates near one.

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
   the smaller dt=2.5e-5 was retained. The corrective pass below replaces the
   original single-time H1 gate with multi-time evidence; the L2 gate is unchanged.
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
no many-rank/multi-node scaling or general asymmetric-failure recovery is claimed.
Ordinary initial-array argument rejection is now MPI-coherent, as detailed below. Both
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

Original pre-review combined regression: **155 passed in 189.05 s**, including all 98 original
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

### Revised H1 evidence

The spatial test now observes t=.13,.20,.31,.37 s and integrates squared FE errors
over [0,.4] s on the same three diagonal families, N=8,16,32,64,128, dt=2.5e-5 s.
The diagnostic is the time-RMS seminorm
`sqrt(integral_0^.4 |u_h-u|_H1² dt / .4)`. Spatial quadrature remains degree 12;
a time Constant permits reusing the error forms without compiling each sample.
Time integration uses the trapezoidal rule with spacing .00125 s (321 samples),
and comparison with spacing .0025 must change either RMS norm by less than .1%.

Fit `log(E)=p log(1/N)+C` over N=16,32,64,128. The H1 order gate now applies to
the integrated fit (.9<p<1.1), accompanied by a pointwise O(h) envelope: at each
observation time the largest E_h/h divided by its smallest over those four
refinements must be <1.5. That is a finite-refinement regression guard against
loss of the envelope, not a theorem about arbitrary times or all mesh sizes.
All raw pointwise errors, pairwise rates, fitted orders, and all 321 H1 samples
per mesh are retained in the corrective regression output. N=8 remains a reported
coarse diagnostic. The original three L2 gates [1.8,2.2] at t=.31 are unchanged.

The finest solution is repeated at dt/2. Compare **FE field differences**, using
both L2 and H1, at the four observation times and over the time interval; every
relative change must be <.1% of its corresponding spatial error. This directly
checks H1 temporal contamination, which the old L2-only comparison did not do.
Neither timestep nor H1 acceptance is chosen to make a single observation time
look favorable. Oscillating pairwise rates are retained as evidence.

### Initial acceleration qualification

The counterexample is reproduced with FE L2 integration of
`D^-1(F0-K I_h U0)` against the continuum `-omega² U0`, without a time-difference
approximation. Thus the initial defect exists before time stepping or MPI.
For N=8,16,32,64, alternating-mesh errors are
**4.41689012, 5.06746640, 5.32381022, 5.44183736** despite nodal displacement
interpolation errors **.04509489, .01149321, .00288718, .00072266**.
Single-diagonal initial acceleration errors are **.81209656, .25501969,
.06990469, .01817533** on both left and right families.

An independently hand-assembled interior quadratic patch makes the mechanism
explicit: for U=(x²,0), the continuum `-div sigma(U)/rho` is
`(-2*(lambda+2*mu)/rho,0)`. Alternating nodal stars give .75 and 1.5 times its x
component from `D^-1 K I_h U`, at both N=4 and N=8; the factors do not approach
one on refinement. Single-diagonal patches give the continuum value. Weak FEM
consistency does not imply this strong nodal-operator consistency. The mismatch
from nodal initialization can excite mesh-scale semidiscrete transients; it is
consistent with oscillatory H1 errors and displacement convergence.

A test-only Ritz projection uses the independently expanded elastic load:
`K_ff R_h U0 = F0 + omega² integral rho phi_f U0`. Its initial acceleration is
therefore `-omega² D_f^-1 integral rho phi_f U0`, with fixed components zero.
The test verifies this identity and the existing start/evaluate initialization,
and separately measures the continuum FE L2 error. The alternating errors become
**.56442384, .14830262, .03754367, .00941546**, with rates **1.928235,
1.981903, 1.995466**. Left/right errors become **.57533737, .15300037,
.03900179, .00981552**, with rates **1.910873, 1.971923, 1.990404**.
Projected displacement also converges at order two. This supplies meaningful
compatible-initial-data evidence; no new production projection API was added.

This verifies a specific continuum **initial** acceleration with compatible
data. It does not establish acceleration convergence throughout a simulation.
The existing serial/MPI acceleration comparison establishes partition consistency
only. No acceleration receiver or method redesign is introduced.

### Scope and remaining limitations after correction

Only two production paths changed: ordinary initial-array validation and exact
zero load assembly. Constitutive law, mass/stiffness formulas, component masks,
spectral algorithm, 1D modules and CentralDifference remain unchanged. Added tests
complement the existing hand-matrix, rigid-mode, affine-energy, component-mask,
serial/MPI field and spectrum comparisons instead of duplicating them.

Each `stable_dt` access rebuilds, scales and destroys a free submatrix; `start()`
accesses it again. Consistent mass is retained for the operator lifetime. These
are setup and memory costs to account for in larger workflows, not optimized
paths. Collective object lifetime and borrowed `apply()` views remain caller
responsibilities. No caching was introduced. General rank-asymmetric runtime
exceptions, callback failures and MPI fault recovery remain outside the contract.

The evidence remains limited to the documented homogeneous rectangular vector-P1
foundation and finite refinement/time studies. Dynamic convergence still uses
the unit-square MMS with a shared component time factor; a nonsquare MMS with
distinct component time factors has not been systematically refined here.
Continuum acceleration at later times, general mesh families and the other physical regimes listed above remain
unverified. No source, receiver, absorber, geology, GUI or performance feature
was added. The branch is for final adversarial review, not merged or retagged.

### Corrective H1 measurements

Errors below are FE H1 seminorms. Each fit uses N=16,32,64,128; every pairwise
rate is reported, including the coarse N=8→16 rate. The [complete corrective
run](plane_strain_corrective_regression.txt) retains full-precision JSON records
with all pointwise L2/H1 errors and all 321 H1 samples per mesh, not only these
rounded tables. The [pre-fix reproduction](plane_strain_review_reproduction.txt)
records the two failing functional regressions.


**left**

| t | E8 | E16 | E32 | E64 | E128 | Pairwise rates (8→16 through 64→128) | Fitted p |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.13 | 1.211134571 | 0.6131955983 | 0.3075309641 | 0.1538804883 | 0.07695455959 | 0.981940, 0.995616, 0.998921, 0.999732 | 0.998173 |
| 0.20 | 1.143127964 | 0.5776753707 | 0.2895208205 | 0.1448418685 | 0.07243095558 | 0.984656, 0.996592, 0.999188, 0.999800 | 0.998593 |
| 0.31 | 0.9554872662 | 0.48099043 | 0.2407573041 | 0.1204031953 | 0.06020433614 | 0.990228, 0.998429, 0.999706, 0.999934 | 0.999391 |
| 0.37 | 0.8148032453 | 0.4095337439 | 0.2048850473 | 0.1024478838 | 0.05122423429 | 0.992469, 0.999168, 0.999925, 0.999992 | 0.999718 |

**right**

| t | E8 | E16 | E32 | E64 | E128 | Pairwise rates (8→16 through 64→128) | Fitted p |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.13 | 1.211134571 | 0.6131955983 | 0.3075309641 | 0.1538804883 | 0.07695455959 | 0.981940, 0.995616, 0.998921, 0.999732 | 0.998173 |
| 0.20 | 1.143127964 | 0.5776753707 | 0.2895208205 | 0.1448418685 | 0.07243095558 | 0.984656, 0.996592, 0.999188, 0.999800 | 0.998593 |
| 0.31 | 0.9554872662 | 0.48099043 | 0.2407573041 | 0.1204031953 | 0.06020433614 | 0.990228, 0.998429, 0.999706, 0.999934 | 0.999391 |
| 0.37 | 0.8148032453 | 0.4095337439 | 0.2048850473 | 0.1024478838 | 0.05122423429 | 0.992469, 0.999168, 0.999925, 0.999992 | 0.999718 |

**left_right**

| t | E8 | E16 | E32 | E64 | E128 | Pairwise rates (8→16 through 64→128) | Fitted p |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.13 | 1.257700549 | 0.6164998883 | 0.3043786656 | 0.1535626883 | 0.08443974558 | 1.028616, 1.018233, 0.987040, 0.862834 | 0.959136 |
| 0.20 | 1.170121222 | 0.6173061438 | 0.2864524928 | 0.1588841748 | 0.07209788467 | 0.922600, 1.107690, 0.850320, 1.139947 | 1.014419 |
| 0.31 | 0.9324775733 | 0.5209259849 | 0.2453745612 | 0.1197023895 | 0.06232130842 | 0.839991, 1.086093, 1.035534, 0.941655 | 1.022538 |
| 0.37 | 0.8220334651 | 0.4291460876 | 0.2134197615 | 0.1038753486 | 0.05118049491 | 0.937728, 1.007775, 1.038840, 1.021187 | 1.024225 |

**Time-RMS H1 over [0,.4]**

| Family | Errors N=8,16,32,64,128 | Pairwise rates | Fitted p |
|---|---|---|---:|
| left | 1.09932031, 0.5555604616, 0.278454329, 0.1393076761, 0.06966374936 | 0.984596, 0.996503, 0.999166, 0.999795 | 0.998556 |
| right | 1.09932031, 0.5555604616, 0.278454329, 0.1393076761, 0.06966374936 | 0.984596, 0.996503, 0.999166, 0.999795 | 0.998556 |
| left_right | 1.109747629, 0.5691133779, 0.2867641688, 0.143936305, 0.07206789069 | 0.963444, 0.988851, 0.994434, 0.998002 | 0.993830 |

The integrated fitted orders support first-order H1 convergence across these
refinements. At t=.20 on alternating meshes, .850320 and 1.139947 remain outside
the former pointwise gate despite a fitted pointwise order of 1.014419. Pointwise
fits over all four observation times range from .959136 to 1.024225.

**Temporal-contamination and envelope checks**

Percentages are half-timestep H1 field differences divided by the corresponding
H1 error, not differences between error norms.

| Family | t=.13 (%) | t=.20 (%) | t=.31 (%) | t=.37 (%) | Time-RMS (%) | Max time-quadrature change (%) | Max E_h/h envelope ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| left | 4.7751037e-07 | 6.4496455e-07 | 1.0932302e-06 | 1.4622632e-06 | 7.4295694e-07 | 0.00020688518 | 1.003981 |
| right | 4.808782e-07 | 6.455262e-07 | 1.0924857e-06 | 1.4648924e-06 | 7.4376985e-07 | 0.00020688518 | 1.003981 |
| left_right | 0.0010969693 | 0.0058247502 | 0.014876463 | 0.023441412 | 0.0076015258 | 0.00089045624 | 1.109667 |

The largest H1 temporal change is .0234414% pointwise and .00760153% in time-RMS,
below .1%. Doubling time-quadrature spacing changes the H1 RMS by at most
.000890456%. The H1 oscillation is therefore not removed by reducing dt, nor is
the integrated order an artifact of a single observation time. These finite
checks support the stated envelope; they do not certify every pointwise slope.

### Corrective files, tests and reproduction

Changes relative to the reviewed `7cb5e4e` (excluding the user's pre-existing
`.gitignore` and `scripts/` work):

| File | Change |
|---|---|
| `src/seisfem/fem2d.py` | Modified: collective initial-array validation and explicit zero load |
| `tests/plane_strain/test_operators.py` | Modified: zero-load and independent collapsed-component/coordinate checks |
| `tests/plane_strain/test_mpi.py` | Modified: 2/4-rank edge-case subprocess tests with process-group timeout cleanup |
| `tests/plane_strain/mpi_edge_worker.py` | Added: asymmetric validation, no-free-DOF and empty-rank probes |
| `tests/plane_strain/manufactured.py` | Modified: reusable integrated FE error forms |
| `tests/plane_strain/test_manufactured_solution.py` | Modified: multiple observation times, time-RMS H1, fitted orders, envelope, L2/H1 temporal contamination and raw time series |
| `tests/plane_strain/test_initial_acceleration.py` | Added: three mesh-family projection/acceleration studies and independent quadratic-patch mechanism |
| `docs/development/plane_strain_core.md` | Modified: numerical evidence, acceleration qualification, MPI validation and setup/lifetime costs |
| `docs/validation/plane_strain_core.md` | Modified: corrected claims and complete review response |
| `docs/validation/plane_strain_corrective_regression.txt` | Added: final commands, results and full-precision numerical records |
| `docs/validation/plane_strain_review_reproduction.txt` | Added: pre-fix failure evidence |

Eight collected cases were added: two operator tests, two MPI subprocess tests,
three compatible-initial-data studies and one quadratic-patch test. The three
existing spatial cases were strengthened. Within each new MPI case, nine
asymmetric argument cases cover all three initial arrays and three failure types.
The existing serial-to-2/4-rank field/spectrum comparisons still exercise valid
inputs and ghost accumulation; their acceleration evidence is partition consistency.

Exact commands, from the repository root in the existing `fenicsx0.11` environment:

```bash
conda activate fenicsx0.11
python -m pytest -ra -s
ruff check src tests examples
ruff format --check src tests examples
pre-commit run --all-files
git diff --check

# Repeated MPI comparisons and edge cases at both 2 and 4 ranks.
python -m pytest -q -s tests/plane_strain/test_mpi.py

# Focused numerical diagnostics, if desired.
python -m pytest -q -s tests/plane_strain/test_manufactured_solution.py -k spatial
python -m pytest -q -s tests/plane_strain/test_initial_acceleration.py
mpiexec -n 2 python -m tests.plane_strain.mpi_edge_worker
mpiexec -n 4 python -m tests.plane_strain.mpi_edge_worker
```

The functional corrections are commit `ab7d14f`; numerical tests and developer
qualifications are commit `c48ced8`. The final documentation commit records this
report and its raw evidence. No production refactoring or performance optimization
was performed. `main` and `v0.1.1^{commit}` remain
`43f3dd70c1875a7dafe37910fc98e7c490e9b5dd`.

Final corrective regression: **163 passed in 435.22 s (7:15)**, comprising all
**98 original 1D tests and 65 plane-strain cases**. Ruff check passed; Ruff format
reported 35 files already formatted; pre-commit `--all-files` passed; and
`git diff --check` passed. The repeated dedicated MPI suite passed **4 cases in
2.97 s**, covering serial/2/4 equivalence and both new 2/4-rank edge workers.
These runs include the unchanged original 1D asymmetric/distributed cases.

The controlled spectral instability test remains unchanged and passed in the
full run: at .999 of the critical timestep the highest-mode maximum norm was
.9999962; at 1.001 it reached 2.245654e11 after 300 updates. The new work changes
neither the sufficient spectral bound nor valid-input dynamics. Full command
output and numerical precision are preserved in the linked corrective run.
