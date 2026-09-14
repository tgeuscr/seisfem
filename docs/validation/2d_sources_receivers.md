# Homogeneous 2D seismic experiments: design and validation

This milestone adds a public source → plane-strain propagation → receiver pathway
on `feature/2d-sources-receivers`, starting at `5fa7ee9` (`v0.2.0`). The 1D solver,
2D operators, material conversion, and `CentralDifference` implementation are
unchanged. No merge or tag change is part of this work.

## Public workflow and architecture

```python
from seisfem import Simulation2D, SimulationConfig2D

cfg = SimulationConfig2D.model_validate(
    {
        "domain": {"lower": [-2400, -2400], "upper": [2400, 2400], "cells": [240, 240]},
        "material": {"density": 2400, "vp": 3200, "vs": 1800},
        "time": {"dt": 0.0005, "duration": 0.95},
        "source": {
            "type": "force",
            "position": [3.7, 2.9],
            "direction": [1, 0],
            "wavelet": {"type": "ricker", "f0": 8, "amplitude": 1e8, "time_shift": 0.1875},
        },
        "receivers": [
            {"name": "axial", "position": [903.7, 2.9]},
            {"name": "transverse", "position": [3.7, 902.9]},
        ],
    }
)
result = Simulation2D(cfg).run()
# Complete traces on every rank, shape (1901, 2, 2):
ux, uz = result.displacement[:, :, 0], result.displacement[:, :, 1]
vx, vz = result.velocity[:, :, 0], result.velocity[:, :, 1]
# Optional dependency, never used during propagation:
ds = result.to_xarray()
```

The 240-cell example intentionally remains small: its S envelope peak has about
24 ms spatial dispersion. The refinement validation below uses 320, 480 and 640
cells per direction. Run `examples/2d/homogeneous.py --cells 640` for the finest
case, or omit `--cells` for the lighter 240-cell case.

`config2d.py` adds `RickerWavelet`, `ForceSource2D`, `Receiver2D`, and
`SimulationConfig2D`. The experiment schema extends the existing minimal
`PlaneStrainConfig`, reusing `Isotropic`, `Rectangle`, `ZeroDisplacement`, and
`TimeConfig`. Unknown keys, malformed tuples, nonfinite numbers, duplicate names,
zero directions, out-of-domain points, and nonintegral step counts are rejected.
No material-admissibility condition is changed. Boundaries are natural traction
free unless existing homogeneous component constraints are supplied, for example
`constraints=[{"side": "left", "components": ["x"]}]`.

`points2d.PointMap2D` supplies a shared source/receiver functional.
`sources2d.PointForce2D` combines a cached spatial vector and callable wavelet;
additional wavelet implementations can change the configuration union without
changing the time loop. Independent source instances can be summed; the public
schema exposes one optional source now. `simulation2d.Simulation2D` owns build,
run and close, including cleanup after a rejected timestep. A standalone `run()`
closes automatically; a context manager supports repeated runs from rest on one
assembled operator. `results2d.SimulationResult2D` holds plain NumPy arrays.
Binary imports remain lazy at package entry. The existing `Simulation` and
`SimulationConfig` APIs, CLI, and 1D result contract remain unchanged.

## Physical meaning and units

Coordinates are `(x,z)` in metres, with positive-up z. The PDE is

\[
\rho u_{tt}-\nabla\cdot\sigma(u)=F(t)\widehat d\,\delta_2(x-x_s),\qquad
u=(u_x,u_z),
\]

\[
\epsilon(u)=\tfrac12(\nabla u+\nabla u^T),\qquad
\sigma=\lambda\operatorname{tr}(\epsilon)I+2\mu\epsilon,
\quad\mu=\rho V_s^2,\quad\lambda=\rho(V_p^2-2V_s^2).
\]

This is **plane strain**, with the ordinary 3D Lamé moduli, not plane stress.
The source is a **line force per unit out-of-plane length**: `amplitude` and
`F(t)` are **N/m**, `delta_2` is m⁻², and the body force density is N/m³.
Displacement and velocity are m and m/s. A horizontal or vertical vector force
generally excites both P and S waves; it is not a pure-polarization source.

The finite nonzero input direction is normalized to `d_hat=d/||d||`; its magnitude
has no physical scaling role. Scaling before the norm prevents overflow and
underflow for very large or small finite vectors. Signed wavelet amplitude is
allowed. The Ricker convention is exactly

\[
F(t)=A\,[1-2\pi^2 f_0^2(t-t_0)^2]\exp[-\pi^2 f_0^2(t-t_0)^2],
\]

with `f0 > 0` in Hz, finite `amplitude=A`, and finite `time_shift=t0` in seconds.
`F(t0)==A`; there is no additional normalization or causal truncation. Evaluation
rejects nonfinite time. Extreme tails are evaluated as zero below floating-point
range to avoid overflow multiplied by zero. Runs begin at t=0 with zero initial
conditions; an appreciable initial tail is included in startup, not silently
removed. The validation uses `f0*t0=1.5`.

## FE source and receiver functional

The source is the weak functional

\[
\ell(v_h)=F(t)\widehat d\cdot v_h(x_s),\qquad
(b_s)_{2a+c}=N_a(x_s)\widehat d_c.
\]

It has no nearest-node approximation, no cell-area or mesh-size scaling, and no
mass division until the existing integrator applies the lumped inverse mass.
The static vector satisfies `b_s.T @ v_dofs = d_hat · v_h(xs)`. In particular,
all affine vector fields are reproduced to floating-point precision. For a
point on a fixed side, the same functional is assembled, and the integrator's
existing fixed-component projection applies it to the admissible free subspace.
A fully constrained component at that point produces no motion.

Setup uses DOLFINx 0.11
[`bb_tree`, `compute_collisions_points`, and `compute_colliding_cells`](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.geometry.html),
restricted to **owned cells**. For each candidate triangle, sort its three stored
mesh-geometry vertex coordinates lexicographically; choose the smallest resulting
cell key globally, then rank as a final tie breaker. Stored mesh geometry matters:
tabulated FE DOF coordinates can acquire tiny permutation-dependent mapping
roundoff, which initially caused a strict MPI cell-key comparison to fail despite
matching traces. The implementation uses geometry coordinates for the key and the
regression explicitly compares the selected keys across partitions.

A geometric triangle identity is independent of local/global DOF renumbering and
rank count for this generated rectangle mesh. The owning rank can change with
partition; the selected geometric cell and resulting functional remain the same.
If no owned cell contains a point, all ranks reject it. The public schema checks
the closed rectangle first, accepting physical boundaries and rejecting outside
coordinates. No tolerance is used to admit outside points in the public schema.

Each elected owner caches the containing cell, nodal reference coordinate
`(xi,eta)`, weights `(1-xi-eta,xi,eta)`, local parent scalar DOFs, and global scalar
DOFs. The coordinates follow the affine map from the three P1 nodal coordinates;
no higher-order or curved-cell generalization is claimed. For sources, only the
chosen cell inserts weights; one reverse-add ghost exchange sends contributions
to DOF owners. It occurs once during preprocessing, not at every timestep.

A receiver uses precisely the same weights on each displacement/velocity
component. This adjoint relationship between interpolation and source injection
is essential to reciprocity. User receiver ordering is preserved, including
coincident coordinates with distinct names. Node, interior, facet, near-facet,
vertex, physical boundary, empty receiver set, and translated nonsquare geometry
are covered. Unit-square affine receiver error is at most `4.44e-16` in the MPI
probe; the corresponding source functional error is at most `2.22e-16`.
The larger translated-domain test allows scale-appropriate absolute error `2e-12`.
Normalization sums equal `(0.6,-0.8)` for direction `(3,-4)` to roundoff, including
points on and within `1e-12 m` of a central partition vertex.

## Force and velocity timing

The unchanged central-difference recurrence uses **F(t_n)** to compute `u[n+1]`.
For zero initial data, startup includes

\[
a^0=D^{-1}F(0)b_s,\qquad
u^{-1}=\tfrac12\Delta t^2a^0,\qquad
u^1=\tfrac12\Delta t^2a^0
\]

on free DOFs. Tests independently reconstruct the first two steps with nonzero
initial forcing. The same source callable is evaluated at zero for initialization
and at `t_n` for the recurrence. No forcing is shifted to a half step.

Both returned fields refer to the integer time `result.time[n]`:

\[
v^n=(u^{n+1}-u^{n-1})/(2\Delta t).
\]

Velocity is the integrator's centered reconstruction, not an undisclosed
half-step variable. The final sample evaluates one auxiliary next state using
force at the final requested time; that next state is not recorded or advanced.
At startup, velocity equals the prescribed zero initial velocity up to roundoff.
Tests check interior samples against centered differences of receiver displacement
and the final sample against an independently constructed lookahead. No validated
continuum acceleration receiver is exposed.

Stability is checked by `PlaneStrainOperators.start()` using the existing
sufficient assembled absolute-row-sum spectral bound for the mass-scaled free
elastic operator and the existing safety policy. The external force does not
change this homogeneous stability bound. Unsafe requested dt is rejected; no new
heuristic CFL replacement is introduced. Geometry and material checks also remain
those of the validated core.

## Result and MPI semantics

* `time`: `(N_time,)`, including t=0 and the requested final time.
* `receiver_coordinates`: `(N_receiver,2)` in input order.
* `receiver_names`: input names; `components == ("x","z")`.
* `displacement`, `velocity`: `(N_time,N_receiver,2)`.
* `metadata`: normalized configuration, MPI size, elected owners, units and timing.
* `to_xarray()`: variables with dimensions `(time,receiver,component)`, receiver
  names and IDs, auxiliary `receiver_x`/`receiver_z`, and SI unit attributes.

All ranks call the same public API and receive **complete results on every rank**.
This is an explicit 2D convenience choice; existing 1D results remain rank-local.
Only owners store receiver histories during propagation. A single final allgather
assembles those small histories by receiver ID. No full wavefield is globally
gathered. Two field ghost refreshes per sample support local displacement and
velocity interpolation; no point search or source-vector assembly occurs in the
loop. Support work is O(time × receivers) after setup, in addition to the existing
sparse propagation and ghost communication costs.

Setup replicates O(ranks × points) candidate keys. Complete result histories cost
O(time × receivers) memory on every rank at completion, with temporary gathered
pieces. This is suitable for this milestone, not a claim of large acquisition-set
or streaming-I/O scalability.

`Simulation2D` also accepts a raw mapping: local validation errors are gathered
before later collectives. Passing different valid configurations is rejected
collectively. Point-coordinate and source-direction setup use the same guard.
MPI tests intentionally put malformed/NaN/Inf/zero-direction input on only rank 0,
and cover out-of-domain points, mismatched valid inputs, and empty cell ownership
on a two-triangle mesh under four ranks. All ranks raise; timeout protection kills
the worker group if that contract regresses. All participating ranks must enter
the API: a validation exception in caller code before the call cannot be caught
inside the simulation.

## Measured homogeneous wave experiment

Material: rho=2400 kg/m³, Vp=3200 m/s, Vs=1800 m/s. Rectangle: `[-2400,2400]² m`,
right-diagonal P1 triangles, traction-free sides. Horizontal force at `(3.7,2.9)` m,
A=1e8 N/m, f0=8 Hz, t0=0.1875 s. Stations 900 m away are axial, transverse, and
45-degree oblique; their exact coordinates are in the executable example.
Time step is fixed at 0.0005 s, duration 0.95 s throughout refinement.

Before running, predict `tP=t0+900/3200=0.46875 s` and
`tS=t0+900/1800=0.68750 s`. Measure the P displacement at the axial station and
S displacement at the transverse station, both x components. The diagnostic
zero-pads the full trace to four times its length, constructs an FFT analytic
signal, finds its envelope maximum in ±0.065 s around the prediction, then uses
three-sample parabolic interpolation. A maximum at the window boundary is an
error. The window and method are fixed before mesh comparison; there is no
visual arrival picking or per-mesh tuning.

| Cells per side | Grid spacing (m) | Measured P (s) | P error (ms) | Measured S (s) | S error (ms) |
|---:|---:|---:|---:|---:|---:|
| 320 | 15 | 0.469199697497 | +0.450 | 0.701814671340 | +14.315 |
| 480 | 10 | 0.467550255848 | −1.200 | 0.693405007617 | +5.905 |
| 640 | 7.5 | 0.466982133673 | −1.768 | 0.690272456256 | +2.772 |

The finest errors are −0.377% and +0.403% of the predicted peak times. Apparent
speeds from these peaks are approximately 3220 and 1790 m/s. This demonstrates
two distinct propagation speeds. It does not make a finite-bandwidth envelope
peak an exact continuum onset: 2D tails, near-field contributions, numerical
anisotropy, and dispersion affect the peak. In particular, P peak error need not
approach zero monotonically. The test allows 5 ms (4% of a wavelet period).

The shortest geometric source→side→receiver path at the fastest speed is
**1.2164375 s even for a force applied at t=0**, later than the entire 0.95 s record.
The finite-domain experiment is interpreted before those unwanted returns. No
absorber is used or implied. Very small discrete precursor effects are not a
claim of strict finite propagation speed for the semidiscrete ODE.

For the finest mesh, the P-window x-component peaks are `2.02590e-4 m` axially
and `2.01080e-5 m` transversely (ratio 0.0993). S-window peaks are `2.35013e-5 m`
axially and `4.83323e-4 m` transversely (ratio 0.0486). The oblique station has
both branches. Thus P/S amplitude ratios depend strongly on azimuth. The nominally
forbidden z component on the axes has peak fractions 0.00870 and 0.00365; these
fractions decrease with refinement. A single-diagonal mesh does not have exact
reflection symmetry, and near-field terms prevent identifying the other branch
with an exact zero in all time windows. This is a radiation sanity check, not an
analytical Green-function amplitude benchmark.

## Reciprocity and refinement

Four independent public runs exchange off-grid A=`(0.313,0.487)` and
B=`(0.679,0.723)` m, using identical time functions, material and boundary
conditions. Duration is 0.5 s, so this small-domain test also exercises boundary
interaction. Relative error is `||trace_A-trace_B||₂/||trace_B||₂`.

| Boundary case | Pair | Displacement relative error | Velocity relative error |
|---|---|---:|---:|
| Free | xx ↔ xx | 5.55e-15 | 2.16e-14 |
| Free | xz ↔ zx | 1.56e-14 | 1.48e-14 |
| Left x fixed, lower z fixed | xx ↔ xx | 8.71e-15 | 2.64e-14 |
| Left x fixed, lower z fixed | xz ↔ zx | 2.28e-14 | 2.59e-14 |

The regression tolerance `2e-12` allows accumulated platform-dependent roundoff
above these measured errors. It is not a tolerance inferred from visually similar
traces. Both same- and cross-component displacement and velocity are tested.

Refinement holds source coordinate, strength, f0, shift, dt, receiver coordinates,
material, and geometry fixed. No comparison is made at the singular source.
With the empirical bandwidth of interest `fmax=3*f0=24 Hz`, the shortest wavelength
is `Vs/fmax=75 m` (the Ricker spectral amplitude there is about 0.003 of peak, not
a hard spectral cutoff). Grid spacing/wavelength is 0.2, 0.1333, 0.1; maximum
triangle diameter/wavelength is sqrt(2) times those values.

| Mesh pair | Combined axial-P/transverse-S trace relative difference | P trace | S trace |
|---|---:|---:|---:|
| 320 → 480 | 0.157686 | 0.033227 | 0.171426 |
| 480 → 640 | 0.056347 | 0.011789 | 0.061160 |

These are unfiltered displacement traces over the complete pre-return record,
normalized by the finer trace, with no time alignment or amplitude rescaling.
Peak axial/transverse amplitudes are `(2.04594e-4,4.68048e-4)`,
`(2.03012e-4,4.80765e-4)`, and `(2.02590e-4,4.83323e-4)` m.
The last change is −0.208% for P and +0.532% for S, despite decreasing spacing by
25%. Together with the exact discrete source functional, this rules out arbitrary
powers-of-h source normalization. Traces stabilize, but the remaining 6.1% S
trace difference is material for amplitude-sensitive studies. No formal norm
convergence of the singular continuum field is claimed.

## Serial, two-rank and four-rank evidence

The MPI wave case uses the same physical experiment with 240 cells per side and
dt=0.0005 s. P and S peaks are approximately **0.47157548 s** and **0.71193570 s**
on all three rank counts. This coarser case has known spatial dispersion;
partition agreement is not evidence of continuum accuracy. The worker also
checks static source normalization, affine interpolation, selected geometric
cell keys, ordered traces at/near the central partition vertex, complete results
on every rank, reciprocity, and coherent invalid-input handling.

| Relative to serial | Max displacement difference (m) | Max velocity difference (m/s) | Max difference / serial peak, u / v |
|---|---:|---:|---|
| 2 ranks | 6.58e-18 | 9.51e-16 | 1.49e-14 / 4.84e-14 |
| 4 ranks | 7.05e-18 | 8.92e-16 | 1.60e-14 / 4.54e-14 |

Source and receiver affine errors remain below `4.45e-16`. Awkward central-point
trace differences are at most `1.74e-18 m` and `4.45e-16 m/s`. Maximum differences
in the stacked reciprocity u/v traces are `1.99e-15` and `1.33e-15` for two and
four ranks, respectively. The automated comparison allows `3e-11` times the
serial peak, including near-zero samples, and checks arrivals separately. The maximum partition-dependent arrival shift
is `3.33e-16 s`; static source-vector entries differ by at most `2.22e-16`.
Distributed reciprocity relative errors remain below `3.72e-14` for both fields.

## Snapshots and remaining limits

Displacement snapshots are **deferred**. There is no snapshot field, timestep
mapping, topology gathering, or opaque PETSc snapshot result in this API. A small
follow-up should add selected times, actual mapped times and one mesh topology
representation with explicit distributed ownership. Receiver correctness was
completed first; snapshot MPI/topology and output policy remain separate work.

This milestone does not establish heterogeneous materials, geological interfaces,
absorbing boundaries/PML, moment tensors, explosive or pure P/S production
sources, anisotropy, 3D propagation, or validated continuum acceleration
receivers. The point-source continuum field is singular; all physical refinement
claims here concern receivers away from it. Resolution and boundary-return checks
must be repeated for new spectra, paths and model sizes. Long-term trace streaming,
large acquisition sets and multi-node performance are not validated here.

The next scientific milestone should validate absorbing boundaries for this same
homogeneous plane-strain source/receiver pathway, with quantitative angle- and
frequency-dependent reflection tests before adding geological complexity.

## Reproduction and regression

Use the existing binary environment/locks: DOLFINx 0.11.0, real-double PETSc,
matching mpi4py/MPICH, and the repository's dev/xarray dependencies. The measured
stack is Python 3.14.6, Basix 0.11.0, UFL 2026.1.0, PETSc 3.25.5,
MPICH 5.0.1, NumPy 2.5.3, and Pydantic 2.13.5. On the
implementation machine the active environment is
`/home/m/packages/anaconda3/envs/fenicsx0.11`. From the repository root:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -ra
python -m pytest -q -s tests/experiments2d/test_science.py
python -m pytest -q -s tests/experiments2d/test_mpi.py
mpiexec -n 1 python -m tests.experiments2d.mpi_worker /tmp/seisfem-experiment-1.npz
mpiexec -n 2 python -m tests.experiments2d.mpi_worker /tmp/seisfem-experiment-2.npz
mpiexec -n 4 python -m tests.experiments2d.mpi_worker /tmp/seisfem-experiment-4.npz
python examples/2d/homogeneous.py --cells 240
python examples/2d/homogeneous.py --cells 640
ruff check .
ruff format --check .
pre-commit run --all-files
# Include unstaged, staged and whole-branch whitespace checks when reviewing:
git diff --check
git diff --cached --check
git diff 5fa7ee9..HEAD --check
```

Run pytest once, not under mpiexec: marked MPI tests launch their own workers.
The complete suite includes every existing 1D and 2D-core regression unchanged.
Validation outcomes on 2026-09-14:

| Command / selection | Outcome |
|---|---|
| `python -m pytest -ra` | **204 passed in 573.18 s**: 98 original 1D, 65 existing 2D core, 41 new cases |
| New MPI suite, explicit 1/2/4-rank worker launches | **2 comparison tests passed**; normalization, ordering, arrivals, reciprocity and coherent errors included |
| Final affected-test rerun (command below) | **39 passed, 2 deselected in 21.42 s**, without warnings |
| `python examples/2d/homogeneous.py --cells 240` | Completed; P=0.47157548 s, S=0.71193570 s, both result shapes `(1901,3,2)` |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `pre-commit run --all-files` | Both Ruff hooks passed |
| Working, staged and whole-branch `git diff --check` | Passed |

The full run revealed 376 warnings from the deprecated `geometry.dofmap` getter.
The final correction uses `geometry.dofmaps[0]`. Inspection of the installed
DOLFINx source confirmed both access exactly the same underlying geometry array;
only the former emits a warning. Source/receiver, workflow, reciprocity, and
serial/2/4-rank tests were rerun after this API-only change with no warnings:

```bash
python -m pytest -ra tests/experiments2d -k \
  'not p_and_s_arrivals_and_radiation and not fixed_physical_point_source_refinement'
```

The two costly wave/refinement cases passed in the complete run; their numerical
path is unchanged by that equivalent getter replacement. No old test, core
operator, 1D runtime, or central-difference implementation was modified.
