# Heterogeneous exterior impedance boundaries

This milestone adds a **local first-order isotropic impedance absorber** to the
production horizontal-layered 2D plane-strain solver, starting from
`f467b1d6de3178391c61e9d92a58fe36ef04d64b` (the main commit containing the validated
oblique-interface work). It is **not a PML or an exact oblique nonreflecting
condition**. The homogeneous expression and assembly path are preserved.

The validated scope is isotropic elasticity, vector P1 plane strain, DG0
material fields, horizontal mesh-aligned layers, and the existing structured
rectangular triangular meshes. No time integrator, source physics, internal
interface term, or material state was added.

## Boundary condition and discretization

On an absorbing exterior, with outward unit normal n,

\[
\sigma(u)n+B(x)\dot u=0,\qquad
B=Z_s I+(Z_p-Z_s)n\otimes n,
\]

\[
Z_s=\sqrt{\rho\mu}=\rho V_s,\qquad
Z_p=\sqrt{\rho(\lambda+2\mu)}=\rho V_p.
\]

For a heterogeneous model, the expression uses the existing synchronized DG0
rho/lambda/mu Functions. Each exterior facet has one adjacent cell, whose
material trace supplies its impedance. A vertical side crossing layer contacts
therefore has piecewise material-dependent damping. No redundant Vp/Vs Function
is created. The homogeneous path retains its original expression
`rho * (Vs I + (Vp-Vs) outer(n,n))` exactly.

The consistent weak operator is

\[
C(w,v)=\int_{\Gamma_{abs}}w\cdot B(x)v\,ds.
\]

The existing facet tags, consistent matrix assembly, reverse-add accumulation,
and direct row-sum integration are unchanged. For an edge of length L in one
material, each Cartesian component contributes
`Z L/6 [[2,1],[1,2]]`; its lumped endpoint weights are `Z L/2`.
On vertical facets the components are `(Zp,Zs)`, and on horizontal facets they
are `(Zs,Zp)`. Contact vertices and corners receive the sums of adjacent edge
contributions. This is particularly important where a vertical wall crosses a
layer interface: no single material is assigned to that shared displacement DOF.

Axis-aligned normals make B diagonal and positive; P1 traces are nonnegative
and form a partition of unity. Consequently the row-sum diagonal C_L is
nonnegative. This argument is restricted to the supported geometry and element
space. Valid negative lambda is supported: positivity requires the existing
solid-material conditions, not lambda >= 0.

The production update remains

\[
(D+\tfrac{\Delta t}{2}C_L)u^{n+1}
=2Du^n-(D-\tfrac{\Delta t}{2}C_L)u^{n-1}
+\Delta t^2(f^n-Ku^n).
\]

D is the existing lumped volume mass. The stiffness-based assembled stability
bound and damped Taylor startup are unchanged. Welded internal interfaces still
use the conforming volume weak form; no interior-facet traction term is added.

A free side contributes no damping. A side still cannot be both absorbing and
essentially constrained. Constraints on another side can meet an absorber at a
corner; constrained components remain zero. Positive-up z means `upper` is the
physical top and `lower` is the bottom.

## Independent assembly and homogeneous-limit checks

The focused suite is in [tests/heterogeneous_absorbing2d](../../tests/heterogeneous_absorbing2d/).
Its edge reference builds the exact matrix from physical edge midpoints and
configured rho/Vp/Vs, independently of DG0 storage and the production impedance
expression. It checks all four sides and all four triangle diagonal patterns.
The test rectangle is `[-1,2] x [-3,3]`, with 6 x 12 cells and contacts at
z=-1 and z=1. From bottom to top, `(rho,Vp,Vs)` are `(2.3,3.2,1.8)`,
`(4.1,4.6,2.5)`, and `(1.9,2.8,1.4)`.

This catches a global side impedance, wrong layer assignment, swapped P/S
components, incorrect edge weights, and duplicated/missing MPI facet
contributions. Checks include consistent C, row sums, symmetry, nonnegative
quadratic forms/eigenvalues to roundoff, free-top interior nodes, and constrained
corners. A separate auxetic case exercises lambda=-0.5, mu=1.2, rho=2.3.

Identical homogeneous physics is described by an Isotropic model, one DG0 layer,
and three equal-property DG0 layers. M, K, lumped mass, consistent C, lumped C_L,
stable dt, and two-second receiver displacement/velocity histories are compared.
A separate before/after capture of the original homogeneous path found M, K, C,
mass, damping, dt, displacement, and velocity **bitwise unchanged** in this
locked environment. This is an observed result, not a cross-platform guarantee.
That capture used the existing homogeneous absorber fixture: rectangle
`[-1,2] x [2,4]`, 6 x 4 cells, `(rho,Vp,Vs)=(2.3,3.2,1.8)`, absorbing
left/right/bottom, dt=0.01 s, and duration=0.3 s. Its force was at `(0.17,2.73)`
with direction `(1,-2)`, Ricker f0=5 Hz, amplitude=2, time shift=0.08 s, and
receiver `(0.41,3.23)`.

<!-- OPERATOR_RESULTS -->

| Quantity | One layer: relative error | Three equal layers: relative error |
|---|---:|---:|
| M | 0.000e+00 | 0.000e+00 |
| K | 0.000e+00 | 0.000e+00 |
| mass | 0.000e+00 | 0.000e+00 |
| C | 1.810e-16 | 1.810e-16 |
| damping | 1.207e-16 | 1.207e-16 |
| dt | 0.000e+00 | 0.000e+00 |
| u | 0.000e+00 | 0.000e+00 |
| v | 0.000e+00 | 0.000e+00 |

Across 16 facet/diagonal cases, maximum absolute consistent-matrix error is `1.554e-15` and maximum lumped-diagonal error is `2.442e-15`. Every assembled damping diagonal is nonnegative.

<!-- END_OPERATOR_RESULTS -->

## Local-layer packet experiments

The physical packet model has one interface at z=0:

| Medium | rho (kg/m³) | Vp (m/s) | Vs (m/s) |
|---|---:|---:|---:|
| Lower | 2400 | 3200 | 1800 |
| Upper | 2800 | 4000 | 2200 |

The rectangle is `[0,2400] x [-4800,4800] m`, with h=20 m. The right side spans
both materials. Right-boundary packets target z=-2400 or +2400 m; bottom packets
target `(1200,-4800)`. Thus the tested packets stay inside their local material
before unrelated interface returns can affect the receiver window. The measured
agreement with a homogeneous model provides a quantitative check of this
isolation, including small Gaussian tails and any discrete mode contamination.

The scalar potential is `Phi=s exp[-(s/w)²-(q/W)²]`, with
`w=c_inc/(pi*8 Hz)` and `W=400 m`. P uses grad Phi; SV uses `(Phi_z,-Phi_x)`.
Initial velocity is the consistent translation `v=-c_inc partial_s u`.
These are test-only initial data passed through `PlaneStrainOperators.start()`;
no public source redesign is involved.

The beam starts 1200 m along the incident ray before the target boundary point.
A receiver is 400 m along the specular reflected ray. dt=0.002 s is identical
across all controls and passes the production assembled stability check. The
measurement window is `|t-1600/c_inc| <= 0.8/8 s`. A finite transverse packet
contains an angular spectrum even when its central ray is normal.

For **each** case and each representation (layered and homogeneous local
material), run absorbing, free, and extended-boundary controls. In the extended
control the tested wall moves 2400 m outward, preserving h, source, receiver,
and material contact locations. Subtract the extended trace to isolate the
boundary return, then report

\[
R_{proxy}=\frac{\max_{window}\|u_{abs}-u_{extended}\|}
{\max_{window}\|u_{free}-u_{extended}\|}.
\]

This is a vector displacement reflection **proxy relative to the free control**,
not a signed plane-wave coefficient, a total reflected-energy fraction, or a
claim of exact absorption. Mesh dispersion and finite angular bandwidth remain
in the measured values. Both displacement and velocity histories are also
compared directly between layered and homogeneous local-material runs.

<!-- PACKET_RESULTS -->

| Mode | Layer / wall | Angle | Free scattered peak (m) | Layered proxy | Homogeneous proxy | Proxy difference |
|---|---|---:|---:|---:|---:|---:|
| P | lower / right | 0° | 0.71812133 | 0.01126746 | 0.01126746 | 0.000e+00 |
| P | upper / right | 0° | 0.68757609 | 0.00878671 | 0.00878671 | 0.000e+00 |
| SV | lower / right | 0° | 0.68413728 | 0.04035929 | 0.04035929 | 6.939e-17 |
| SV | upper / right | 0° | 0.66366830 | 0.03661159 | 0.03661159 | 8.674e-16 |
| P | lower / lower | 0° | 0.71812133 | 0.01126746 | 0.01126746 | 0.000e+00 |
| SV | lower / lower | 0° | 0.68403557 | 0.04034052 | 0.04034052 | 0.000e+00 |
| P | upper / right | 15° | 0.62740352 | 0.01178270 | 0.01178270 | 0.000e+00 |
| P | upper / right | 30° | 0.65706396 | 0.04014707 | 0.04014707 | 0.000e+00 |
| P | upper / right | 45° | 0.75020878 | 0.08714646 | 0.08714646 | 0.000e+00 |

Maximum direct history differences relative to the homogeneous trace peak are `1.656e-12` for displacement and `1.872e-12` for velocity. These are much smaller than the reflection being characterized.

<!-- END_PACKET_RESULTS -->

Normal P/SV tests hit a vertical side in **each** material, and the horizontal
bottom in the lower material. Together with the three-layer matrix/MPI tests,
these exercise a genuine wall spanning multiple materials. Oblique P cases use
the upper material and 15, 30, and 45 degree incidence from the outward normal.
The rising reflection is expected for this local condition; equality with the
homogeneous local-material control, rather than zero reflection, is the
heterogeneous-extension acceptance criterion. These experiments do not validate
critical/evanescent interface scattering or an exact arbitrary-angle absorber.

## Dissipation and stability

For source-free updates, the existing half-step energy obeys

\[
E^{n+1/2}-E^{n-1/2}=-\Delta t\,(v^n)^T C_L v^n,\qquad
v^n=(u^{n+1}-u^{n-1})/(2\Delta t),
\]

where
`E^(n+1/2)=0.5 ((u^(n+1)-u^n)/dt)^T D ((u^(n+1)-u^n)/dt)`
`+ 0.5 (u^(n+1))^T K u^n`.
The three-layer test uses random initial u/v, a free/essential top, absorbing
left/right/bottom, dt=0.8 times the assembled bound, and 1000 steps. It checks
nonnegative loss, nonincreasing nonnegative energy to roundoff, and the exact
discrete work balance. This is stronger than checking that a trace looks stable.

<!-- ENERGY_RESULTS -->

dt is `0.058377417150 s`. The maximum per-step balance residual divided by initial energy is `1.807e-16`; the cumulative residual is `2.313e-16`. Final energy is `5.23023180e-05` of initial energy after 1000 steps. No energy growth or negative damping was observed.

<!-- END_ENERGY_RESULTS -->

## MPI and production surface/well example

MPI checks use 1, 2, and 4 ranks. The small three-layer mesh compares physical
cell locations/materials (including DG0 ghost checks), M, K, C, full owned mass
and damping diagonals, distributed C action, and stable dt. Each rank also checks
the global C against the independent edge reference. Matrices and fields are
ordered by physical coordinates, never by assumed local cell numbering.

The public example supplies the MPI displacement/velocity histories:

```bash
python examples/2d/layered_absorbing.py
mpiexec -n 4 python examples/2d/layered_absorbing.py
```

It uses `SimulationConfig2D`/`Simulation2D`, a small 60 x 60 rectangular mesh,
three horizontal layers, an ordinary vertical line force near the surface, two
surface receivers and three well receivers. The top is free; left, right and
bottom are absorbing. There is no custom boundary matrix or exploratory-script
dependency. The test verifies zero top-interior damping and compares the free
top with an absorbing-top control to confirm that the physical free-surface
signal remains meaningful.

<!-- MPI_EXAMPLE_RESULTS -->

| MPI ranks | Relative C error | Relative C_L error | Relative dt error | Relative C-action error | Relative displacement error | Relative velocity error |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.499e-17 | 2.013e-14 | 2.594e-14 |
| 4 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.866e-16 | 1.627e-14 | 2.594e-14 |

All MPI comparisons use `max absolute difference / peak serial value`, with a `3e-12` acceptance limit. Material values and layer IDs agree exactly. C, C_L, dt, and receiver coordinates are bitwise identical in these MPI runs.

The example has contacts at z=-1200 and -400 m in a domain extending to z=-2400 m. Bottom/middle/top `(rho,Vp,Vs)` are `(2600,3800,2100)`, `(2400,3200,1800)`, and `(2100,2800,1500)` in SI units. dt=0.001 s and duration=1.2 s.

| Receiver | Peak ux (m) | Peak uz (m) | Peak vx (m/s) | Peak vz (m/s) |
|---|---:|---:|---:|---:|
| surface-left | 8.39271124e-04 | 2.41238383e-03 | 3.08995598e-02 | 9.10888780e-02 |
| surface-right | 7.97010625e-04 | 1.49998604e-03 | 2.27735952e-02 | 4.31925415e-02 |
| well-shallow | 5.60702991e-04 | 1.13644799e-03 | 2.41684906e-02 | 4.75511456e-02 |
| well-middle | 1.62348734e-04 | 2.27042840e-04 | 6.30137026e-03 | 8.85808969e-03 |
| well-deep | 4.01894823e-05 | 1.54089863e-04 | 1.14685060e-03 | 5.84904552e-03 |

Changing only the top to absorbing changes the surface displacement histories by `0.81280909` relative norm. The free-top run has exactly zero damping at all 59 top-interior nodes; endpoint contributions from side walls are retained.

<!-- END_MPI_EXAMPLE_RESULTS -->

Local surface/well scripts can later set the normal production `boundaries`
mapping and remove their experimental facet-impedance assembly, manual damping
replacement, and layered-boundary validation bypasses. Their remaining geometry,
sampling, visualization, and scientific interpretation still require their own
review. No untracked `scripts/` or `runs/` file was changed or committed here.

## Reproduction and regression

In the repository's locked DOLFINx 0.11 / real-double PETSc environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SEISFEM_HETERO_ABSORBER_REPORT_DIR=/tmp/ha-report \
  python -m pytest -ra tests/heterogeneous_absorbing2d
python -m tests.heterogeneous_absorbing2d.report /tmp/ha-report \
  docs/validation/2d_heterogeneous_absorber_measurements.json
```

Tests always recompute simulations and analytic edge matrices; committed JSON
is evidence, never a test oracle. The packet fixture runs each case once per
pytest invocation and shares it across the acceptance assertions.
The [measurements JSON](2d_heterogeneous_absorber_measurements.json) preserves
unrounded values; the [check log](2d_heterogeneous_absorber_checks.txt) records
commands and regression results.

Validation results in this environment:

- Focused heterogeneous absorber suite: **33 passed in 222.01 s**.
- Existing absorber, heterogeneous-material, and oblique-interface suites: **89 passed in 762.09 s**.
- Complete repository regression: **326 passed in 1296.25s (0:21:36)**, including MPI subprocess tests.
- Ruff, formatter, pre-commit, and whitespace checks passed.

The measurements above were collected during the complete regression. They reproduce the focused-run measurements exactly.

The old categorical rejection case was replaced by the still-invalid
same-side absorbing/essential conflict and comprehensive positive layered
absorber tests. No physical regression threshold was weakened or test removed.
Historical v0.5.0 material results retain their original scope; documentation
now links this separate extension.

## Limits

This is a **local first-order isotropic impedance absorber**. Continuum
normal-incidence plane P/SV waves in a locally homogeneous isotropic medium
satisfy it exactly. Finite beams, discrete P1 waves, and oblique incidence retain
reflection; this work demonstrates consistency with the validated homogeneous
condition and the absence of a new layered-boundary artifact in the stated tests.

It is **not** a PML, an exact oblique nonreflecting condition, or validated for
anisotropy, attenuation/Q, poroelasticity, arbitrary heterogeneous geometry,
dipping/non-planar material interfaces, higher-order elements, or
critical/evanescent branches. No claim is made about arbitrary corner/interface
interaction beyond the assembled local condition and the tested cases.
