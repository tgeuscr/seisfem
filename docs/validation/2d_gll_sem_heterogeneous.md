# Element-aligned heterogeneous isotropic GLL SEM

This milestone extends the assembled 2D GLL backend to **horizontal isotropic
layers with one constant material per element and every interface on an element
edge**. It retains degrees 1–6, affine structured rectangles, positive-up z,
free exterior boundaries, physical point forces/receivers, explicit central
differences and MPI. The baseline is `d87afcdc961f03e0cfd146b66309ecaba37b3fd6`.
The independent operator and normal-incidence evidence below validates this
scope; no oblique SEM scattering claim is made here.

## Configuration and assignment

The existing layered material API is reused without a new material schema:

```python
from seisfem.config2d import PlaneStrainConfig

config = PlaneStrainConfig.model_validate(
    {
        "domain": {"lower": [-1000, -1000], "upper": [1000, 1000], "cells": [8, 8]},
        "discretization": {"type": "quad_gll", "degree": 4},
        "material": {
            "type": "layered",
            "layers": [
                {"lower": -1000, "upper": 0, "material": {"density": 2200, "vp": 3000, "vs": 1700}},
                {"lower": 0, "upper": 1000, "material": {"density": 2500, "vp": 4000, "vs": 2300}},
            ],
        },
    }
)
```

Layers must cover the domain, in increasing z order, without gaps or overlaps.
Density/speeds or density/Lamé moduli retain existing material validation:
finite values, rho>0, mu>0 and lambda+2mu/3>0. Configuration requires each
interface's mesh-row coordinate to lie within `64*ulp(nz)` of an integer.
In physical units this tolerance is `64*ulp(nz)*(zmax-zmin)/nz`. A cut interface
raises an actionable error asking the user to adjust mesh or layer depths.

`CellMaterials2D` is the single material-to-cell assignment stage. It selects
a layer from each cell's physical center, then independently checks **all cell
vertices** against that layer, with tolerance
`64*eps*max(1, abs(zmin), abs(zmax))`. A failed containment check is collective.
This prevents center sampling from accepting a cut element. Layer IDs and
rho/lambda/mu are assigned per local cell, including ghosts; forward scattering
synchronizes the coefficient DOFs. No local cell index is a cross-rank identifier.

SEM supplies a Basix tensor-product, discontinuous Q0 space to this existing
mapper. Its scalar basis is constant one, with one coefficient per element.
The triangular caller retains its original DG0 construction. Homogeneous SEM
retains its original scalar-coefficient branch. Omitted discretization still
selects triangular P1; there is no fallback from an unsupported SEM model.

## Formulation and factorization

For every element, mu=rho*Vs² and lambda=rho*(Vp²−2Vs²). The weak forms are

```
M(w,u) = integral rho w·u dx
K(w,u) = integral [lambda div(w) div(u) + 2 mu epsilon(w):epsilon(u)] dx
```

The displacement space is continuous tensor-product GLL Lagrange, and the
quadrature remains GLL of degree 2p−1. Since the coefficient is constant on
each element, collocation produces diagonal elemental mass. Contributions
from both sides of an interface add at shared displacement DOFs. **No nodal
material averaging and no row-sum lumping are used.** Welded displacement
continuity is conforming; traction continuity follows from the weak form.
There are no explicit interface forces, penalties or traction terms.

Both forms retain `form_compiler_options={"sum_factorization": True}`. The Q1
coordinate element and Q0 coefficient element both expose tensor-product
factorization metadata. This is sum-factorized form evaluation with an
**assembled global PETSc stiffness matrix**, not matrix-free time stepping.
No integrator, boundary implementation or triangular numerical assembly changed.

## Independent operators, stability and energy

The NumPy-only reference constructs GLL abscissae/weights from Legendre
polynomials, differentiates Lagrange polynomials, integrates elastic element
matrices and scatters them into global matrices. Material selection uses
physical vertex containment and supplied density/speeds, never production
DG0 values, UFL or FFCx. Tests compare complete M/K and an arbitrary K*x.

The operator experiment uses a 2×2 rectangle mesh on [-1,2]×[-1,1], with
(rho,Vp,Vs)=(2.3,3.2,1.8) below zero and (4.1,4.6,2.5) above. These materials
have general nonzero lambda and exercise coupled plane-strain elasticity.
Each displacement component has integrated mass 19.2.

| p | M reference | K reference | K*x reference | critical dt | safe dt |
|---|---:|---:|---:|---:|---:|
| 1 | 5.125e-16 | 5.100e-16 | 3.903e-16 | 0.21821143 | 0.19227288 |
| 2 | 1.055e-15 | 9.226e-16 | 9.360e-16 | 0.08575071 | 0.07274926 |
| 3 | 2.529e-15 | 1.895e-15 | 1.794e-15 | 0.04724197 | 0.03780645 |
| 4 | 1.218e-14 | 8.233e-15 | 8.186e-15 | 0.02934881 | 0.02307973 |
| 5 | 6.778e-14 | 5.102e-14 | 4.872e-14 | 0.01984032 | 0.01552706 |
| 6 | 1.981e-13 | 1.403e-13 | 1.397e-13 | 0.01427420 | 0.01115011 |

Matrix errors are relative Frobenius norms; action errors are relative vector
2-norms. The p=6 reference error remains below 2e-13; polynomial-coefficient
arithmetic in the independent NumPy reference contributes to its growth with p.
The existing operator tolerances are retained (mass 3e-11, stiffness/action
3e-10), without claiming those tolerances are observed errors.

Across p=1–6, mass off-diagonals are at most 7.37e-16 of the largest diagonal,
integrated-mass relative error is at most 2.56e-15, stiffness symmetry residual
is at most 1.63e-16 and rigid-translation residual at most 7.85e-16. The latter
two use maximum absolute matrix entry normalization, as in the existing suite.
All mass entries are positive. The symmetric mass-scaled stiffness eigenvalues
are nonnegative to roundoff, using the existing 1e-12 relative eigenvalue gate.

The independent dense eigenvalue calculation uses D^(-1/2) K D^(-1/2).
Reported critical dt=2/sqrt(lambda_max) agrees within 7.78e-16 relative. The
existing conservative row bound remains below critical dt without modification.
For 200 source-free steps at half the conservative bound, the half-step energy

```
E[n+1/2] = 0.5 v[n+1/2]^T D v[n+1/2] + 0.5 u[n+1]^T K u[n]
```

has relative peak-to-peak drift at most 1.37e-15. Longer interface runs below
also pass the unchanged 3e-13 energy tolerance. This is the central-difference
invariant, not displacement squared or integer-time uncorrected energy.

## Homogeneous limit and compatibility

For p=1,2,4,6, assigning the same material to both layers produces **zero
measured difference** from the homogeneous SEM path in M, K, diagonal mass,
conservative dt, public point-force displacement histories and velocity
histories. Sources and receivers are off-node; source directions have both
components. No source/receiver implementation changes were needed.

The archived-main and archived-`d87afcd` triangular compatibility snapshots are
bitwise identical to the new checkout for M, K, diagonal mass, stable dt,
stiffness action, source vector, displacement histories and velocity histories.
All eight maximum differences are zero. A separate archived-`d87afcd`
heterogeneous triangular operator/material/history snapshot also matches bitwise.
The existing default-versus-explicit
tri_p1 test also passes. The old SEM rejection example for a valid layered
model has become a cut-layer rejection example because aligned layers are now
intentionally supported; no numerical tolerance was loosened.

## Normal-incidence analytical experiment

The domain is [-400,400]×[-4000,4000] m with interface z=0. The pulse starts at
z=-1600 m and travels toward +z. The two media are:

| medium | rho (kg/m³) | Vp (m/s) | Vs (m/s) | Zp (kg/m²/s) |
|---|---:|---:|---:|---:|
| A | 2000 | 2000 | 1414.213562373095 | 4,000,000 |
| B | 2400 | 3000 | 2121.320343559642 | 7,200,000 |

Vs=Vp/sqrt(2) makes lambda zero to floating-point roundoff. This is an
admissible isotropic material choice, not a scalar replacement for the elastic
kernel. It makes the uniform-in-x longitudinal packet satisfy the free vertical
side tractions exactly, avoiding side returns without an enormous domain or
unsupported constraints. General nonzero-lambda contrasts are independently
validated by the operator and MPI experiments. An initial wide-domain probe
with nonzero lambda exposed premature numerical side contamination on very
coarse horizontal cells; it is not used for the formal measurements.

The reused analytical initial packet has
`s=(z+1600)/w`, `w=Vp_inc/(pi*6 Hz)`,
`u_z=(1−2s²) exp(−s²)`, and `v_z=−Vp_inc*d(u_z)/dz`.
The other components vanish. This is a smooth finite longitudinal pulse in z,
uniform across the strip, not a localized oblique beam. It uses prescribed
initial fields through the production operator start/step path. Duration is
1.4 s; dt is duration divided by enough steps to stay below 0.15 of the assembled
conservative bound. Top/bottom reflected arrivals are outside this window;
even an initial downward startup component in the fastest incident medium
would require about 1.87 s to reflect at the bottom and reach the lower receiver.

Receivers at (17.3,−800.3) and (17.3,800.3) m separate the incident, reflected
and transmitted packets. Each amplitude is the signed projection onto the
independent analytical pulse in a ±0.8/6 s window around its analytical arrival,
then normalized by the measured incident amplitude. For A→B the arrival centers
are 0.39985, 1.20015 and 1.0667667 s. For B→A they are 0.2665667, 0.8001 and
0.9334833 s. Arrival errors use a quadratic interpolation of the pulse extremum.
Waveform errors are L2 differences from the analytical coefficient times the
pulse, normalized by the unit template L2 norm.

All coefficients use the **fixed +z displacement component**, including the
reflected pulse. Thus displacement continuity gives 1+R=T; opposite propagation
directions in traction give Z1(1−R)=Z2*T. Consequently

```
R = (Z1−Z2)/(Z1+Z2),   T = 2Z1/(Z1+Z2)
F_R = R²,             F_T = (Z2/Z1) T².
```

For A→B, R=−2/7 and T=5/7; for B→A, R=2/7 and T=9/7. Both have
F_R=4/49 and F_T=45/49; analytical flux closure is within 3e-16. These are
normal-incidence elastic energy-flux fractions, not raw squared transmission.
For this nondispersive 1D continuum solution, the exact template is appropriate
at every pulse frequency; no finite-beam angle/Jacobian correction is required.

| case | h_z (m) | R measured | T measured | abs R error | abs T error | flux closure error |
|---|---:|---:|---:|---:|---:|---:|
| increase | 100 | -0.2857754246 | 0.7141562576 | 6.114e-05 | 1.295e-04 | 2.979e-04 |
| increase | 50 | -0.2857139709 | 0.7142833610 | 3.148e-07 | 2.353e-06 | 6.231e-06 |
| decrease | 100 | 0.2857718761 | 1.2858812351 | 5.759e-05 | 1.669e-04 | 2.714e-04 |
| decrease | 50 | 0.2857139940 | 1.2857184938 | 2.917e-07 | 4.208e-06 | 5.845e-06 |

At p=4, h_z=50 m, the worst arrival error is below 1.57e-5 s, transverse receiver
displacement is below 7.1e-14 of the unit incident displacement, and flux closure
error is below 6.24e-6. Refinement improves both signed amplitudes and waveform
errors. No convergence order or exponential convergence claim is fitted.

At p=6, h_z=50 m with identical materials, the measured reflected coefficient
is −3.22e-13 and transmission is 0.9999999624. A homogeneous control produces
matching displacement and velocity histories to roundoff. Remaining transmitted
waveform error includes central-difference temporal error; a missing interface
is not claimed to remove propagation dispersion. The raw reflected coefficient
is reported without subtracting a control trace.

| triangular h_z (m) | R measured | T measured | abs R error | abs T error | reflected arrival error (s) | transmitted waveform error |
|---|---:|---:|---:|---:|---:|---:|
| 10 | -0.2843531881 | 0.7094924495 | 1.361e-03 | 4.793e-03 | 4.372e-03 | 8.059e-02 |
| 5 | -0.2858882630 | 0.7138412964 | 1.740e-04 | 4.444e-04 | 1.103e-03 | 2.012e-02 |

The triangular comparison uses the same physical materials, pulse, free domain,
receivers, duration and analytic diagnostic. Its four horizontal strips and
vertical refinements differ from SEM's two horizontal elements. Both approaches
converge toward the independent impedance solution; triangular FEM is not used
as the reference truth. Small transverse P1 interpolation artifacts decrease
with refinement. This is a physics comparison, not an equal-work benchmark.

## MPI

A p=4, 8×8 mesh on [-1,1]² uses three layers: LOW below −0.25, HIGH between
−0.25 and 0.25, LOW above 0.25, with the general coupled materials from the
operator experiment. The central layer is split across ranks, and each material
is present on multiple ranks. Owned layer IDs are [0,1]/[1,2] for two ranks,
and [0,1]/[0,1]/[1,2]/[1,2] for four ranks. This exercises distributed material
assignment, shared material-interface nodes and coefficient ghost updates.

Public off-node point-force histories and a separate smooth initialized run
are compared by physical coordinates. Cell centers and material IDs/properties
are also ordered physically. There is no comparison of local PETSc numbering.

| relative difference from serial | 2 ranks | 4 ranks |
|---|---:|---:|
| cells | 0.000e+00 | 0.000e+00 |
| u | 1.473e-14 | 1.027e-14 |
| v | 1.098e-14 | 1.074e-14 |
| safe_dt | 0.000e+00 | 0.000e+00 |
| dtcrit | 3.713e-16 | 3.713e-16 |
| mass | 0.000e+00 | 1.339e-17 |
| action | 3.557e-14 | 3.373e-14 |
| final_u | 1.377e-14 | 3.051e-14 |
| final_v | 4.701e-13 | 4.113e-13 |

All comparisons satisfy the existing 3e-11 global-scale MPI criterion.
Near-zero K*x entries are compared using the whole field's scale, avoiding a
meaningless pointwise relative error. The conservative dt is identical across
ranks; spectral dt differs only by eigensolver/reduction roundoff.

## Reproduction and limitations

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SEISFEM_SEM_REPORT_DIR=/tmp/sem-het \
  python -m pytest -ra tests/sem2d
python -m tests.sem2d.heterogeneous_report /tmp/sem-het \
  docs/validation/2d_gll_sem_heterogeneous_measurements.json
```

The collection command also expects the archived P1 compatibility record,
generated by running `tests/sem2d/compatibility.py` against each archived src
and the current src, then comparing all eight npz arrays bitwise. The
[machine-readable audit](2d_gll_sem_heterogeneous_measurements.json) contains
operator, homogeneous-limit, wave, stability, energy, MPI and P1 results.

Final acceptance gates completed on 2026-09-25:

| gate | result |
|---|---:|
| focused operators/configuration/factorization/MPI | 28 passed, 1.93 s |
| analytical interface/refinement/control tests | 4 passed, 59.00 s |
| complete SEM suite | 82 passed, 546.55 s |
| seven existing 2D suites | 283 passed, 1857.00 s |
| complete repository regression | **504 passed, 2461.49 s** |
| Ruff check and format, all-file pre-commit, whitespace checks | passed |

No existing numerical acceptance threshold was weakened. The
[execution audit](2d_gll_sem_heterogeneous_checks.txt) records commands and
compatibility provenance. The homogeneous research artifacts remain unchanged.

Unsupported: within-element jumps, non-horizontal material geometry, VTI or
other anisotropy, absorbing/fixed/componentwise SEM boundaries, curved or
non-affine quadrilaterals, PML, attenuation, poroelasticity, matrix-free
execution and 3D. Oblique scattering is not added by this milestone. The
finite-element weak form is general within the supported isotropic layers,
but the new propagation evidence is normal P incidence. No spectral-convergence
claim is made across discontinuous coefficients. No performance superiority
claim follows from this small interface validation.
