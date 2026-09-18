# Vertical-axis VTI plane-strain elasticity

This milestone extends the production 2D vector P1 solver to **VTI plane strain
with the symmetry axis along physical z**. Coordinates are (x,z), z is positive
up, displacement is (ux,uz), and SI units apply. The baseline is main/v0.6.0 at
`bbf07f5dbc2f639741b20c0f1620fce96ddc7677`.

The evidence establishes homogeneous qP/qSV propagation, the isotropic limit,
horizontal layered material assembly, stability, and MPI consistency. It does
**not** establish quantitative anisotropic interface-scattering amplitudes.
**VTI configurations reject all absorbing boundaries.** The existing isotropic
impedance condition has not been generalized to anisotropy.

This report retains the original v0.7.0 scope and measurements. The subsequent
[isotropic/VTI interface validation](2d_vti_interface.md) separately validates
below-critical propagating scattering for isotropic P incidence at a single
horizontal welded interface; it does not generalize the absorbing boundaries.

## Constitutive law and API

With epsilon = sym(grad u) and epsilon_yy=0,

\[
\sigma_{xx}=C_{11}\epsilon_{xx}+C_{13}\epsilon_{zz},\quad
\sigma_{zz}=C_{13}\epsilon_{xx}+C_{33}\epsilon_{zz},\quad
\sigma_{xz}=2C_{55}\epsilon_{xz}.
\]

The engineering-strain vector is e=(epsilon_xx,epsilon_zz,2epsilon_xz), with

\[
H=\begin{pmatrix}C_{11}&C_{13}&0\\C_{13}&C_{33}&0\\0&0&C_{55}\end{pmatrix},
\qquad W=\tfrac12 e^T H e.
\]

The symmetric weak stiffness is `integral strain(w):sigma(u) dx`. Density enters
the unchanged vector mass form. Row-sum mass lumping, central differences,
startup, constraints, source normalization, and receiver interpolation retain
their existing implementations. No interface traction term is added: conforming
displacement and the elastic weak form supply welded-interface coupling.

The exact new material mapping is:

```python
material = {
    "type": "vti",  # required explicit discriminator
    "density": 2500.0,  # kg/m^3
    "c11": 40e9,  # Pa
    "c33": 25e9,
    "c13": 9e9,
    "c55": 7e9,
}
```

Use this as `SimulationConfig2D.material`, or in any layer's `material` field:

```python
material = {
    "type": "layered",
    "layers": [
        {"lower": -1000, "upper": 0, "material": {"density": 2200, "vp": 3000, "vs": 1700}},
        {
            "lower": 0,
            "upper": 1000,
            "material": {
                "type": "vti",
                "density": 2500,
                "c11": 40e9,
                "c33": 25e9,
                "c13": 9e9,
                "c55": 7e9,
            },
        },
    ],
}
```

Layers must cover the domain, be ordered in increasing z, and meet exactly on
horizontal mesh rows. Mixed models use DG0 rho/C11/C33/C13/C55 fields; isotropic
layers are canonicalized only inside this new assembly path. Entirely isotropic
models retain their original classes, rho/lambda/mu fields, constitutive form,
and absorber assembly. DG0 ghost values are synchronized before assembly.

Free boundaries and fixed/componentwise-fixed constraints are supported.
Even an isotropic-limit material explicitly described as VTI rejects absorption;
use the existing isotropic input for validated isotropic absorbers. A generic
point line force excites both eigenbranches; it is not a pure qP/qSV source.

There is no Thomsen convenience input in this milestone. Canonical stiffnesses
are the only anisotropic constitutive representation. C66, gamma, tilt, and
unknown material keys are rejected. This in-plane model neither specifies nor
tests a full 3D TI tensor or independent SH/gamma sensitivity.

## Admissibility and stability

The sufficient in-plane energy conditions are

\[
\rho>0,\quad C_{11}>0,\quad C_{33}>0,\quad C_{55}>0,\quad
C_{11}C_{33}-C_{13}^2>0.
\]

C13 may be positive, zero, or negative. Values must be finite. The implementation
normalizes stiffnesses by S=max(C11,C33,abs(C13),C55), computes the two eigenvalues
of the normalized normal-strain block using its determinant and largest
eigenvalue, and requires both its smallest eigenvalue and C55/S to exceed
`64 * ulp(1.0)` (about 1.42e-14). This deliberately excludes numerically singular
materials even if their exact-arithmetic determinant is positive. Squared-speed
scales and the stiffness sum used to guard overflow must remain finite and
positive. Invalid inputs are rejected, never clamped.

The existing bound on the absolute row sums of `D_f^-1/2 K_ff D_f^-1/2` remains
applicable to a symmetric VTI stiffness. No velocity-based CFL replacement is
introduced. Small-matrix tests compare this bound with an independent dense
eigensolve, test a 30:1 stiffness contrast, and verify the expected factor-two
decrease in dt when all stiffnesses increase fourfold at fixed density.

Source-free runs take 1000 steps at 0.8 times the assembled bound. Componentwise
anchors (left ux and bottom uz) remove rigid motion for a well-conditioned
energy audit. The conserved central-difference half-step energy is

\[
E^{n+1/2}=\tfrac12 v_{n+1/2}^TDv_{n+1/2}
+\tfrac12 (u^{n+1})^TKu^n.
\]

| Stiffness contrast | dt (s) | Energy range / initial | Bound dt / dense critical dt |
|---:|---:|---:|---:|
| 1:1 | 7.50650247e-05 | 2.856e-15 | 0.87654069 |
| 30:1 | 1.37049358e-05 | 3.990e-15 | 0.85061282 |

## Independent assembly and isotropic limit

The test reference integrates each physical triangle with an independently
constructed engineering-strain B matrix: `K_e=area B.T H B`. It checks consistent
mass/stiffness, symmetry, nonnegative eigenvalues to roundoff, positive lumped
mass, all four mesh diagonal patterns, and mixed-layer DG0 assignment. MPI tests
also check every local and ghost cell against its physical z coordinate.

For isotropy, C11=C33=lambda+2mu, C13=lambda, C55=mu. The identical model is
constructed through isotropic and explicit VTI paths, both homogeneous and with
one/two equal-property layers. Comparisons include M, K, lumped mass, the stable
dt bound, largest-eigenvalue diagnostic, initial K action and elastic energy,
and source-driven displacement/velocity histories.

| Representation | Maximum relative difference over all compared quantities |
|---|---:|
| Homogeneous | 0.000e+00 |
| One layer | 0.000e+00 |
| Two layers | 0.000e+00 |

Across eight independent assembly cases, maximum relative stiffness error is 1.623e-16.

A separate before/after capture of the old homogeneous absorber fixture found
M, K, C, lumped mass/damping, dt, displacement, and velocity bitwise unchanged.
That fixture is the existing `tests.absorbing2d.test_operators.config` with
left/right/bottom absorption, dt=0.01, duration=0.3, source (0.17,2.73), direction
(1,-2), Ricker f0=5, amplitude=2, shift=0.08, and receiver (0.41,3.23).
Bitwise equality is an observation in the locked environment, not a portability
guarantee. No existing numerical acceptance threshold or test was changed.

## Independent Christoffel reference

`tests/vti2d/reference.py` imports only NumPy. For unit phase direction n,

\[
\Gamma(n)=\begin{pmatrix}
C_{11}n_x^2+C_{55}n_z^2 &(C_{13}+C_{55})n_xn_z\\
(C_{13}+C_{55})n_xn_z&C_{55}n_x^2+C_{33}n_z^2
\end{pmatrix},\quad \Gamma d=\rho v^2d.
\]

qP denotes the faster eigenbranch and qSV the slower one. Eigenvector sign is
arbitrary; polarization comparisons are sign-invariant. The chosen material
has separated eigenvalues. No unique polarization is claimed at a degeneracy.
Tests cover 0, +/-15, +/-35, +/-60, +/-90 degrees from +z toward +x, direct axial
speeds/polarizations, and 25 isotropic-limit directions. NumPy eigenvalues are
checked against the independent closed-form quadratic roots.

For arbitrary k, let D(k)=|k|^2 Gamma(k/|k|), with eigenvalue rho omega^2.
Differentiating this eigenproblem for a normalized d gives the group velocity

\[
g_i=\partial_{k_i}\omega
=\frac{d^T(\partial_{k_i}D)d}{2\rho\omega}.
\]

The reference implements these derivatives directly and checks them against
centered finite differences. A phase angle is a wavevector angle; a packet
centroid angle is not a phase angle.

Maximum normalized Christoffel eigenpair residual is 4.439e-16; the closed-form speed comparisons pass a 1e-15 relative limit. Analytical group velocities agree with finite differences within the tested 2e-9 relative / 1e-7 m/s absolute tolerances.

| Direction | qP (m/s) | qSV (m/s) |
|---|---:|---:|
| 0 degrees | 3162.277660 | 1673.320053 |
| 15 degrees | 3150.828268 | 1809.476448 |
| 35 degrees | 3225.506773 | 2090.465409 |
| 60 degrees | 3682.036445 | 1934.582026 |
| 90 degrees | 4000.000000 | 1673.320053 |

## Homogeneous finite-packet experiments

The validation material is rho=2500 kg/m³, (C11,C33,C13,C55)=(40,25,9,7) GPa.
The free-boundary square is [-2400,2400]² m. At t=0 a packet is centered at (0,0)
with spectral amplitude proportional to `exp(-400² |k-k0|²/2)`. Every Fourier
component uses its own analytical eigenpolarization. Initial velocity multiplies
that component by `-i omega(k)`; it is not set to zero or approximated by a
constant translation speed. The real field and velocity initialize the normal
production `start()` path. No public source or receiver API is modified.

The carrier is `k0=(2*pi/4800)*(ix,iz)` for indices (0,16), (16,0), (8,12), and
(-8,12), giving phase directions 0, 90, and +/-33.6900675 degrees. Physical
duration is 0.2 s. Axial cases use h=20 m; both oblique signs use h=40,30,20 m.
dt is 0.2 divided by the smallest integer step count that places it at or below
half the assembled stability bound. Materials, packet spectrum, physical domain,
and duration remain fixed during refinement.

These packets have finite angular bandwidth: 4.854 degrees RMS axially and
5.389 degrees RMS obliquely. Frequency RMS widths relative to the carrier are
8.43–9.58%; central frequencies range from 5.578 to 13.333 Hz. Therefore the
whole-field and centroid comparisons use the full continuum spectrum, not a
single ray approximation. Exact bandwidth values are in the measurements JSON.

Diagnostics are independent of the FEM stiffness assembly:

1. Fit the unwrapped phase of the carrier's displacement Fourier coefficient
   versus time to measure omega/|k0|. Project onto the analytical polarization
   and report orthogonal-coefficient leakage as a polarization error proxy.
2. Find the spatial spectral peak over the outgoing hemisphere, without a
   narrow predicted-angle search. This verifies the selected carrier bin, not
   sub-bin angular accuracy or arbitrary-angle interface refraction.
3. Compare the final entire displacement field to the independent continuum
   spectrum evolved by `exp(-i omega(k)t)`.
4. Measure motion of the squared-displacement centroid and compare to the
   spectrum-weighted group velocity `sum(|A|² g)/sum(|A|²)`. This is a packet
   location diagnostic, not an energy or flux measurement. The continuum
   centroid itself agrees with that formula within 6e-8 relative in these cases.

| Mode | Phase angle | h (m) | dt (s) | Phase-speed error | Polarization leakage | Field error | Group-velocity error |
|---|---:|---:|---:|---:|---:|---:|---:|
| qP | 0.00000° | 20 | 0.00181818 | 0.006794 | 0.000000 | 0.094405 | 0.020411 |
| qP | 90.00000° | 20 | 0.00181818 | 0.006373 | 0.000000 | 0.112789 | 0.019442 |
| qP | 33.69007° | 40 | 0.00363636 | 0.030364 | 0.009339 | 0.384327 | 0.088070 |
| qP | 33.69007° | 30 | 0.00273973 | 0.017212 | 0.005045 | 0.217310 | 0.049567 |
| qP | 33.69007° | 20 | 0.00181818 | 0.007696 | 0.002227 | 0.096875 | 0.022110 |
| qP | -33.69007° | 40 | 0.00363636 | 0.011924 | 0.015991 | 0.161635 | 0.040765 |
| qP | -33.69007° | 30 | 0.00273973 | 0.006709 | 0.008965 | 0.090636 | 0.022859 |
| qP | -33.69007° | 20 | 0.00181818 | 0.002989 | 0.003975 | 0.040278 | 0.010158 |
| qSV | 0.00000° | 20 | 0.00181818 | 0.007366 | 0.000000 | 0.048654 | 0.021415 |
| qSV | 90.00000° | 20 | 0.00181818 | 0.007366 | 0.000000 | 0.049628 | 0.021299 |
| qSV | 33.69007° | 40 | 0.00363636 | 0.032050 | 0.007699 | 0.256763 | 0.113012 |
| qSV | 33.69007° | 30 | 0.00273973 | 0.018548 | 0.004225 | 0.150568 | 0.067559 |
| qSV | 33.69007° | 20 | 0.00181818 | 0.008415 | 0.001840 | 0.068879 | 0.031332 |
| qSV | -33.69007° | 40 | 0.00363636 | 0.010065 | 0.012760 | 0.087857 | 0.036103 |
| qSV | -33.69007° | 30 | 0.00273973 | 0.005658 | 0.007182 | 0.049367 | 0.020245 |
| qSV | -33.69007° | 20 | 0.00181818 | 0.002517 | 0.003193 | 0.021945 | 0.008980 |

Errors above are fractions, not percentages. Every outgoing spectral peak occurs at the prescribed carrier bin.

| Mode at +33.69007°, h=20 m | Continuum phase speed | Measured phase speed | Central group angle | Measured packet angle |
|---|---:|---:|---:|---:|
| qP | 3211.349146 m/s | 3186.634470 m/s | 43.988586° | 44.181923° |
| qSV | 2081.679733 m/s | 2099.197111 m/s | 45.364169° | 45.401704° |

The group-error column uses the spectrum-integrated reference; the central-ray group angles illustrate why phase and packet direction differ.

Refinement reduces phase, polarization, full-field, and group-motion errors for
both modes and both signs. The triangle orientation produces different errors
for +/-horizontal directions; this numerical asymmetry decreases with h.
The tables include time-discretization error because dt scales with h. No
convergence order is fitted or claimed.

For an isolation control, move every free boundary from 2400 to 3600 m while
keeping h=20 m, physical packet, carrier, and time fixed. The maximum changes
among phase error, polarization leakage, group error, and relative field error
are below 4.4e-7, much smaller than mesh errors. This bounds boundary
contamination in the stated measurement window.

## Layered scope, MPI, and public example

Mixed isotropic/VTI assembly, DG0 assignment, interface alignment rejection,
contrasted-material stability, and source/receiver operation are tested.
Layered MPI runs exercise wave motion on both sides of the contact. A formal
anisotropic interface-slowness/refraction or signed-amplitude experiment is
**deferred**. Layer support must not be described as validated anisotropic
Zoeppritz scattering.

MPI comparisons use 1, 2, and 4 ranks for homogeneous and mixed-layer VTI. Global
operators/actions and material fields are ordered by physical coordinates;
tests do not assume local cell numbering. They compare M, K, mass, stable dt,
K action, and complete displacement/velocity receiver histories. Acceptance is
3e-12 times the peak serial magnitude of each quantity.

The MPI operator/trace model is [-1,1]² m with 4x4 cells. The mixed case uses
the isotropic lower material (rho,Vp,Vs)=(2200,3000,1700) and the VTI upper
material listed above, meeting at z=0. Trace runs use dt=1e-5 s, duration=0.002 s,
source (0.13,-0.37), direction (1,2), f0=2000 Hz, amplitude=1e6 N/m,
time shift=0.0003 s, and receivers (0.21,-0.17), (-0.21,0.37). These deliberately
small runs test partition equivalence; propagation accuracy is established by
the separately resolved packet experiments.

| Material | Ranks | Relative K error | Relative mass error | Relative dt error | Relative K-action error | Relative displacement error | Relative velocity error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Homogeneous | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 6.915e-16 | 7.895e-15 | 4.038e-15 |
| Homogeneous | 4 | 0.000e+00 | 1.819e-16 | 0.000e+00 | 1.383e-15 | 1.134e-14 | 4.701e-15 |
| Mixed layers | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.076e-15 | 3.355e-15 | 2.602e-15 |
| Mixed layers | 4 | 0.000e+00 | 1.819e-16 | 0.000e+00 | 1.537e-15 | 1.071e-14 | 3.633e-15 |

Consistent mass matrices and layered material values also agree exactly in these runs.

Run the tracked public example:

```bash
python examples/2d/vti.py
mpiexec -n 4 python examples/2d/vti.py
```

It compares VTI to isotropy with identical vertical P/S speeds, using an 80x80
mesh on [-1600,1600]² m, dt=0.001 s, duration=0.35 s, and a central (1,1) force
with f0=12 Hz, amplitude=1e8 N/m, time shift=0.06 s. Receivers are at (600,0),
(0,600), and (420,420). Boundaries are free; the selected duration precedes
returns. Horizontal peak time changes from 0.274 to 0.228 s, while the vertical
peak changes from 0.274 to 0.275 s. These are mixed-mode waveform peaks, not
phase-speed estimates; the formal phase validation is the packet experiment.

## Reproduction and regression

Measured environment: Python 3.14.6, DOLFINx 0.11.0, NumPy 2.5.3, PETSc 3.25.5
with real float64 scalars. Numerical runs limit OpenMP and OpenBLAS to one
thread per process.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SEISFEM_VTI_REPORT_DIR=/tmp/vti-report python -m pytest -ra tests/vti2d
python -m tests.vti2d.report /tmp/vti-report \
  docs/validation/2d_vti_measurements.json
```

Tests always recompute their simulations and references. Committed measurements
are audit evidence, never cached acceptance oracles. See the
[unrounded measurements](2d_vti_measurements.json) and
[commands/results log](2d_vti_checks.txt).

- Focused VTI suite: **59 passed in 27.38s**.
- Existing 2D core/material/absorber/oblique suites: **187 passed in 1531.64s (0:25:31)**.
- Complete repository regression: **385 passed in 1827.39s (0:30:27)**.
- Ruff, formatting, pre-commit, and whitespace checks passed.

Full-run measurements reproduce the focused-run values exactly. The complete suite includes every previous regression and MPI subprocess test.

## Explicit limitations

This establishes **2D plane-strain VTI with vertical symmetry axis**, positive
in-plane strain energy, and the supported horizontal mesh-aligned layers.
It does not establish TTI, arbitrary anisotropy, anisotropic absorbing boundaries,
PML, attenuation, viscoelasticity, poroelasticity, 3D SH physics, C66/gamma
sensitivity, or non-horizontal heterogeneous interfaces. Thomsen convenience
input is not implemented. Critical/evanescent branches and quantitative
anisotropic interface directions/amplitudes remain outside the validation claim.
