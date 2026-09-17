# Oblique scattering at one horizontal elastic interface

This validation extends the evidence for the **unchanged v0.5.0 production
kernel**, baseline `76e657088370890cfef024554846da82033e2ae7`. It does not alter the
heterogeneous-materials milestone, production solver, public API, or boundaries.
All new numerical machinery lives in `tests/oblique2d/`.

The supported statement is restricted to **isotropic, horizontal, planar,
mesh-aligned interfaces and below-critical propagating P/SV branches**. Both
incident modes reproduce signed conversion and correctly normalized central
spectral flux, with improving complex-amplitude and finite-beam angle agreement
under refinement. This is a quantitative convergence result, not a claim of
negligible phase error at the finest mesh.

## Equations and independent reference

Positive-up coordinates are `(x,z)`. In each material,

\[
\rho\ddot u=\nabla\cdot\sigma,\qquad
\sigma=\lambda(\nabla\cdot u)I+\mu(\nabla u+\nabla u^T).
\]

The welded interface at `z=0` enforces continuity of `u_x`, `u_z`, `sigma_xz`,
and `sigma_zz`. The lower material has `(rho,Vp,Vs)=(2200,3000,1700)` in SI units;
the upper material has `(2500,4000,2300)`. Incidence is upward: P at 25 degrees
or SV at 15 degrees, measured from the interface normal.

The NumPy-only [reference](../../tests/oblique2d/reference.py) imports no seisfem,
FEM, or packet implementation. For `u=A d exp[i omega(p x+q z-t)]`, it uses

\[
n=(pc,\ \pm\sqrt{1-p^2c^2}),\quad d_P=n,\quad d_S=(n_z,-n_x),\quad q=n_z/c.
\]

**SV signs refer specifically to this rotated-gradient polarization.** Reflected
polarizations reverse at normal incidence, so a positive reflected scalar
coefficient need not mean positive reflected vertical displacement.

After removing the common `i omega`, the traction column is

\[
t=\begin{pmatrix}
\mu(qd_x+pd_z)\\
\lambda(pd_x+qd_z)+2\mu qd_z
\end{pmatrix}.
\]

With `b=(d_x,d_z,t_x,t_z)`, solve
`[b_RP b_RS -b_TP -b_TS] A = -b_inc` in that fixed branch order. Traction rows
are divided by `rho_lower Vp_lower` before solving. No dimensional matrix
condition number is interpreted. A separate residual implementation rebuilds
full gradient and stress tensors; displacement and scaled traction residuals
are below `6e-16`. Analytic outgoing flux closes to one within `2e-15`.
Zero contrast, both normal-incidence impedance limits, and negative horizontal
slowness are also tested.

For each propagating displacement coefficient,

\[
F_j=\frac{\rho_j c_j|n_{z,j}|}{\rho_i c_i n_{z,i}}|A_j|^2.
\]

The coefficient squared alone is **not** an energy fraction.

## Packets, domain, and fixed refinement protocol

The [test-only packet](../../tests/oblique2d/packets.py) is generated from
`Phi=exp[-s²/(2 sigma_s²)-q²/(2 sigma_q²)] cos(k0 s)/k0`, where `s` and `q`
are longitudinal and transverse coordinates relative to the incident ray.
P uses `grad Phi`; SV uses `(Phi_z,-Phi_x)`. Both prescribe the consistent
translating velocity `v=-c partial_s u`. These are pure continuum modes;
translation differs slightly from dispersive spreading of a finite angular
spectrum. The independent continuum calculation explicitly decomposes these
initial data into positive/negative temporal frequencies.

| Parameter | P incidence | SV incidence |
|---|---:|---:|
| Carrier frequency | 5 Hz | 5 Hz |
| Incident wavelength | 600 m | 340 m |
| Longitudinal sigma | 360 m | 204 m |
| Transverse sigma | 1500 m | 1500 m |
| Center `(x,z)` | `(-1400,-3000)` m | `(-1400,-3000)` m |
| Final measurement time | 2.25 s | 3.10 s |
| Energy-weighted angular standard deviation | 2.38364 degrees | 1.34967 degrees |
| Energy-weighted frequency mean / standard deviation | 5.64419 / 0.88595 Hz | 5.64157 / 0.88633 Hz |

Measured incident div/curl spectral spreads at h=40,30,20 m are
`2.42322, 2.41914, 2.41736 degrees` (P) and
`1.38258, 1.37592, 1.37169 degrees` (SV). These displacement-spectrum lobe
measurements differ slightly from the energy-weighted bandwidths above because
of the frequency weighting and 2% lobe threshold.

The frequency distribution is moderately broad; it is **not** approximated as
a monochromatic Gaussian template. The angular spectrum is substantially
narrower than the early `sigma_q=500 m` exploratory packet. The amplitude
measurement extracts the specified central 5 Hz spectral component explicitly.

The fixed square domain is `[-7200,7200]² m`, with traction-free outer edges.
Meshes have `h=40,30,20 m`, respectively 130321, 231361, and 519841 vector nodes.
All have an exact horizontal row at `z=0`. Physical packet parameters, domain,
materials, and measurement times are held fixed. The existing production
`PlaneStrainOperators.start()` and central differences advance nodal initial
data with row-sum lumped mass and the unchanged DG0 material fields. There is
no public point force, absorber, modified operator, or special interface load.

`dt=T/ceil[T/(0.8 stable_dt)]` uses the existing assembled bound. The three
bounds are `0.0065011516734`, `0.0048758637551`, and `0.0032505758367 s`.
Actual P steps are 433, 577, 866; SV steps are 597, 795, 1193.

The domain accommodates several transverse beam widths and the outgoing packets
before significant boundary returns. Enlarging **every** outer boundary by
1200 m while preserving `h=40 m`, initial data, final time, and the original
measurement window changes amplitudes by at most `2.93e-4`, imaginary components
by `1.16e-4`, flux fractions by `1.20e-4`, and angles by `0.00693 degrees`.
Gaussian tails have no compact support: this is a measured contamination bound,
not an assertion that boundary influence is mathematically zero.

## Diagnostics and independent estimator audit

Initial purity uses exact cellwise P1 derivatives, area weighted over cells
whose mean squared nodal displacement exceeds `1e-4` of the global maximum.
Report `RMS(curl_y u)/RMS(div u)` for P and its inverse for SV. Owned cells are
counted once; ghosts are synchronized. This measures interpolation contamination,
not the analytically zero continuum contamination.

Angles use centered div/curl proxies on the gathered physical nodal grid.
A half-space cosine window is zero within 120 m of the interface and reaches
one at 480 m. A twice-padded 2D FFT selects the outgoing quadrant and the broad
2.5–7.5 Hz band. A connected lobe above 2% of the observed maximum gives the
power-weighted angle and angular standard deviation, after dividing proxy power
by `|k|²`. **The estimator receives no predicted Snell angle.** Tests at unrelated
9 and 38 degree directions exercise this independence.

Finite beams have conversion-dependent spectral weighting. In particular,
SV→P's continuum centroid is about one degree from the central Snell ray.
Consequently, the tables show both central Snell errors and differences from
an independently synthesized continuum beam processed with the identical
measurement window. Individual Snell errors need not decrease monotonically;
the norm of errors relative to the continuum beam does decrease. No convergence
order is fitted.

Signed amplitudes use complex spatial Fourier integrals of
`(u+i v/omega0)/2`, projected onto the branch polarization. Half-space separation,
outgoing wavevector, and distinct P/S wavenumber radii separate the branches.
The integral is evaluated at the specified central frequency and conserved
horizontal wavenumber, rather than moved to a peak that maximizes agreement.
This diagnostic uses the known central spectral coordinates; the angle
measurement above is independent of them.

At fixed `kx`, the change of variables between incident and outgoing vertical
wavenumbers is

\[
J=\left|\frac{\partial k_{z,j}}{\partial k_{z,i}}\right|
=\frac{c_i n_{z,i}}{c_j|n_{z,j}|},\qquad
\widehat A_j(k_j,T)=Z_j\widehat A_i(k_i,0)e^{-i\omega_0T}/J.
\]

Thus the estimator is `Zhat=J Uout/Uin exp(i omega0 T)`, with the same Fourier
quadrature and temporal-frequency projection for incident and outgoing fields.
It uses no fitted Gaussian template and no fitted reference amplitude.
The **real part** is the signed coefficient; the imaginary part is retained
as a phase residual. `signed_error=|Re Zhat-Z|` and
`complex_error=|Zhat-Z|` are different quantities. The machine-readable field
`amplitude_error` means the latter. Flux uses `|Zhat|²` with the impedance/angle
factor, not just the square of the real part.

The separate [continuum beam calculation](../../tests/oblique2d/continuum.py)
propagates the exact Gaussian potential spectrum through `Z(p)`, including the
coordinate Jacobian and exact continuum dispersion. It imports no seisfem or
FEM code. Applying the numerical estimator to these fields recovers central
signed coefficients within `3e-4` and flux closure within `6e-4`. As a separate
Jacobian check, integrating physical kinetic plus elastic strain energy using
spectral derivatives gives final/initial energy ratios `0.999999155` (P) and
`0.999999698` (SV).

Integrating plane-wave flux fractions against the incident spectrum's
`omega² |U_inc(k)|²` energy weights changes each fraction by less than `4e-4`
from its central-angle value. The incident energy outside the all-propagating
reference domain is `6.83e-9` (P) and `2.16e-7` (SV); those tails are excluded
from the analytic beam synthesis, not treated as critical-angle validation.

## Numerical results

The tables below and [complete measurements](2d_oblique_measurements.json)
record the reference, every refinement, bandwidth, MPI diagnostics, receiver
samples, operator invariants, and boundary sensitivity. Angles and spreads are
in degrees. Amplitudes are signed displacement ratios and fluxes are normalized
normal elastic flux fractions of the central spectral component.

<!-- NUMERICAL_TABLES -->

### Refinement summary

| Incident | h (m) | P1 impurity | max beam-angle error (deg) | max signed error | complex-error norm | flux sum | closure error |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 40 | 0.122397 | 0.28266 | 0.075356 | 0.393046 | 1.01614881 | 0.01614881 |
| P | 30 | 0.091618 | 0.15971 | 0.022575 | 0.222086 | 1.00941883 | 0.00941883 |
| P | 20 | 0.060993 | 0.07155 | 0.003573 | 0.098739 | 1.00447807 | 0.00447807 |
| SV | 40 | 0.130692 | 0.26626 | 0.034296 | 0.198362 | 1.05629956 | 0.05629956 |
| SV | 30 | 0.097706 | 0.14664 | 0.008988 | 0.099813 | 1.03266736 | 0.03266736 |
| SV | 20 | 0.064991 | 0.06267 | 0.005074 | 0.040882 | 1.01488930 | 0.01488930 |

### P incidence: 25 degrees

RP/RS denote reflected P/SV; TP/TS denote transmitted P/SV.

| Branch | Snell angle | Signed reference | Reference flux | Spectrum-integrated flux |
|---|---:|---:|---:|---:|
| RP | 25.000000 | +0.15429789 | 0.02380784 | 0.02393912 |
| RS | 13.856069 | +0.16018046 | 0.01557563 | 0.01538595 |
| TP | 34.297570 | +0.82612979 | 0.94258649 | 0.94246594 |
| TS | 18.905445 | -0.14080438 | 0.01803004 | 0.01820900 |

Angular standard deviations describe the selected lobe, not an uncertainty in its mean.

| h (m) | Branch | Measured angle | Spread | Snell error | Continuum-beam error |
|---:|---|---:|---:|---:|---:|
| 40 | RP | 24.88515 | 2.46979 | 0.11485 | 0.11811 |
| 40 | RS | 13.87671 | 1.26249 | 0.02064 | 0.10377 |
| 40 | TP | 34.47575 | 3.55256 | 0.17818 | 0.15740 |
| 40 | TS | 19.55792 | 1.82978 | 0.65247 | 0.28266 |
| 30 | RP | 24.83551 | 2.44245 | 0.16449 | 0.06847 |
| 30 | RS | 13.92109 | 1.25610 | 0.06502 | 0.05938 |
| 30 | TP | 34.40817 | 3.54510 | 0.11060 | 0.08982 |
| 30 | TS | 19.43497 | 1.79914 | 0.52953 | 0.15971 |
| 20 | RP | 24.79811 | 2.42968 | 0.20189 | 0.03108 |
| 20 | RS | 13.95377 | 1.25092 | 0.09771 | 0.02670 |
| 20 | TP | 34.35984 | 3.53868 | 0.06227 | 0.04149 |
| 20 | TS | 19.34681 | 1.77440 | 0.44137 | 0.07155 |

| h (m) | Branch | Signed Re A | Im A | Signed error | Complex error | Flux | Flux error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 40 | RP | +0.144704 | +0.068244 | 0.009594 | 0.068915 | 0.02559655 | 0.00178872 |
| 40 | RS | +0.111394 | +0.115960 | 0.048787 | 0.125805 | 0.01569553 | 0.00011989 |
| 40 | TP | +0.750774 | +0.356778 | 0.075356 | 0.364650 | 0.95427335 | 0.01168686 |
| 40 | TS | -0.147438 | -0.029924 | 0.006634 | 0.030650 | 0.02058338 | 0.00255334 |
| 30 | RP | +0.152665 | +0.038540 | 0.001633 | 0.038574 | 0.02479193 | 0.00098409 |
| 30 | RS | +0.144303 | +0.070699 | 0.015877 | 0.072460 | 0.01567520 | 0.00009957 |
| 30 | TP | +0.803555 | +0.204449 | 0.022575 | 0.205692 | 0.94950511 | 0.00691862 |
| 30 | TS | -0.145360 | -0.015941 | 0.004555 | 0.016579 | 0.01944659 | 0.00141655 |
| 20 | RP | +0.154751 | +0.017047 | 0.000454 | 0.017053 | 0.02423860 | 0.00043077 |
| 20 | RS | +0.157139 | +0.032398 | 0.003041 | 0.032541 | 0.01562703 | 0.00005139 |
| 20 | TP | +0.822557 | +0.091301 | 0.003573 | 0.091371 | 0.94596400 | 0.00337751 |
| 20 | TS | -0.143038 | -0.006781 | 0.002234 | 0.007140 | 0.01864844 | 0.00061840 |

### SV incidence: 15 degrees

RP/RS denote reflected P/SV; TP/TS denote transmitted P/SV.

| Branch | Snell angle | Signed reference | Reference flux | Spectrum-integrated flux |
|---|---:|---:|---:|---:|
| RP | 27.176912 | -0.10156690 | 0.01676594 | 0.01648033 |
| RS | 15.000000 | +0.11467077 | 0.01314939 | 0.01326114 |
| TP | 37.516219 | +0.10207646 | 0.02287747 | 0.02325302 |
| TS | 20.497526 | +0.79707455 | 0.94720720 | 0.94700552 |

Angular standard deviations describe the selected lobe, not an uncertainty in its mean.

| h (m) | Branch | Measured angle | Spread | Snell error | Continuum-beam error |
|---:|---|---:|---:|---:|---:|
| 40 | RP | 27.43238 | 2.49318 | 0.25547 | 0.04664 |
| 40 | RS | 14.45821 | 1.27336 | 0.54179 | 0.08686 |
| 40 | TP | 38.49208 | 3.80406 | 0.97586 | 0.02858 |
| 40 | TS | 20.76353 | 1.93806 | 0.26600 | 0.26626 |
| 30 | RP | 27.40792 | 2.51557 | 0.23101 | 0.02218 |
| 30 | RS | 14.49872 | 1.29071 | 0.50128 | 0.04635 |
| 30 | TP | 38.48064 | 3.85792 | 0.96442 | 0.04002 |
| 30 | TS | 20.64390 | 1.92497 | 0.14637 | 0.14664 |
| 20 | RP | 27.39328 | 2.53881 | 0.21637 | 0.00754 |
| 20 | RS | 14.52593 | 1.30469 | 0.47407 | 0.01914 |
| 20 | TP | 38.49948 | 3.93152 | 0.98326 | 0.02118 |
| 20 | TS | 20.55994 | 1.91624 | 0.06241 | 0.06267 |

| h (m) | Branch | Signed Re A | Im A | Signed error | Complex error | Flux | Flux error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 40 | RP | -0.092774 | -0.051613 | 0.008793 | 0.052357 | 0.01831819 | 0.00155225 |
| 40 | RS | +0.080375 | +0.096510 | 0.034296 | 0.102422 | 0.01577429 | 0.00262491 |
| 40 | TP | +0.093880 | +0.054867 | 0.008197 | 0.055476 | 0.02596062 | 0.00308314 |
| 40 | TS | +0.803256 | +0.151658 | 0.006181 | 0.151784 | 0.99624645 | 0.04903926 |
| 30 | RP | -0.100123 | -0.028429 | 0.001444 | 0.028466 | 0.01760603 | 0.00084008 |
| 30 | RS | +0.106417 | +0.057094 | 0.008253 | 0.057688 | 0.01458444 | 0.00143505 |
| 30 | TP | +0.101383 | +0.030511 | 0.000693 | 0.030519 | 0.02461181 | 0.00173434 |
| 30 | TS | +0.806063 | +0.069371 | 0.008988 | 0.069951 | 0.97586508 | 0.02865788 |
| 20 | RP | -0.101917 | -0.012317 | 0.000350 | 0.012322 | 0.01712823 | 0.00036229 |
| 20 | RS | +0.114562 | +0.025465 | 0.000109 | 0.025466 | 0.01377287 | 0.00062349 |
| 20 | TP | +0.103017 | +0.013196 | 0.000940 | 0.013229 | 0.02368314 | 0.00080567 |
| 20 | TS | +0.802149 | +0.025889 | 0.005074 | 0.026381 | 0.96030505 | 0.01309785 |

<!-- END_NUMERICAL_TABLES -->

## MPI and regression gates

MPI compares serial (`COMM_SELF`), two ranks, and four ranks for **both h=20 m
cases**. It checks geometry-based DG0 layer/rho/lambda/mu assignment, global
lumped mass and its squared norm, initial stiffness quadratic form and action
norm, stability bound, actual dt, purity, receiver samples, all branch angles,
complex amplitudes, and fluxes. Global arrays are reconstructed from integer
physical grid coordinates; no cell-number or partition-order assumption is used.
All numerical diagnostics must agree within `rtol=3e-10, atol=3e-12`.
The assembled stability bound and actual dt agree exactly. Relative differences
in mass, purity, and operator invariants are at most `6.15e-15`.

<!-- MPI_RESULTS -->

| Incident | MPI ranks | max angle difference (deg) | max signed-amplitude difference | max imaginary difference | max flux difference | max receiver difference |
|---|---:|---:|---:|---:|---:|---:|
| P | 2 | 3.553e-15 | 3.886e-16 | 2.012e-16 | 1.214e-16 | 2.130e-15 |
| P | 4 | 7.105e-15 | 1.443e-15 | 8.160e-15 | 1.221e-15 | 4.829e-15 |
| SV | 2 | 1.421e-14 | 1.110e-15 | 2.220e-16 | 2.776e-15 | 1.516e-15 |
| SV | 4 | 7.105e-15 | 7.494e-16 | 1.226e-14 | 8.882e-16 | 8.715e-15 |

<!-- END_MPI_RESULTS -->

Reproduce in the repository's locked DOLFINx 0.11 / real-double PETSc environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SEISFEM_OBLIQUE_REPORT_DIR=/tmp/oblique-report \
  python -m pytest -ra tests/oblique2d
python -m tests.oblique2d.report /tmp/oblique-report \
  docs/validation/2d_oblique_measurements.json
```

The report collector only packages freshly written test outputs. Tests always
run the FEM and independent continuum calculations; they never pass by reading
committed measurements. Refinement simulations are shared by session fixtures,
so each physical case/resolution runs once per pytest invocation.

Final full regression: **293 passed in 1395.73 s**, including all 31 new oblique
tests and all 262 previous tests; no skips or expected failures. The separate
heterogeneous run passed all 25 tests. Ruff, formatter, pre-commit, and whitespace
checks pass on the clean review checkout. See the
[check transcript](2d_oblique_checks.txt). No existing tracked file changed.

## Limits of the claim

- The real signed amplitudes at h=20 m differ by at most 0.00508; the largest
  complex error is 0.0914, dominated by accumulated numerical propagation phase.
  The phase residual is not discarded or called converged to roundoff.
- Outgoing central-component flux sums approach one but retain 0.45% (P) and
  1.49% (SV) excess at the finest resolution. This is a correctly normalized
  scattering diagnostic, **not** a measurement of total discrete FEM energy
  conservation. The continuum physical-energy audit is a separate check.
- Finite-beam centroid shifts, spectral thresholds, Gaussian tails, finite box
  quadrature, and boundaries leave an estimator floor of a few `1e-4` in amplitude.
  Raw centroid-to-Snell angle errors can plateau; comparison to the independent
  continuum beam distinguishes this from FEM error.
- Only this isotropic, horizontal, planar, mesh-aligned interface and the stated
  below-critical P/SV cases are established. Critical/evanescent branches, head
  waves, anisotropy, attenuation, poroelasticity, non-planar interfaces, arbitrary
  geology, faults, PML, heterogeneous absorbers, and laminate science are excluded.

No production defect was exposed, and no production code was changed. The next
step is human review of this branch and its quantitative evidence; any push,
merge, or release decision remains with the repository owner.
