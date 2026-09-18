# P-to-qP/qSV scattering at a welded isotropic/VTI interface

This validation starts from main/v0.7.0,
`43aca95f199542429792f936284dd466055fa4eb`. **No production solver change is
required.** The existing conforming plane-strain weak form and DG0 layered VTI
materials supply the interface physics.

The formal claim is restricted to **incident isotropic P scattering into four
propagating branches at one horizontal, planar, mesh-aligned welded
isotropic/VTI interface in 2D vertical-axis VTI plane strain**. Coordinates are
(x,z), with positive-up z and the interface at z=0. This is not general
anisotropic Zoeppritz validation. VTI absorbing boundaries remain unsupported
and continue to be rejected by production configuration validation.

## Media and cases

The lower medium is isotropic: rho=2200 kg/m³, Vp=3000 m/s, Vs=1700 m/s.
Its canonical stiffnesses are C11=C33=19.8 GPa, C13=7.084 GPa, C55=6.358 GPa.
The upper medium is VTI: rho=2500 kg/m³,
(C11,C33,C13,C55)=(38,34.225,8,12.1) GPa. Its vertical qP/qSV speeds are
3700/2200 m/s; the different horizontal and oblique behavior follows from
Christoffel, not an isotropic speed substitution.

P incidence is from below at 0, 15, 25, 35, and -25 degrees from +z toward +x.
A 701-point root scan over [-35,35] degrees verifies four real propagating
roots in each medium. The reference rejects nonpropagating/degenerate cases;
55-degree P incidence is an explicit rejection test. The chosen contrast gives
measurable converted coefficients at every nonzero acceptance angle.

## Independent slowness reference and signs

`tests/vti_interface2d/reference.py` uses NumPy and the standard library only.
It imports no seisfem assembly or existing isotropic scattering solver. The
old independent isotropic reference is imported by tests solely for comparison.

Use

\[
u=A d\exp[i\omega(px+qz-t)],\qquad \|d\|=1.
\]

p=kx/omega is conserved by the horizontal interface. For prescribed p, solve

\[
\det\begin{pmatrix}
C_{11}p^2+C_{55}q^2-\rho &(C_{13}+C_{55})pq\\
(C_{13}+C_{55})pq&C_{55}p^2+C_{33}q^2-\rho
\end{pmatrix}=0.
\]

The determinant is an even quartic in q. Writing y=q² gives

\[
C_{55}C_{33}y^2+
\left[C_{33}(C_{11}p^2-\rho)+C_{55}(C_{55}p^2-\rho)
-(C_{13}+C_{55})^2p^2\right]y
+(C_{11}p^2-\rho)(C_{55}p^2-\rho)=0.
\]

The implementation nondimensionalizes with c0=sqrt(C33/rho), solves the
quadratic with a cancellation-resistant formula, and examines both signs of
both real q roots. A symmetric Christoffel eigensolve identifies which
eigenvalue equals rho and hence whether the candidate is the faster qP or
slower qSV eigenbranch. Polarizations are deterministic: with
n=(p,q)/sqrt(p²+q²), choose qP so d.n>0 and qSV so d.(nz,-nx)>0. This recovers
the existing isotropic `d_P=n`, `d_S=(nz,-nx)` convention exactly. Coefficients
retain their signs and complex phase.

For D(k)d=rho omega²d, differentiating the eigenproblem gives

\[
g_i=\partial\omega/\partial k_i
=d^T(\partial D/\partial k_i)d/(2\rho\omega).
\]

Each candidate's group velocity is calculated before selection. The incident
and transmitted roots require gz>0; reflected roots require gz<0. **Root
selection uses vertical group velocity, not the sign of q.** This restricted
reference requires exactly one selected root per eigenbranch; it does not
silently resolve more complicated root multiplicities. Phase and group angles
are reported separately, using atan2(horizontal component, absolute vertical
component), i.e. deviations from the outgoing normal. Signed q and gz remove
any ambiguity about the actual propagation half-space.

## Traction, continuity and mechanical flux

The displacement gradient is `i omega A d outer (p,q)`. Therefore
epsilon_xx/(i omega A)=p dx, epsilon_zz/(i omega A)=q dz, and
2epsilon_xz/(i omega A)=q dx+p dz. On the **+z-oriented plane**, stripping
the common factor i omega A gives

\[
\tau_x=C_{55}(q d_x+p d_z),\qquad
\tau_z=C_{13}p d_x+C_{33}q d_z.
\]

Let s=(dx,dz,tau_x,tau_z). Welded displacement and traction continuity gives

\[
[s_{RP},s_{RSV},-s_{TqP},-s_{TqSV}]
\begin{pmatrix}R_P\\R_{SV}\\T_{qP}\\T_{qSV}\end{pmatrix}=-s_{inc}.
\]

Traction rows are divided by sqrt(rho_lower C33_lower) before the complex solve.
No raw dimensional matrix condition number is used. A separate residual check
reconstructs displacement gradients, strain tensors and stress tensors rather
than reusing the traction-column calculation.

Particle velocity has amplitude -i omega A d; traction has amplitude
i omega A tau. Consequently

\[
P_z=-\langle (\sigma e_z)\cdot v\rangle
=\tfrac12\omega^2|A|^2\operatorname{Re}(\tau\cdot d^*).
\]

The propagating modes here have real polarizations. Their signed unit-amplitude
flux divided by omega² is `0.5*tau.dot(d)`. As an independent check, this equals
rho*gz/2, where gz comes from the Christoffel derivative. Reflected fractions
are minus their downward flux divided by the upward incident flux; transmitted
fractions retain their upward sign. **The VTI flux is not calculated using an
isotropic rho*c*cos(theta) formula, and |A|² is not called energy.**

<!-- ANALYTICAL -->

## Reference self-validation and isotropic limit

Tests check determinant and eigenvector residuals for all q candidates,
independent finite-difference group derivatives, displacement/traction
continuity, flux closure, and outgoing group signs. Under p -> -p, qP
coefficients are even and qSV coefficients are odd with this polarization
convention; flux fractions are even.

Specializing the upper material to the existing isotropic reference's
rho=2500, Vp=4000, Vs=2300 reproduces that independently formulated reference
for P incidence at 0/15/25/35 degrees and SV incidence at 0/5/15/20 degrees.
The comparison includes signed amplitudes, fluxes, q, directions and
polarizations; no sign correction or magnitude-only comparison is needed.

<!-- REFERENCE_ERRORS -->

The FEM isotropic-limit test reproduces the old 25-degree P experiment exactly:
its physical packet, 7200 m extent, 2.25 s measurement time and h=20 m. Only the
upper material representation changes to VTI stiffnesses. Small interface
matrices are also compared directly. The old diagnostic and its existing signed
amplitude, imaginary-part and flux tolerances are retained unchanged.

<!-- FEM_ISOTROPIC -->

## Incident packet and continuum estimator audit

The production experiment uses a free-boundary square [-8400,8400]² m and
measurement time 2.8 s. Initial center is
`(-3600*tan(theta), -3600)` m, placing the nominal interface intersection at x=0.
The scalar potential is

\[
\Phi=\exp[-s^2/(2\sigma_s^2)-r^2/(2\sigma_r^2)]\cos(k_0s)/k_0,
\quad u=\nabla\Phi,\quad v=-c_{inc}\partial_s u,
\]

with f0=5 Hz, sigma_s=360 m, sigma_r=1500 m, and k0=2*pi*f0/3000. Thus the
continuum incident displacement is irrotational, with consistent translating
initial velocity. The test-only initial fields enter the normal production
`PlaneStrainOperators.start()` path. No point-force source or new public API is
used. The finite packet has approximately 2.384-degree incident RMS angular
bandwidth; it is not literally a single plane wave.

The independent NumPy continuum calculation Fourier-decomposes these initial
u/v data into positive-frequency forward/backward waves. It scatters the forward
spectrum using the analytical coefficients. The very small backward component
is not mistaken for an interface reflection. Nonpropagating Gaussian tails are
excluded from that restricted reference and quantified: at most 1.24e-6 of the
incident spectral energy over the chosen cases. This is an explicit truncation,
not validation of those critical/evanescent tails.

At fixed kx, conservation of omega implies

\[
J=\left|\frac{\partial k_{z,out}}{\partial k_{z,in}}\right|
=\left|\frac{g_{z,in}}{g_{z,out}}\right|,
\quad U_{out}(k_{out},t)=Z(p)U_{in}(k_{in},0)e^{-i\omega t}/J.
\]

This Jacobian changes spatial Fourier density; omitting it biases amplitudes.
The formula is checked by independently differentiating the slowness-root map.
An additional kinetic-plus-strain energy integral of the synthesized continuum
field audits its normalization without using J or the scattering flux factors.

For amplitude recovery, spatially isolate each half-space and transform
`(u+i*v/omega0)/2` at the central frequency and conserved kx. Project onto the
branch's analytical anisotropic polarization, divide by the incident transform
measured with the same convention, multiply by J, and remove propagation phase.
All four outputs remain complex. This transfer estimate is evaluated at the
specified input frequency/slowness; it is not fitted to maximize agreement.

The independent phase-direction diagnostic windows away from the interface,
computes a padded 2D displacement spectrum, projects each wavenumber onto the
appropriate Christoffel eigenpolarization, and selects a connected outgoing
lobe in a broad 0.5–1.5 f0 band. It receives no predicted Snell-angle search
window. It reports a power-weighted lobe angle and angular spread. No isotropic
divergence/curl mode decomposition is used for VTI. The central-ray angle and
finite-band lobe centroid need not coincide; convergence is checked against
the independently synthesized continuum packet as well as the central root.

<!-- CONTINUUM_AUDIT -->

At normal incidence the central p=0 converted coefficients vanish. The finite
beam still contains oblique sidebands that can convert; their lobe angles and
integrated energy are not evidence of spurious central conversion. Phase
residual is undefined for a zero target coefficient and is stored as null.

## FEM refinement, phase and flux

The +/-25-degree cases use h=40,30,20 m; 0/15/35 degrees use h=20 m. Each h
places z=0 exactly on a mesh row. Physical packet, materials, domain and time
are fixed across refinement. dt is chosen from the existing assembled bound,
at or below 0.8 times that bound, with an integer number of steps to 2.8 s.
Initial discrete curl/div contamination is measured and decreases with h.

Tables report real and imaginary coefficients, magnitudes, phase residuals,
signed-real errors, complex errors, normalized branch fluxes and closure.
Closure here refers to **central spectral transfer flux fractions**, not a
claim of total FEM energy conservation. Spectrum-integrated continuum fractions
are also retained in the measurements to quantify finite-bandwidth effects.

<!-- FEM_RESULTS -->

Individual signed-real errors can be nonmonotone through cancellation: a coarse
coefficient's real part may accidentally agree while its imaginary part is large.
The complex error and its refinement must be considered together. No convergence
order is fitted, and the remaining complex phase error is not hidden by replacing
coefficients with magnitudes.

<!-- ANGLES_SYMMETRY -->

Group/energy directions in this report are analytical Christoffel predictions.
No FEM packet-centroid group-angle measurement is claimed in this interface
suite. The homogeneous VTI milestone separately validated group propagation.

## Outer boundaries and MPI

The 8400 m half-width accommodates the broad interface footprint and separated
outgoing packets. At 25 and 35 degrees, a control moves every free boundary to
9600 m, retaining h=20 m, physical initialization and measurement time. The
larger grid is cropped to the original Fourier window for comparison. This
tests the main acceptance angle and the largest positive angle rather than
assuming boundary isolation from a plot. No VTI absorber is used.

MPI validation uses the 25-degree, h=20 m case on 1, 2, and 4 ranks, with all
four scattered branches present. Every local and ghost cell's material is
checked by physical position. Comparisons include mass and stiffness-action
invariants, stable dt, sampled displacement/velocity histories, and all extracted
complex coefficients, fluxes and spectral angles. Field gathering uses integer
physical grid coordinates, never assumed local cell numbering.

<!-- BOUNDARIES_MPI -->

## Reproduction and regression

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SEISFEM_VTI_INTERFACE_REPORT_DIR=/tmp/vi-report \
  python -m pytest -ra tests/vti_interface2d
python -m tests.vti_interface2d.report /tmp/vi-report \
  docs/validation/2d_vti_interface_measurements.json
```

The committed [measurements](2d_vti_interface_measurements.json) preserve unrounded
values; the [check log](2d_vti_interface_checks.txt) records commands and results.
Tests always recompute simulations and references; committed evidence is never
used as an acceptance oracle. Full trace arrays are compared during MPI tests;
the compact committed file stores their peak scales and comparison errors.

<!-- REGRESSION -->

## Limits

This establishes only the stated below-critical propagating central branches
for **isotropic P incidence from below** at a single horizontal, planar,
mesh-aligned welded isotropic/VTI interface in 2D vertical-axis VTI plane strain.
The SV-incidence isotropic-limit reference checks do not constitute anisotropic
SV-incidence FEM validation. VTI-to-isotropic incidence is not validated here.

Critical/evanescent branches, head waves, interface waves, TTI, arbitrary
anisotropy, tilted/arbitrary interfaces, anisotropic absorbers, attenuation,
poroelasticity and 3D TI/SH physics remain excluded. The reference's restricted
root multiplicity is not a general anisotropic radiation-condition solver.
No production API, constitutive law, interface term, or time integrator changed.
