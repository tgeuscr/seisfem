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

| Incident ° | p (µs/m) | RP | RSV | TqP | TqSV | \|ΣF−1\| |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.167192 | 0 | 0.832808 | 0 | 2.22045e-16 |
| 15 | 86.273 | 0.144344 | 0.108358 | 0.83808 | -0.064226 | 1.11022e-16 |
| 25 | 140.873 | 0.110248 | 0.151766 | 0.855456 | -0.127674 | 4.44089e-16 |
| 35 | 191.192 | 0.0805139 | 0.149765 | 0.908483 | -0.214835 | 4.44089e-16 |
| -25 | -140.873 | 0.110248 | -0.151766 | 0.855456 | 0.127674 | 4.44089e-16 |

All analytical imaginary parts are zero for these lossless propagating cases.

| Incident ° | Branch | q (µs/m) | gz (m/s) | Phase ° | Group ° | Flux fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | RP | -333.333 | -3000 | 0 | 0 | 0.0279533 |
| 0 | RS | -588.235 | -1700 | 0 | 0 | 0 |
| 0 | TP | 270.27 | 3700 | 0 | 0 | 0.972047 |
| 0 | TS | 454.545 | 2200 | 0 | 0 | 0 |
| 15 | RP | -321.975 | -2897.78 | 15 | 15 | 0.0208351 |
| 15 | RS | -581.874 | -1681.62 | 8.43366 | 8.43366 | 0.00681367 |
| 15 | TP | 257.484 | 3517.65 | 18.524 | 17.255 | 0.968895 |
| 15 | TS | 441.152 | 2136.74 | 11.0653 | 17.2877 | 0.00345641 |
| 25 | RP | -302.103 | -2718.92 | 25 | 25 | 0.0121547 |
| 25 | RS | -571.118 | -1650.53 | 13.8561 | 13.8561 | 0.0139821 |
| 25 | TP | 233.099 | 3138.65 | 31.1465 | 31.2574 | 0.959974 |
| 25 | TS | 418.74 | 2038.75 | 18.594 | 26.9928 | 0.0138896 |
| 35 | RP | -273.051 | -2457.46 | 35 | 35 | 0.00648248 |
| 35 | RS | -556.297 | -1607.7 | 18.9672 | 18.9672 | 0.0146737 |
| 35 | TP | 191.163 | 2457.1 | 45.0044 | 48.4628 | 0.937751 |
| 35 | TS | 388.656 | 1925.4 | 26.194 | 34.3602 | 0.0410926 |
| -25 | RP | -302.103 | -2718.92 | -25 | -25 | 0.0121547 |
| -25 | RS | -571.118 | -1650.53 | -13.8561 | -13.8561 | 0.0139821 |
| -25 | TP | 233.099 | 3138.65 | -31.1465 | -31.2574 | 0.959974 |
| -25 | TS | 418.74 | 2038.75 | -18.594 | -26.9928 | 0.0138896 |

RP/RSV carry energy downward; TqP/TqSV carry energy upward. The JSON also records both
polarization components and the full group-velocity vector for every branch.

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

Determinants are divided by rho², eigenvector residuals by rho, and traction
continuity by the incident impedance. These reported residuals are dimensionless.

| Reference check | Maximum absolute residual/difference |
| --- | --- |
| determinant_residual | 6.65209e-16 |
| eigen_residual | 4.2613e-16 |
| continuity_residual | 2.91038e-16 |
| flux_closure | 4.44089e-16 |

Isotropic-specialization differences (dimensionless except q in s/m):

| Quantity | Maximum difference |
| --- | --- |
| amplitude | 2.498e-16 |
| flux | 5.55112e-16 |
| q | 1.0842e-19 |
| direction | 1.11022e-16 |
| polarization | 1.38778e-16 |
| group_direction | 4.44089e-16 |

Finite-difference audits: relative group-vector norm error 6.14093e-11; relative
Fourier-Jacobian error 1.39038e-10.

The FEM isotropic-limit test reproduces the old 25-degree P experiment exactly:
its physical packet, 7200 m extent, 2.25 s measurement time and h=20 m. Only the
upper material representation changes to VTI stiffnesses. Small interface
matrices are also compared directly. The old diagnostic and its existing signed
amplitude, imaginary-part and flux tolerances are retained unchanged.

| Isotropic FEM comparison | Relative max error |
| --- | --- |
| small matrices: M | 0 |
| small matrices: K | 1.43342e-16 |
| small matrices: mass | 0 |
| small matrices: dt | 0 |
| full experiment: dt | 0 |
| full experiment: stable_dt | 0 |
| full experiment: steps | 0 |
| full experiment: purity | 0 |
| full experiment: mass | 0 |
| full experiment: mass_square | 0 |
| full experiment: stiffness_form | 0 |
| full experiment: action_square | 0 |
| full experiment: trace | 0 |
| full experiment: final_u_v | 0 |

The old isotropic diagnostic gives closure error 0.00447807. Its original acceptance limits pass
unchanged: signed error <0.006, |imaginary part| <0.10, branch flux error <0.014, closure error
<0.016. The final-field comparison includes displacement and velocity.

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

| Incident ° | Branch | Re Â | Im Â | \|Â\| | Phase residual (rad) | Real error | Complex error | Flux |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | RP | 0.167192 | -5.14676e-10 | 0.167192 | -3.07834e-09 | 2.4406e-11 | 5.15254e-10 | 0.0279533 |
| 0 | RS | -2.08218e-19 | 1.67449e-19 | 2.67196e-19 | undefined | 2.08218e-19 | 2.67196e-19 | 4.04565e-38 |
| 0 | TP | 0.832808 | -1.20362e-08 | 0.832808 | -1.44525e-08 | 2.00441e-08 | 2.33802e-08 | 0.972047 |
| 0 | TS | -5.48334e-18 | -1.61909e-18 | 5.71738e-18 | undefined | 5.48334e-18 | 5.71738e-18 | 2.72404e-35 |
| 15 | RP | 0.144341 | 2.24665e-06 | 0.144341 | 1.55648e-05 | 2.49371e-06 | 3.35648e-06 | 0.0208344 |
| 15 | RS | 0.108356 | 4.06192e-07 | 0.108356 | 3.74866e-06 | 1.22285e-06 | 1.28855e-06 | 0.00681352 |
| 15 | TP | 0.838097 | -2.55121e-06 | 0.838097 | -3.04406e-06 | 1.68774e-05 | 1.70691e-05 | 0.968934 |
| 15 | TS | -0.0642255 | -1.2178e-06 | 0.0642255 | 1.89613e-05 | 4.66059e-07 | 1.30393e-06 | 0.00345636 |
| 25 | RP | 0.110244 | 5.70231e-06 | 0.110244 | 5.17245e-05 | 4.49488e-06 | 7.26087e-06 | 0.0121537 |
| 25 | RS | 0.151773 | -1.61483e-06 | 0.151773 | -1.06398e-05 | 7.22183e-06 | 7.40017e-06 | 0.0139834 |
| 25 | TP | 0.855488 | 2.40894e-05 | 0.855488 | 2.81586e-05 | 3.21121e-05 | 4.01433e-05 | 0.960046 |
| 25 | TS | -0.127669 | -5.25977e-06 | 0.127669 | 4.11985e-05 | 5.02341e-06 | 7.27322e-06 | 0.0138885 |
| 35 | RP | 0.0805182 | -6.1836e-06 | 0.0805182 | -7.67976e-05 | 4.29532e-06 | 7.52906e-06 | 0.00648318 |
| 35 | RS | 0.149723 | -1.97879e-05 | 0.149723 | -0.000132163 | 4.16906e-05 | 4.61483e-05 | 0.0146655 |
| 35 | TP | 0.908555 | -0.000167732 | 0.908555 | -0.000184614 | 7.24864e-05 | 0.000182724 | 0.937901 |
| 35 | TS | -0.214806 | 3.31923e-05 | 0.214806 | -0.000154523 | 2.9798e-05 | 4.46055e-05 | 0.0410812 |
| -25 | RP | 0.110244 | 5.70231e-06 | 0.110244 | 5.17245e-05 | 4.49488e-06 | 7.26087e-06 | 0.0121537 |
| -25 | RS | -0.151773 | 1.61483e-06 | 0.151773 | -1.06398e-05 | 7.22183e-06 | 7.40017e-06 | 0.0139834 |
| -25 | TP | 0.855488 | 2.40894e-05 | 0.855488 | 2.81586e-05 | 3.21121e-05 | 4.01433e-05 | 0.960046 |
| -25 | TS | 0.127669 | 5.25977e-06 | 0.127669 | 4.11985e-05 | 5.02341e-06 | 7.27322e-06 | 0.0138885 |

| Incident ° | RMS angular width ° | Excluded spectral energy | Central closure error | Integrated RP / RSV / TqP / TqSV flux |
| --- | --- | --- | --- | --- |
| 0 | 2.38364 | 1.08965e-08 | 4.67824e-08 | 0.0277525 / 0.000197146 / 0.971981 / 6.94953e-05 |
| 15 | 2.38364 | 1.09789e-08 | 3.81001e-05 | 0.0207625 / 0.00685422 / 0.968731 / 0.00365235 |
| 25 | 2.38364 | 1.41736e-08 | 7.13199e-05 | 0.012241 / 0.0137984 / 0.959603 / 0.0143575 |
| 35 | 2.38357 | 1.23846e-06 | 0.000130807 | 0.00677929 / 0.0142688 / 0.936995 / 0.0419567 |
| -25 | 2.38364 | 1.41736e-08 | 7.13199e-05 | 0.012241 / 0.0137984 / 0.959603 / 0.0143575 |

Independent physical packet-energy relative errors: 0°: 6.35773e-07, 25°: 6.44077e-07, 35°:
1.89106e-06. The finite-difference Jacobian check passes a 3e-9 relative tolerance. The
continuum transfer error is much smaller than FEM dispersion error; the largest finite-band
integrated-versus-central branch flux difference is 0.000864125.

At normal incidence the central p=0 converted coefficients vanish. The finite
beam still contains oblique sidebands that can convert; their lobe angles and
integrated energy are not evidence of spurious central conversion. Phase
residual is undefined for a zero target coefficient and is stored as null.
The reference also checks the vertical impedance reduction explicitly:
Z=sqrt(rho C33), RP=(Z_upper-Z_lower)/(Z_upper+Z_lower), and
TqP=2 Z_lower/(Z_upper+Z_lower), with the reflected polarization pointing down.

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

| Incident ° | h m | Branch | Re Â | Im Â | \|Â\| | Phase residual rad | Real error | Complex error | Flux | Flux error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 40 | RP | 0.0990364 | 0.0579199 | 0.11473 | 0.529194 | 0.0112119 | 0.0589951 | 0.0131629 | 0.00100824 |
| 25 | 40 | RS | 0.0822651 | 0.128869 | 0.152888 | 1.00265 | 0.0695005 | 0.146416 | 0.0141897 | 0.00020763 |
| 25 | 40 | TP | 0.732283 | 0.45139 | 0.860227 | 0.552402 | 0.123173 | 0.467894 | 0.970712 | 0.0107379 |
| 25 | 40 | TS | -0.127677 | -0.0473457 | 0.136173 | 0.355104 | 3.20214e-06 | 0.0473457 | 0.0158003 | 0.00191075 |
| 25 | 30 | RP | 0.107827 | 0.0329873 | 0.11276 | 0.296887 | 0.00242124 | 0.0330761 | 0.0127148 | 0.00056015 |
| 25 | 30 | RS | 0.12859 | 0.0818729 | 0.152442 | 0.566966 | 0.0231752 | 0.0850897 | 0.0141071 | 0.000124972 |
| 25 | 30 | TP | 0.817186 | 0.262175 | 0.858213 | 0.310453 | 0.0382697 | 0.264953 | 0.966171 | 0.00619718 |
| 25 | 30 | TS | -0.12976 | -0.0258842 | 0.132316 | 0.196893 | 0.00208587 | 0.0259681 | 0.014918 | 0.00102844 |
| 25 | 20 | RP | 0.1104 | 0.0146276 | 0.111365 | 0.131729 | 0.000151876 | 0.0146283 | 0.0124021 | 0.000247476 |
| 25 | 20 | RS | 0.147232 | 0.0380361 | 0.152066 | 0.252814 | 0.00453366 | 0.0383054 | 0.0140375 | 5.53609e-05 |
| 25 | 20 | TP | 0.848591 | 0.117755 | 0.856722 | 0.137885 | 0.00686519 | 0.117955 | 0.962817 | 0.0028436 |
| 25 | 20 | TS | -0.1292 | -0.0112105 | 0.129686 | 0.0865517 | 0.00152625 | 0.0113139 | 0.0143307 | 0.000441151 |
| -25 | 40 | RP | 0.0970538 | 0.058185 | 0.113159 | 0.540061 | 0.0131944 | 0.0596623 | 0.0128049 | 0.000650262 |
| -25 | 40 | RS | -0.12921 | -0.0797131 | 0.15182 | 0.552773 | 0.0225556 | 0.0828429 | 0.0139922 | 1.00792e-05 |
| -25 | 40 | TP | 0.828745 | 0.232766 | 0.860813 | 0.273811 | 0.026711 | 0.234293 | 0.972033 | 0.0120595 |
| -25 | 40 | TS | 0.116335 | 0.0623296 | 0.13198 | 0.49186 | 0.0113394 | 0.0633527 | 0.0148423 | 0.000952704 |
| -25 | 30 | RP | 0.106732 | 0.0335468 | 0.11188 | 0.304533 | 0.00351659 | 0.0337307 | 0.012517 | 0.000362362 |
| -25 | 30 | RS | -0.14479 | -0.0458748 | 0.151883 | 0.306831 | 0.00697593 | 0.0464021 | 0.0140038 | 2.16985e-05 |
| -25 | 30 | TP | 0.848426 | 0.131181 | 0.858507 | 0.153402 | 0.00703055 | 0.131369 | 0.966833 | 0.00685958 |
| -25 | 30 | TS | 0.125105 | 0.0355811 | 0.130067 | 0.277092 | 0.00256866 | 0.0356737 | 0.0144151 | 0.000525489 |
| -25 | 20 | RP | 0.109952 | 0.0150001 | 0.110971 | 0.135586 | 0.000295723 | 0.015003 | 0.0123146 | 0.000159883 |
| -25 | 20 | RS | -0.150464 | -0.0204402 | 0.151846 | 0.135021 | 0.0013018 | 0.0204816 | 0.0139969 | 1.47863e-05 |
| -25 | 20 | TP | 0.854864 | 0.0582029 | 0.856843 | 0.0679796 | 0.000592645 | 0.058206 | 0.963088 | 0.00311415 |
| -25 | 20 | TS | 0.12775 | 0.0158327 | 0.128727 | 0.123307 | 7.5696e-05 | 0.0158329 | 0.0141197 | 0.000230072 |

Other accepted cases at h=20 m:

| Incident ° | Branch | Re Â | Im Â | \|Â\| | Phase residual rad | Real error | Complex error | Flux | Flux error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | RP | 0.166629 | 0.0222855 | 0.168113 | 0.132954 | 0.000563521 | 0.0222926 | 0.0282618 | 0.000308526 |
| 0 | RS | 1.1573e-09 | 8.57057e-10 | 1.4401e-09 | undefined | 1.1573e-09 | 1.4401e-09 | 1.1752e-18 | 1.1752e-18 |
| 0 | TP | 0.829947 | 0.0846057 | 0.834249 | 0.10159 | 0.0028601 | 0.084654 | 0.975414 | 0.00336711 |
| 0 | TS | 2.05737e-10 | 4.94647e-11 | 2.116e-10 | undefined | 2.05737e-10 | 2.116e-10 | 3.73121e-20 | 3.73121e-20 |
| 15 | RP | 0.14411 | 0.0191519 | 0.145377 | 0.132124 | 0.000233917 | 0.0191534 | 0.0211344 | 0.000299323 |
| 15 | RS | 0.103676 | 0.0307184 | 0.108131 | 0.28805 | 0.00468122 | 0.031073 | 0.00678526 | 2.84131e-05 |
| 15 | TP | 0.834237 | 0.0936631 | 0.839478 | 0.111806 | 0.00384345 | 0.093742 | 0.97213 | 0.00323525 |
| 15 | TS | -0.0644406 | -0.0104509 | 0.0652825 | 0.16078 | 0.000214598 | 0.0104531 | 0.00357107 | 0.000114656 |
| 35 | RP | 0.0806999 | 0.0111714 | 0.0814695 | 0.137558 | 0.000186018 | 0.011173 | 0.00663727 | 0.00015479 |
| 35 | RS | 0.14683 | 0.0344189 | 0.150811 | 0.230255 | 0.00293446 | 0.0345438 | 0.0148793 | 0.000205629 |
| 35 | TP | 0.897347 | 0.146654 | 0.909252 | 0.161999 | 0.0111356 | 0.147076 | 0.93934 | 0.00158889 |
| 35 | TS | -0.21792 | -0.00365071 | 0.217951 | 0.016751 | 0.00308466 | 0.00477942 | 0.042293 | 0.00120037 |

Refinement summaries use the Euclidean norm across four branch errors:

| Incident ° | h m | Steps | dt s | RMS curl/div | Real error norm | Complex error norm | Flux error norm | ΣF | \|ΣF−1\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 1009 | 0.00277502 | 0.00470858 | 0.00291508 | 0.08754 | 0.00338121 | 1.00368 | 0.00367564 |
| 15 | 20 | 1009 | 0.00277502 | 0.0370884 | 0.0060652 | 0.10114 | 0.00325121 | 1.00362 | 0.00362081 |
| 25 | 40 | 505 | 0.00554455 | 0.122397 | 0.141872 | 0.496069 | 0.0109551 | 1.01386 | 0.0138645 |
| 25 | 30 | 673 | 0.00416048 | 0.0916179 | 0.0448539 | 0.281441 | 0.0063081 | 1.00791 | 0.00791074 |
| 25 | 20 | 1009 | 0.00277502 | 0.0609932 | 0.00836884 | 0.12539 | 0.00288877 | 1.00359 | 0.00358759 |
| 35 | 20 | 1009 | 0.00277502 | 0.0782053 | 0.0119232 | 0.151566 | 0.00200791 | 1.00315 | 0.00314968 |
| -25 | 40 | 505 | 0.00554455 | 0.0440383 | 0.0390501 | 0.263305 | 0.0121145 | 1.01367 | 0.0136725 |
| -25 | 30 | 673 | 0.00416048 | 0.03298 | 0.0108193 | 0.147721 | 0.00688924 | 1.00777 | 0.00776913 |
| -25 | 20 | 1009 | 0.00277502 | 0.021964 | 0.00146256 | 0.0654461 | 0.00312676 | 1.00352 | 0.00351889 |

At normal incidence the central converted amplitudes are 1.4401e-09 (RSV) and 2.116e-10 (TqSV).
These are separate from the physical finite-beam sidebands. The largest h=20 m phase residual
across nonzero branches is 0.28805 rad (16.50°, reflected SV at 15° incidence). This remaining
coherent phase error limits the precision of the validation.

Individual signed-real errors can be nonmonotone through cancellation: a coarse
coefficient's real part may accidentally agree while its imaginary part is large.
The complex error and its refinement must be considered together. No convergence
order is fitted, and the remaining complex phase error is not hidden by replacing
coefficients with magnitudes.

| Incident ° | Branch | Central phase ° | Analytical group ° | Continuum lobe ° | FEM lobe ° | FEM RMS spread ° | FEM−continuum absolute ° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | RP | 0 | 0 | 2.45984e-15 | 2.63733e-05 | 2.36035 | 2.63733e-05 |
| 0 | TP | 0 | 0 | 1.04083e-17 | 4.15875e-06 | 2.97042 | 4.15875e-06 |
| 15 | RP | 15 | 15 | 14.7527 | 14.7604 | 2.38178 | 0.00769508 |
| 15 | RS | 8.43366 | 8.43366 | 8.80867 | 8.78112 | 1.29338 | 0.0275442 |
| 15 | TP | 18.524 | 17.255 | 18.5283 | 18.5417 | 3.00348 | 0.0134261 |
| 15 | TS | 11.0653 | 17.2877 | 11.836 | 11.8329 | 1.79547 | 0.00314708 |
| 25 | RP | 25 | 25 | 24.5879 | 24.619 | 2.40975 | 0.0311308 |
| 25 | RS | 13.8561 | 13.8561 | 13.9717 | 13.9456 | 1.24693 | 0.0260942 |
| 25 | TP | 31.1465 | 31.2574 | 31.1583 | 31.1855 | 3.14295 | 0.0271548 |
| 25 | TS | 18.594 | 26.9928 | 19.1757 | 19.2253 | 1.84679 | 0.0496182 |
| 35 | RP | 35 | 35 | 34.8409 | 34.9202 | 2.57053 | 0.0792648 |
| 35 | RS | 18.9672 | 18.9672 | 18.8397 | 18.8254 | 1.14063 | 0.0143381 |
| 35 | TP | 45.0044 | 48.4628 | 45.0679 | 45.1361 | 3.70429 | 0.0682237 |
| 35 | TS | 26.194 | 34.3602 | 26.6554 | 26.818 | 1.83679 | 0.162602 |
| -25 | RP | -25 | -25 | -24.5879 | -24.5671 | 2.39189 | 0.0207935 |
| -25 | RS | -13.8561 | -13.8561 | -13.9717 | -13.9607 | 1.2518 | 0.0110254 |
| -25 | TP | -31.1465 | -31.2574 | -31.1583 | -31.1821 | 3.14147 | 0.0237169 |
| -25 | TS | -18.594 | -26.9928 | -19.1757 | -19.1546 | 1.83137 | 0.0210919 |

Normal converted sideband lobes are deliberately omitted from this central-direction table. The
sign-symmetry residual is Â(+25)−sÂ(−25), with s=+1 for P and −1 for SV.

| h m | Complex symmetry L2 error | Max branch complex error | Max flux symmetry error | Max angle antisymmetry error ° |
| --- | --- | --- | --- | --- |
| 40 | 0.249156 | 0.238959 | 0.00132156 | 0.288423 |
| 30 | 0.140751 | 0.134668 | 0.000662396 | 0.159013 |
| 20 | 0.0626874 | 0.0598819 | 0.000270547 | 0.0707101 |

The fixed diagonal orientation of the triangular mesh need not be invariant under x reflection.
The measured complex symmetry discrepancy decreases with refinement; no exact finite-grid
symmetry is claimed.

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

| Boundary-control incident ° | Max ΔRe | Max ΔIm | Max Δflux | Max Δangle ° |
| --- | --- | --- | --- | --- |
| 25 | 9.15444e-06 | 1.83018e-05 | 9.65984e-06 | 0.000282534 |
| 35 | 3.35701e-05 | 2.29841e-05 | 5.9334e-06 | 0.000612159 |

MPI differences relative to the serial run (invariants and histories relative to their maximum
serial scale; coefficient components, fluxes, and angles absolute):

| Quantity | 2 ranks | 4 ranks |
| --- | --- | --- |
| mass | 1.84045e-16 | 1.84045e-16 |
| mass_square | 0 | 0 |
| dt | 0 | 0 |
| stable_dt | 0 | 0 |
| stiffness_form | 1.31262e-15 | 2.1877e-16 |
| action_square | 3.96376e-15 | 4.79823e-15 |
| trace | 7.29698e-15 | 1.52721e-14 |
| velocity_trace | 7.43216e-15 | 6.21897e-15 |
| real | 9.99201e-16 | 1.33227e-15 |
| imaginary | 4.08007e-15 | 9.45077e-15 |
| flux | 1.11022e-15 | 3.33067e-16 |
| phase_angle | 3.19744e-14 | 1.42109e-14 |

Material assignment error is exactly zero on all ranks, including DG0 ghosts. All recursive
diagnostics pass rtol=3e-10, atol=3e-12. Full receiver histories, not just their peaks, are
compared during execution.

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

All requested checks passed on the recovered branch, with no production changes
and no changes to pre-existing tests or acceptance thresholds.

| Check | Result | Elapsed time |
| --- | --- | --- |
| Focused VTI interface suite, including 1/2/4-rank comparisons | 37 passed | 1016.20 s |
| Strengthened independent reference checks | 11 passed | 0.10 s |
| Existing VTI, oblique, heterogeneous, plane-strain and absorber suites | 246 passed | 1064.03 s |
| Complete repository regression | 422 passed | 1940.95 s |
| Ruff lint | Passed | — |
| Ruff format check | 147 files already formatted | — |
| Pre-commit, all tracked files | Passed | — |
| Git whitespace check | Passed | — |

The existing-suite selection comprises VTI (59), isotropic oblique (31),
heterogeneous materials (25), core plane strain (65), homogeneous absorbers (33),
and heterogeneous absorbers (33). The full regression includes the final
strengthened isotropic group-direction assertion. The focused and full-run
measurement collections agree exactly across all 1,254 numeric fields; maximum
absolute difference is zero. The committed evidence comes from the focused run.

The recovered commits and commands are recorded in the check log. Final audit
completion was on 2026-09-22. The branch was not merged, tagged, or pushed;
pre-existing untracked exploratory files were preserved.

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
No C66 or Thomsen-gamma sensitivity is established by this in-plane system.
No production API, constitutive law, interface term, or time integrator changed.
