# First-order local elastic absorbing boundaries

This branch extends the validated `v0.3.0` homogeneous 2D experiment pathway
with a local impedance/dashpot boundary. It preserves the existing plane-strain
operators, line-force/receiver semantics, and `CentralDifference` implementation.
It is exact for normally incident **continuum** P and SV waves in homogeneous
isotropic elasticity. It is not perfectly nonreflecting at arbitrary incidence.

The starting commit is `d96b689` (`main` and peeled `v0.3.0`). Development is on
`feature/2d-absorbing-boundaries`; no release tag or merge is part of this work.
The pre-implementation regression passed **204 tests in 566.32 s**: 98 original
1D, 65 existing 2D-core, and 41 source/receiver tests. None was removed or weakened.

## Public configuration and architecture

Add this entry to an existing `SimulationConfig2D` mapping:

```python
boundaries = {
    "left": "absorbing",
    "right": "absorbing",
    "lower": "absorbing",
    "upper": "free",
}
```

Then use the same `SimulationConfig2D.model_validate(model)` and
`Simulation2D(cfg).run()` workflow. The complete executable example is
[`examples/2d/absorbing.py`](../../examples/2d/absorbing.py).
Coordinates are `(x,z)` in metres, with positive-up z: `upper` is the top/free
surface and `lower` is the bottom. All four natural conditions default to `free`.
No damping is integrated on a free side. Its corner nodes can still receive the
endpoint weights of adjacent absorbing facets.

Essential conditions retain their existing syntax, for example
`constraints=[{"side": "upper", "components": ["x"]}]`. A side cannot be both
absorbing and constrained, including a componentwise constraint: that combination
is rejected early. Constraints and absorbing conditions on **different** sides
can meet at corners; constrained components stay zero. A `free` natural setting
does not override an essential constraint: unconstrained components have zero
traction there. Unknown side names such as `top` and unsupported values such as
`pml` are errors, rather than aliases or silent fallbacks.

Production changes are small:

* `config2d.py`: `BoundaryConditions2D` and a same-side conflict validator.
* `boundaries2d.py`: static consistent boundary matrix and row-sum diagonal.
* `fem2d.py`: operator ownership/cleanup and passing damping into `start()`.
* `simulation2d.py`: forwarding the boundary configuration to the existing core.

The audited 1D impedance endpoint already contributes a nonnegative diagonal
viscous term to `M u_tt + C u_t + K u = f`. The common integrator already implements
the required centered damped recurrence, including damped startup. Consequently
**no time integrator or 1D code was changed**. There is no second simulation API.
The consistent boundary matrix is retained for inspection; stepping uses only
its lumped diagonal, assembled once. There is no per-step boundary search, form
assembly, sparse solve, or full-field gather.

The source remains `F(t) d_hat delta_2(x-xs)`, with normalized direction and
Ricker amplitude in **N/m** (line force per out-of-plane length). It generally
excites both P and SV. Source injection and receiver interpolation remain FE
basis evaluation at arbitrary coordinates. Results on **every MPI rank** retain
user receiver order and shapes `(time, receiver, 2)`, components `("x", "z")`.
Displacement is in m; velocity is `(u_next-u_previous)/(2 dt)` in m/s at the same
integer physical time as displacement, including the final sample. `to_xarray()`
and metadata retain their previous conventions. No acceleration receivers or
snapshots were added.

## Continuum condition and FE operator

The bulk equation remains

\[
\rho u_{tt}-\nabla\cdot\sigma(u)=f,\quad
\sigma=\lambda\operatorname{tr}(\epsilon)I+2\mu\epsilon,
\quad \mu=\rho V_s^2,\quad\lambda=\rho(V_p^2-2V_s^2).
\]

These are the ordinary 3D solid Lamé moduli under plane strain, not plane-stress
substitutions. On the selected exterior facets, with outward unit normal n and
either unit tangent t,

\[
\sigma n=-Bv,\qquad
B=\rho V_p(n\otimes n)+\rho V_s(t\otimes t)
 =\rho[V_s I+(V_p-V_s)n\otimes n].
\]

The sign of t has no effect. B has units kg/(m² s); Bv is traction in Pa. A normally
outgoing plane P wave has traction `-rho Vp v`, and a normally outgoing SV wave
has traction `-rho Vs v`, giving exact continuum matching. This is the classical
local viscous boundary construction; see
[Lysmer and Kuhlemeyer (1969)](https://ascelibrary.org/doi/abs/10.1061/JMCEA3.0001144).

Integration by parts puts the positive bilinear form on the left-hand side:

\[
c(w,v)=\int_{\Gamma_a} w\cdot Bv\,ds,\qquad
D u_{tt}+C_L u_t+K u=f.
\]

First assemble the consistent matrix `C_ij = c(phi_i,phi_j)` using UFL's outward
facet normal. It is symmetric positive semidefinite. Then integrate its row sum
directly using the trial field `(1,1)`, equivalent to `C @ ones` before essential
projection. On the supported axis-aligned rectangle, B is diagonal:

| Side | x impedance | z impedance |
| --- | ---: | ---: |
| left/right | rho Vp | rho Vs |
| lower/upper | rho Vs | rho Vp |

For an edge of length L and either component impedance Z, the consistent edge
matrix is `Z L/6 [[2,1],[1,2]]`; its lumped endpoint weights are `Z L/2` each.
Nonnegative P1 traces form a partition of unity, so every damping diagonal entry
is nonnegative. C and C_L have units kg/(m s); `v.T @ C_L @ v` is power per
out-of-plane length, W/m. This row-sum positivity argument is specific to these
axis-aligned sides. It is not a general prescription for rotated/curved meshes,
where tensor off-diagonal terms can invalidate scalar row-sum lumping.

At a corner, contributions from each incident absorbing edge simply add. There
is no corner multiplier. At an essential/absorbing intersection, the diagonal is
retained before projection, but the constrained velocity is zero and performs
no damping work.

Operator tests compare the entire matrix to an independent edge formula, on all
four sides. Maximum total normal/tangential weight errors are both at most
**1.78e-15** in the small operator probe (rho=2.3, Vp=3.2, Vs=1.8, side lengths
2 or 3). The minimum diagonal entry is **0**. Matrix symmetry, PSD to roundoff,
nonnegative random-field dissipation, pure-normal/pure-tangential power, and
linear scaling with density, appropriate wave speed, and boundary length pass.

## Explicit timing, startup, stability, and energy

The unchanged integrator evaluates the force at `t_n`, and computes

\[
(D+\tfrac{dt}{2}C_L)u^{n+1}
=2Du^n-(D-\tfrac{dt}{2}C_L)u^{n-1}+dt^2(f^n-Ku^n).
\]

The denominator is positive and diagonal. For `C_L=0` this is exactly the previous
execution path. Before editing the implementation, free and component-fixed
public-run arrays were saved; afterward displacement, velocity, mass, and the
stability bound were **bitwise identical**. A permanent test also compares
omitted versus explicitly all-free boundaries and checks unchanged stability.

Startup uses

\[
a^0=D^{-1}(f^0-Ku^0-C_Lv^0),\qquad
u^{-1}=u^0-dt\,v^0+\tfrac12dt^2a^0.
\]

The first recurrence gives the corresponding Taylor `u^1` and centered `v^0`.
Tests use nonzero initial displacement, velocity, damping, and force, independently
checking the damping contribution to a0. A manufactured scalar problem uses
`m=1.7`, `k=3.2`,
`q(t)=exp(-0.3t) cos(1.7t)+0.2 sin(0.4t)`, and
`F=m q''+c q'+k q`. At final time 0.8 s with dt=0.04, 0.02, 0.01, 0.005 s:

| c | displacement convergence rates | centered-velocity rates |
| ---: | --- | --- |
| 0.8 | 2.000532, 2.000133, 2.000033 | 2.000373, 2.000093, 2.000023 |
| 8 | 2.000241, 2.000060, 2.000015 | 2.000209, 2.000052, 2.000013 |
| 80 | 2.000193, 2.000048, 2.000012 | 1.999927, 1.999982, 1.999995 |

The zero-damping case also gives order two. These tests isolate temporal error
from spatial FEM error; no claimed continuum acceleration observable is involved.

For a scalar homogeneous recurrence set `alpha=c dt/(2m)` and `beta=k dt²/m`.
The characteristic polynomial is

\[
(1+\alpha)r^2+(\beta-2)r+1-\alpha=0.
\]

The quadratic Jury conditions reduce to `|1-alpha| <= 1+alpha`, `p(1)=beta >= 0`,
and `p(-1)=4-beta >= 0`. Thus nonnegative centered viscous damping introduces no
additional damping CFL; the stiffness limit remains beta<=4. Use a strict safety
margin (the undamped endpoint has a repeated root). Numerical root checks span
alpha=0 through 1000 on both sides of beta=4; a 2000-step scalar run with c=2000,
dt=1, m=1, k=3 remains bounded. Very large unresolved damping can still give
oscillatory or inaccurate transients; stability is not an accuracy certificate.

The matrix conclusion does not require simultaneous diagonalization of C and K.
With `v^n=(u^{n+1}-u^{n-1})/(2dt)`, the actual discrete energy is

\[
E^{n+1/2}=\tfrac12\left\|\frac{u^{n+1}-u^n}{dt}\right\|_D^2
 +\tfrac12(u^{n+1})^TKu^n,
\qquad
E^{n+1/2}-E^{n-1/2}=dt\,(v^n)^Tf^n-dt\,(v^n)^TC_Lv^n.
\]

This follows by multiplying the recurrence by `(u^{n+1}-u^{n-1})/2`.
Writing `u_mid=(u^{n+1}+u^n)/2` gives
`E=0.5 v_half.T (D-dt² K/4) v_half + 0.5 u_mid.T K u_mid`.
It is nonnegative under the existing stiffness bound, and damping removes energy.
Rigid-body displacement modes for fully free systems retain the usual zero
stiffness energy. The existing assembled absolute-row-sum bound on
`D_f^(-1/2) K_ff D_f^(-1/2)` and safety policy remain unchanged; an unsafe dt is
still rejected. SLEPc eigenvalue estimates remain diagnostics rather than the
certificate used for acceptance.

In a 600-step unforced 2D random-state test with mixed fixed/free/absorbing sides,
the maximum step energy/work residual divided by initial energy is **3.27e-16**;
the accumulated balance residual is **1.54e-16**. Remaining energy is
**1.46e-4** of the initial value. All damping work is nonnegative. This is a
discrete mechanical energy balance, not a receiver-amplitude surrogate.

## Normal-incidence P and SV validation

The test-only initial fields are exact continuum longitudinal/transverse plane
packets, not new production source types. Use rho=2400 kg/m³, Vp=3200 m/s,
Vs=1800 m/s in `[0,2400] x [-2000,2000]` m. Define
`Phi=A_u (x-800) exp(-((x-800)/w)²)`, `w=c/(pi f0)`, f0=8 Hz, A_u=1 m.
P displacement is grad(Phi); SV displacement is minus rotated_grad(Phi), where
rotated_grad(Phi)=(Phi_z,-Phi_x). Initial velocity is `-c partial_x u`.
The P field is irrotational and the SV field divergence-free; no transverse
Gaussian is multiplied into a fixed polarization.

The receiver is `(1800,0)`. The right boundary is the free/absorbing comparison;
the left always absorbs. Horizontal-side z constraints for P, or x constraints
for SV, admit the exact plane field. The large transverse extent keeps corner
effects outside the measurement window. The incident packet center is at
1000/c, and the first reflected center at 2200/c: P 0.3125/0.6875 s,
SV 0.555556/1.222222 s. Use windows of half-width `0.8/f0` around those predictions.
The reflection diagnostic is **peak absolute reflected displacement divided by
peak absolute incident displacement** in the corresponding component. It is an
amplitude proxy, not an integrated reflected energy coefficient.

Fixed geometry/bandwidth and dt=0.0005 s give:

| Mode | grid spacing h (m) | incident peak (m) | free reflection proxy | absorbing reflection proxy |
| --- | ---: | ---: | ---: | ---: |
| P | 40 | 0.922445 | 0.845593 | 0.0335260 |
| P | 20 | 0.995554 | 0.974707 | 0.00869585 |
| P | 10 | 1.000278 | 0.998233 | 0.00225093 |
| SV | 40 | 0.638655 | 0.489156 | 0.190549 |
| SV | 20 | 0.867457 | 0.887402 | 0.0243248 |
| SV | 10 | 0.988134 | 0.956102 | 0.00650676 |

Incoming peaks agree between each boundary pair to 1e-12 relative tolerance.
The finest free controls reflect strongly; their absorbing counterparts are
less than 1% of the free controls. Residuals decrease with refinement for both
modes. The coarse SV case is underresolved, as its distorted incident and free
peaks demonstrate. It must not be interpreted as a clean amplitude benchmark.
No formal reflection convergence order is asserted from these three meshes.

Here h is Cartesian spacing; maximum triangle diameter is sqrt(2) h. Taking
`lambda_min=c/(3 f0)` as a conservative resolved-band diagnostic, h/lambda_min
is 0.300, 0.150, 0.075 for P and 0.533, 0.267, 0.133 for SV. A Ricker/Gaussian
packet is not strictly band-limited. The refinement claims concern resolved
observables, with spatial dispersion and boundary lumping both contributing.

### Resolved frequency and oblique characterization

The optional study is separate from routine pytest. Raw results are recorded in
[`2d_absorbing_characterization.json`](2d_absorbing_characterization.json).
Normal runs retain dt=0.0005 s and use
P h=8 m and SV h=5 m at f0=6, 8, 10 Hz, providing at least 12 grid intervals per
wavelength at 3 f0:

| Mode | f0 (Hz) | intervals per wavelength at 3 f0 | absorbing amplitude proxy |
| --- | ---: | ---: | ---: |
| P | 6 | 22.22 | 0.000800183 |
| P | 8 | 16.67 | 0.00142168 |
| P | 10 | 13.33 | 0.00221260 |
| SV | 6 | 20 | 0.000994635 |
| SV | 8 | 15 | 0.00175717 |
| SV | 10 | 12 | 0.00268550 |

Incident peaks range from 0.99746 to 1.00037 m. Reflection remains small over
this resolved band and increases with frequency at fixed h, consistently with
the decreasing spatial resolution. This is not a frequency-independent bound
on performance for unresolved wavefields.

An independent continuum harmonic traction solve also characterizes P incidence
on the right boundary. For unit incident polarization d and slowness p,
`traction/(i omega)=lambda (p.d)n + mu[d(p.n)+p(d.n)]`;
`v/(i omega)=-d`. Tangential slowness is shared by reflected P and SV. A 2x2
system solves their amplitudes from `traction+Bv=0`.

| P incidence angle from normal | reflected P amplitude | converted SV amplitude | reflected energy flux / incident flux |
| ---: | ---: | ---: | ---: |
| 0° | 0 | 0 | 0 |
| 15° | 0.00590401 | 0.00988332 | 0.0000911346 |
| 30° | 0.00988646 | -0.0125567 | 0.000196018 |
| 45° | -0.0243597 | -0.0710014 | 0.00427278 |
| 60° | -0.146449 | -0.129202 | 0.0378481 |
| 75° | -0.423534 | -0.125254 | 0.208006 |

Amplitude signs depend on the specified outgoing unit polarizations. Unlike a
receiver squared-trace proxy, the final column **is** a plane-wave energy flux
ratio: `R_P² + Vs cos(theta_S)/(Vp cos(theta_P)) R_S²`. The traction-free control
gives total reflected flux 1 to roundoff at every angle. These are continuum
calculations, distinct from FE validation; they explain why large-angle
reflection cannot be eliminated by refining a local first-order absorber.

The FE beam study uses an analytically irrotational packet `u0=grad Phi` and
`v0=-Vp grad(partial_s Phi)`, with
`Phi=s exp(-s²/w²-q²/W²)`, `w=Vp/(pi*8)`, transverse width W=400 m.
Here s and q are coordinates along and across the incident direction. A finite
beam has angular spread and diffraction; it is not an exact single-angle plane
wave. The domain is `[-1600,1600] x [-3600,3600]` m; the beam center is 2000 m
before `(1600,0)` along its incident direction. The receiver is 600 m from that
point along the specular P reflection direction. An enlarged right extent of
4000 m supplies an incident-field reference. Other sides absorb in all runs.
Subtract this reference from free and absorbing traces; in 0.70–1.05 s compare
their peak vector displacement norms. The resulting absorbing/free scattered
amplitude proxy is not the continuum energy-flux quantity in the table above.
With dt=0.001 s, measured results are:

| Beam angle | h (m) | free scattered peak | absorbing scattered peak | absorbing/free amplitude proxy |
| ---: | ---: | ---: | ---: | ---: |
| 0° | 20 | 0.580523 | 0.00730782 | 0.0125883 |
| 45° | 20 | 0.590699 | 0.0552105 | 0.0934663 |
| 60° | 20 | 0.536591 | 0.108029 | 0.201325 |
| 60° | 10 | 0.595169 | 0.100282 | 0.168493 |

The finite beam's residual grows with obliquity. At 60°, refinement reduces its
proxy from 20.1% to 16.8%, but a significant residual remains, consistently with
the continuum angular limitation. The finite-width, single-receiver proxy
does not equal a monochromatic plane-wave reflection coefficient; numerical
dispersion, angular spread, diffraction, and the chosen time window also matter.

## Public point-source experiment and physical free surface

The production example uses a homogeneous rectangle `[-1200,1200] x [-1600,0]` m,
h=20 m, dt=0.001 s, duration 1.6 s, and the same material as the packet tests.
Its vector force at `(17.3,-400.7)` has direction `(1,1)`, Ricker f0=5 Hz,
amplitude 1e8 N/m and t0=0.3 s. Receivers are `(17.3,-600.7)`, `(300.4,-450.2)`,
and `(-270.5,-350.8)` in that order.

The stronger regression adds two controls to the example's free/three-absorbing
runs: an enlarged rectangle `[-3600,3600] x [-4000,0]` at the same h with a free
upper side, and a small rectangle with all four sides absorbing. The enlarged
domain's artificial outer returns cannot reach these receivers during recording.
It retains the same physical free surface.

The first receiver's predicted surface-reflected P center is
`0.3+(400.7+600.7)/3200 = 0.6129375 s`. In the early 0.50–0.75 s window:

* Changing small-domain free sides to three absorbers changes the displacement
  norm by only **4.52e-5 relative**.
* The three-absorber run differs from the enlarged reference by **8.08e-7 relative**.
* Absorbing the upper side as a deliberate control changes the first receiver's
  trace by **0.7404 relative**: the physical free-surface signal is substantial.

This window precedes significant unwanted outer returns. The untruncated Ricker
has a small initial tail, so this is a quantified signal-window statement, not
an assertion of an exact zero field outside a compact causal wavefront.

In 0.95–1.60 s, define the artificial-return error as the displacement trace
difference from the enlarged reference. The ratio of its L2 norm for three
absorbers to its L2 norm for all-free boundaries is **0.09114** (90.9% reduction).
Per-receiver ratios are **0.04853, 0.08364, 0.15098**. These are trace-error norms,
not energy coefficients. The lightweight two-run example instead prints the
raw late absorbing/free trace-norm ratio **0.117577**, which includes physical
surface-wave tails and should not be confused with the reference-subtracted
diagnostic.

## Reciprocity and its startup qualification

The mass, stiffness, consistent damping and lumped damping operators are symmetric.
The semidiscrete damped elastic system is reciprocal. Swapping interior FE point
sources and receivers in the three-absorber experiment gives these relative L2
trace differences:

| Pair | displacement | centered velocity |
| --- | ---: | ---: |
| x force / x receiver | 2.52e-14 | 6.33e-14 |
| z force / x receiver versus x force / z receiver | 5.29e-14 | 4.69e-14 |

The test uses off-node A=(0.313,0.487), B=(0.679,0.723), dt=0.002, duration=1,
and a Ricker with f0=9, amplitude=2.1, t0=0.04 in the small nondimensional-scale
fixture. It includes a nonnegligible initial force.

There is a specific **fully discrete startup qualification** if a source acts
directly on damped DOFs and `f(0)` is appreciable. With zero initial state, the
existing Taylor startup gives `u1=0.5 dt² D^-1 f0`. In the subsequent damped
recurrence this is equivalent to an initial impulse weighted by
`0.5 (I+0.5 dt C_L D^-1) f0`. That spatial weighting differs for interior and
boundary sources. Thus the unchanged startup need not give roundoff reciprocity
for an exchanged boundary/interior pair, despite symmetric semidiscrete operators.
This is a second-order startup error, not negative damping or matrix asymmetry.

Moving B to `(1,0.723)` on the absorber yields maximum relative error over the
same/cross and displacement/velocity pairs of **0.005676, 0.001412, 0.0003526**
for dt=0.004, 0.002, 0.001. All four measured orders lie between 2.00045 and
2.00677. The test retains this evidence rather than silently changing the audited
integrator or forbidding boundary sources. The tested interior sources have
`C_L b_s=0` and show roundoff reciprocity; delaying a Ricker also reduces its
startup tail. A geometrically interior source whose P1 support touches an
absorbing boundary can also fall into the qualified startup case.
Boundary-source high-precision reciprocity requires a separate startup-policy
study if it becomes a scientific requirement.

## MPI ownership and partition consistency

Boundary integration uses only selected owned exterior facets, with sorted unique
tags. DOLFINx's facet assembly and reverse-add accumulation send shared-node
weights to their owners; there is no rank-based duplication. See the
[DOLFINx 0.11 mesh API](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.mesh.html).
Source and receiver point ownership remains the deterministic `v0.3.0` rule.

The MPI probe compares physical-coordinate-ordered consistent matrices, damping
diagonals, constrained masks, full receiver arrays, normal SV reflection, and
both reciprocity pairs. It includes boundaries cut by partitions, off-node
receivers, and a two-triangle mesh on four ranks with empty owned-cell/DOF sets.
Rank-asymmetric malformed, conflicting, and differing otherwise-valid configs
all fail coherently through the existing collective validation path.

Maximum absolute differences against one rank:

| Quantity | 2 ranks | 4 ranks |
| --- | ---: | ---: |
| consistent C and lumped diagonal | 0 | 0 |
| receiver displacement (m) | 5.64e-18 | 5.43e-18 |
| receiver velocity (m/s) | 6.72e-16 | 7.49e-16 |
| late raw trace-norm ratio | 5.97e-16 | 1.48e-15 |
| normal SV reflection proxy | 1.49e-14 | 2.40e-14 |
| reciprocity probe trace arrays | 2.26e-15 | 1.92e-15 |

The normal SV proxy is 0.0243248350179341, 0.0243248350179490, and
0.0243248350179582 for 1/2/4 ranks. Receiver order and essential masks agree;
coordinate differences are at most 8.88e-16. Distributed reciprocal trace errors
remain below 6.67e-14 relative. These tests establish partition consistency;
continuum and physical claims use the independent evidence above.

## Reproduction and regression

The validation environment uses DOLFINx 0.11.0, Python 3.14.6, real-double
PETSc 3.25.5, and MPICH 5.0.1. From the repository root in that environment:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -ra
python -m pytest -s -q tests/absorbing2d/test_operators.py tests/absorbing2d/test_time_energy.py
python -m pytest -s -q tests/absorbing2d/test_physics.py
python -m pytest -s -q tests/absorbing2d/test_mpi.py
mpiexec -n 1 python -m tests.absorbing2d.mpi_worker /tmp/seisfem-absorbing-1.npz
mpiexec -n 2 python -m tests.absorbing2d.mpi_worker /tmp/seisfem-absorbing-2.npz
mpiexec -n 4 python -m tests.absorbing2d.mpi_worker /tmp/seisfem-absorbing-4.npz
python -m tests.absorbing2d.characterize --output /tmp/seisfem-absorber-study.json
python examples/2d/absorbing.py
ruff check .
ruff format --check .
pre-commit run --all-files
git diff --check
```

Run pytest once, not under mpiexec; marked tests launch their own MPI processes.
The optional frequency/beam study is deliberately excluded from the routine
suite. Final outcomes:

| Command/check | Outcome |
| --- | --- |
| `python -m pytest -ra` | **237 passed in 951.45 s (15:51)** |
| Existing 1D / 2D core / source-receiver tests | **98 / 65 / 41 passed**, unchanged |
| New absorber tests | **33 passed** |
| Focused operator and time/energy tests | 21 passed |
| Focused physical/refinement/reciprocity tests | 5 passed in 86.79 s |
| MPI comparison pytest (launches 1/2/4 ranks) | 1 passed in 12.91 s |
| Explicit `mpiexec -n 1`, `-n 2`, `-n 4` workers | All exited 0; values reported above |
| Optional frequency/beam characterization | Completed, exit 0 |
| Public example | Completed, exit 0; both result arrays `(1601,3,2)` |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed; 82 files already formatted |
| `pre-commit run --all-files` | Both Ruff hooks passed |
| `git diff --check`, staged and full-branch whitespace checks | Passed |

The [baseline and final pytest transcripts](2d_absorbing_regression.txt) preserve
the complete regression evidence. The final suite ran alongside part of the
optional characterization study; its elapsed time is not an isolated performance
benchmark. No numerical implementation changed after this regression.

## Limits and next milestone

This work establishes homogeneous, isotropic, axis-aligned rectangular 2D
plane-strain local impedance absorption with positive lumping and explicit
stepping. It does **not** establish exact nonreflection at oblique incidence,
PML-quality absorption, heterogeneous boundary impedance or geology, curved
boundary optimality, anisotropy, 3D, moment-tensor sources, production pure P/S
sources, or validated continuum acceleration receivers. Discrete normal-incidence
reflection remains finite and resolution dependent. Continuum point sources are
singular: receiver/refinement claims concern points away from the source.
Finite-domain results must be interpreted in their specified return windows.
Snapshots remain deferred; no new output-storage mechanism was needed here.

The planned next milestone is **heterogeneous 2D isotropic material fields**,
followed by a single planar interface validated against normal-incidence 1D
reflection/transmission, then oblique P/S conversion and Zoeppritz physics.
This branch does not begin geology or merge into main.
