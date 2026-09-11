# Verified plane-strain numerical foundation

This milestone adds a Python operator kernel, not a complete 2D seismic frontend.
It supports homogeneous isotropic solids on affine rectangular triangle meshes,
vector P1 displacement, natural boundaries, selected fixed-zero components,
smooth volume loads, and explicit central differences. Point sources, receivers,
absorbers, geology, output systems and 3D remain out of scope. The 1D configuration,
Simulation, CLI and numerical modules retain their original behavior.

## Mechanics and coordinates

Geometric slots 0 and 1 mean physical `(x,z)`, with positive-up z. Nodal component
order is `(u_x,u_z)`. There is no physical y displacement or y dependence:
`epsilon_yy=epsilon_xy=epsilon_yz=0`, but `sigma_yy=lambda*(epsilon_xx+epsilon_zz)`
is generally nonzero. The implemented in-plane stress is

```
epsilon(u) = sym(grad(u))
sigma_xz_block(u) = lambda*tr(epsilon(u))*I_2 + 2*mu*epsilon(u)
rho*u_tt - div(sigma_xz_block(u)) = f.
```

The 3D Lamé parameters are retained unchanged: mu=rho Vs² and
lambda=rho(Vp²-2Vs²). No plane-stress substitution is made. The existing
`Isotropic` validation requires rho>0, mu>0, lambda+2mu/3>0 and representable
positive squared speeds. Admissible negative lambda is tested. Near-incompressible
locking and extreme floating-point scales are not accuracy-certified regimes.

The weak stiffness is `integral epsilon(v):sigma(u) dOmega`; mass and load are
`integral rho v.u dOmega` and `integral v.f dOmega`. These are per unit out-of-plane
length: body force is N/m³, assembled force is N/m, mass is kg/m and energy is J/m.
Natural boundaries mean zero outward traction. Affine-strain tests do not claim
zero boundary traction: their K action includes the required boundary forces;
their integrated elastic energy is the verification quantity.

## Meshes, DOFs and mass

`Rectangle` accepts lower/upper (x,z) corners, positive subdivision counts and
`left`, `right`, `left_right`, or `right_left` diagonals. `PlaneStrainConfig` is
separate from the 1D `SimulationConfig`; it has a homogeneous material and a tuple
of `ZeroDisplacement(side=..., components=...)` constraints. The basis is fixed
at vector triangular P1; there is no silently ignored degree switch.

DOLFINx uses a blocked two-component space. `coordinates` has one row per node,
including ghosts. Owned arrays have `n=2*index_map.size_local` entries, in order
`[ux0,uz0,ux1,uz1,...]`. Operators reject unexpected block layouts. A node's
component c is scalar DOF `2*node+c`. Cell DOF maps contain node/block indices.
Ghost contributions are reverse-added once for mass and loads; field ghosts
are refreshed before each stiffness action.

For a triangle of area S, the consistent element mass is
`rho*S/12 * [[2,1,1],[1,2,1],[1,1,2]] tensor I_2`.
Every component's lumped nodal mass is `rho*S/3 > 0`. Production assembles the
linear form with vector test function dotted into `(1,1)`; independent tests
construct the full element matrices and their row sums. Summing one component's
lumped masses gives rho times domain area; summing both gives twice that value.
There is no cross-component mass coupling. Consistent M is retained for operator
inspection; the explicit update uses only the lumped diagonal.

## Constraints and resource ownership

Side facets are localized geometrically, and their P1 nodes are converted into
selected scalar component DOFs. The resulting `fixed` array is owned-only.
M and K retain their unconstrained entries: no artificial identity rows or
boundary eigenvalues are inserted. Initial displacement/velocity and every
accepted displacement are zero at fixed DOFs through the existing integrator.
Forces at constrained DOFs become reactions and do not move those DOFs.

Spectral calculations restrict to the free principal stiffness submatrix and
corresponding positive diagonal masses. Thus even componentwise constraints
use the correct reduced system. Fully constrained meshes have no dynamic modes
and an infinite timestep bound; no eigenproblem is attempted for an empty system.

`PlaneStrainOperators` is a collective context manager. M, K, temporary spectral
matrices/index sets/vectors, eigensolvers and matvec workspace are explicitly
destroyed. Function-owned vectors remain owned by DOLFINx. All ranks must call
assembly, load assembly, actions, spectral methods, start and close consistently.
Initial-array conversion, shape and finite-value errors in `start()` are collected
before spectral setup: any invalid rank makes every rank raise the same diagnostic
with argument, originating rank and original exception type/message. The caller
must still enter the API collectively; callback failures, process loss and general
MPI fault tolerance are not handled. Small 2/4-rank smoke tests are not
multi-node or performance evidence.

## Spectral stability and stepping

Let D be the lumped mass. On free DOFs form the symmetric operator
`B = D_f^(-1/2) K_ff D_f^(-1/2)`. Its eigenvalues are exactly the generalized
Kx=lambda Dx eigenvalues. With `L=max_i sum_j |B_ij|`, symmetry gives
lambda_max(B)<=L. Therefore `stable_dt=2/sqrt(L)` is a sufficient assembled bound.
This argument uses absolute entries and never assumes the scalar interval
stiffness's off-diagonal signs. It is not an h/c heuristic.

`op.spectral_diagnostic()` independently uses SLEPc Krylov-Schur on B to estimate
its largest eigenvalue, reports the relative eigenpair residual, and also reports
L. The estimated critical timestep is `2/sqrt(lambda_max)`. A converged Ritz
estimate is not a certified upper bound on an unobserved eigenvalue: the default
`op.start(...)` uses L with a safety factor, not the estimated maximum. Sparse
estimates are checked against NumPy dense eigensolutions of independently
hand-assembled triangle matrices on small meshes.

For a mode of frequency omega, central differences have characteristic roots
of unit modulus for `dt*omega < 2`. At equality the roots coalesce at -1 and
arbitrary initial modal velocity can cause linear growth. This is a marginal
threshold, not a robust safe timestep. `start` requires `0<safety<1` (default .9)
and `dt<=safety*stable_dt`, checks consistent rank settings and owned initial-array
shapes, then returns the **unchanged** `CentralDifference` integrator. No damping
or absorbing-boundary operator is introduced.

Each access to `stable_dt` constructs, scales and destroys a free principal
submatrix; `start()` accesses it again. This is a setup operation, not a cheap
cached scalar lookup. Consistent M is also retained throughout the operator's
lifetime. No caching or lifetime redesign is introduced; larger workflows must
account for these costs and preserve collective resource ownership.

The integrator uses U[-1]=U0-dt V0+dt² D^-1(F0-KU0)/2 and centered outputs.
For F=0 its staggered energy is constant up to roundoff. This is the cross-time
invariant documented for central differences, not exact integer-time continuum
energy. The vector tests compare the midpoint expression independently.

## Minimal Python use

```python
import numpy as np
from seisfem.config2d import PlaneStrainConfig
from seisfem.fem2d import PlaneStrainOperators

cfg = PlaneStrainConfig.model_validate(
    {
        "domain": {"cells": [16, 16], "diagonal": "left_right"},
        "material": {"density": 2300, "vp": 3000, "vs": 1600},
        "constraints": [{"side": s} for s in ["left", "right", "lower", "upper"]],
    }
)
with PlaneStrainOperators(cfg) as op:
    dt = 0.8 * op.stable_dt
    step = op.start(dt)  # Zero initial data and zero forcing in this minimal use.
    for n in range(100):
        next_u, velocity, acceleration, energy = step.evaluate(np.zeros(op.n))
        step.advance(next_u)
```

For a separable smooth force f(x,z,t)=g(x,z)T(t), assemble
`load=op.assemble_load(g)` once, call `start(..., force0=load*T(0))`, and evaluate
with `load*T(n*dt)`. General time-dependent forms would need repeated assembly;
no streaming or cached time-dependent load subsystem is added. An explicitly zero
UFL vector is accepted and returns an owned zero vector; UFL simplifies away the
test argument in this special case, so ordinary vector assembly cannot be used. `apply` returns
a borrowed workspace view, valid only until its next call.

## Manufactured solution and independent references

On the unit square define

```
X = sin(pi*x) sin(2*pi*z)
Z = 0.7 sin(2*pi*x) sin(pi*z)
T = cos(2.7*t) + (0.37/2.7) sin(2.7*t)
u = (X,Z) T.
```

Both components vanish on every side. The initial nodal displacement is the P1
interpolant of (X,Z), and initial velocity is .37 times it. This exercises normal
strain, shear, both Lamé terms, mixed spatial derivatives and nonzero initial
velocity. With omega=2.7, the expanded force amplitudes are

```
g_x = [(lambda+2mu)*pi² + mu*4pi² - rho*omega²]*X
      - (lambda+mu)*0.7*2pi²*cos(2pi*x)*cos(pi*z)
g_z = [mu*4pi² + (lambda+2mu)*pi² - rho*omega²]*Z
      - (lambda+mu)*2pi²*cos(pi*x)*cos(2pi*z)
f = g*T.
```

These formulas live in test code, not production. They are checked against the
constant-coefficient Navier identity `mu Laplacian(u)+(lambda+mu)grad(div(u))`,
without calling the production stress helper. Volume loads use quadrature degree
8; integrated FE L2/H1 errors use degree 12 and include between-node variation.
No pointwise nodal error is mislabeled an FE L2 norm.

Spatial refinement retains N=8 as a coarse diagnostic and uses N=16,32,64,128
for fitted orders, on left, right and alternating diagonals. Errors are measured
at t=.13,.20,.31,.37 and over [0,.4]. The time-RMS H1 seminorm is
`sqrt(integral_0^.4 |u_h-u|_H1² dt / .4)`; trapezoidal integration uses spacing
.00125 and is checked against doubled spacing. The H1 gate applies to its
log(error)-versus-log(h) fit (.9<p<1.1), accompanied by a bounded pointwise
E_h/h envelope across refinements (max/min<1.5 at each observation time).
These are finite-refinement regression criteria, not a proof of arbitrary-time
accuracy or a claim that every consecutive pointwise slope lies near one.
Raw errors, pairwise rates and pointwise fits are reported, including the review's
t=.2 counterexample. The original L2 gates at t=.31 remain unchanged. Finest-grid
half-timestep **field differences** are integrated in both L2 and H1, pointwise
and time-RMS, and must be below .1% of the corresponding spatial errors.

Initial acceleration needs a separate qualification. With nodal interpolation,
`a_h(0)=D^-1(F0-K I_h U0)` need not converge strongly to continuum acceleration.
On alternating triangle stars, even the interior action on U=(x²,0) gives .75
or 1.5 times the continuum x component instead of one. This is a strong-form
consistency/initial-projection issue; it does not invalidate the weak stiffness
or displacement convergence. A test-only Ritz projection satisfies
`K_ff R_h U0 = a(U0,phi_f)`, yielding
`a_h(0)=-omega² D_f^-1 [integral rho phi_f U0]`. This compatible initial acceleration
converges in FE L2 for the tested MMS. The projection is not a new production
initialization API. Neither it nor serial/MPI acceleration agreement establishes
continuum acceleration convergence at later times. MPI comparisons establish
partition consistency only.

Temporal convergence has two complementary references. On N=12 the exact forced
semidiscrete solution is available from a dense eigendecomposition of B. If
q_j is an eigenvector, y0_j=q_j^T sqrt(D) U0, and g_j=q_j^T D^-1/2 F0, set
p_j=g_j/(omega_j²-omega²). The exact modal solution is

```
y_j(t) = p_j*T(t) + (y0_j-p_j)*[cos(omega_j*t) + .37/omega_j*sin(omega_j*t)].
```

The chosen problem is nonresonant (checked numerically). This eliminates the
spatial error floor and uses no frequency derived from the time recurrence.
An additional fixed N=64 Richardson study checks temporal refinement directly
on a fine spatial grid. Its successive differences cancel fixed spatial error;
they do not independently measure continuum error. All time comparisons use
four step sizes and report every error/rate.

See [measured verification](../validation/plane_strain_core.md) for results,
limitations, acceptance evidence and exact commands. The relevant binary APIs
are the versioned [DOLFINx mesh API](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.mesh.html)
and [SLEPc EPS API](https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html).
SLEPc/slepc4py 3.25.1 are already in the repository's explicit binary lock;
assembly and ordinary bounded stepping do not import SLEPc. No SciPy dependency
was added. The lock was not recreated for this milestone.
