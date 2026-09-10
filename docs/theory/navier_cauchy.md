# Continuum equations and the implemented reduction

SI units; tension-positive stress; Cartesian positive-up z. The interval stored
in DOLFINx's first geometric slot represents **z**, not depth or physical x.
`lower` and `upper` increase in z (metres). No automatic depth conversion exists:
for depth below datum z0, explicitly use z=z0-depth and reverse layer bounds.
P displacement is u_z; S is u_x (u_y has identical physics but is not co-solved).

In 3D, rho u_tt-div(sigma)=f, epsilon=(grad u+grad u^T)/2,
sigma=lambda tr(epsilon) I+2mu epsilon. With dependence only on z:

* u=(0,0,w): epsilon_zz=w_z; sigma_zz=(lambda+2mu)w_z,
  sigma_xx=sigma_yy=lambda w_z. Thus rho w_tt-[(lambda+2mu)w_z]_z=f_z.
* u=(s,0,0): epsilon_xz=epsilon_zx=s_z/2, sigma_xz=mu s_z.
  Thus rho s_tt-[mu s_z]_z=f_x.

Each polarization solves rho q_tt-(A q_z)_z=f, with A=lambda+2mu (P) or mu (S).
Constant-material speed is sqrt(A/rho), giving Vp or Vs. This is not the
traction-free slender-rod modulus E or an acoustic pressure equation.
Positive 3D strain energy requires rho>0, mu>0, kappa=lambda+2mu/3>0;
negative lambda is allowed. Equivalently Vp²>4Vs²/3. Fluid layers are excluded.
Conversion: mu=rho Vs², lambda=rho(Vp²-2Vs²).

## Weak form and interfaces

Multiply the vector PDE by v vanishing on displacement boundaries, integrate by
parts and use stress symmetry:

$$\int_\Omega\rho\ddot{\mathbf u}\cdot\mathbf v+
\int_\Omega\boldsymbol\sigma(\mathbf u):\boldsymbol\varepsilon(\mathbf v)
=\int_\Omega\mathbf f\cdot\mathbf v+\int_{\Gamma_N}\mathbf t\cdot\mathbf v.$$

Per unit transverse area the scalar weak form is
m(q_tt,v)+a(q,v)=ell(v), with m(p,v)=integral rho p v dz and
a(q,v)=integral A q_z v_z dz. Signed endpoint traction is t=n A q_z.
For the implemented homogeneous displacement constraints, the trial/test space
is V={v in H1(I): v=0 on Gamma_D}; the dynamic solution takes values in V and
the forcing acts in its dual. In 1D, point evaluation is a bounded functional on
H1(I), making the Dirac load compatible with this weak setting. In 2D/3D, point
evaluation is not bounded on H1: a discrete FE point force still exists, but
continuum singularities and energy/global convergence need separate treatment.
A welded interface has continuous q and A q_z. CG trial functions enforce
continuity; the weak form balances flux without differentiating DG0 properties.
Expanding q_h=sum U_j phi_j gives M Uddot+K U=F,
M_ij=integral rho phi_i phi_j and K_ij=integral A phi'_i phi'_j.
UFL constructs only spatial forms.

Fixed endpoints impose q=0. Free endpoints impose natural t=0. The optional
absorbing endpoint imposes t=-Z q_t, with Z=rho c=sqrt(rho A) from the adjacent
layer, producing M Uddot+C Udot+K U=F and a positive endpoint damping coefficient.
This is the exact continuous outgoing impedance for 1D normal incidence, with
residual discrete reflection. It is not a PML or an exact multidimensional ABC.
Nonzero prescribed traction is a future boundary load, not in the current schema.

## Source and interface reference

f(z,t)=B r(t) delta(z-zs), B in N/m² and r dimensionless, yields
F_i=B r(t) phi_i(zs). Partition of unity gives sum F_i=B r(t), independent of h.
There is no nodal-volume division until inverse mass application. Signed B sets
orientation along +z (P) or +x (S). Boundary sources are rejected.

In an infinite homogeneous medium with zero initial conditions and impedance Z,
q(z,t)=B/(2Z) integral_0^{t-|z-zs|/c} r(tau) d tau, zero for negative upper limit.
Thus **velocity**, not displacement, is B/(2Z) r(t-d/c) after the causal front.
The implemented source begins at t=0; an ideal Ricker has infinite tails. Use a
large enough time shift to suppress startup effects.

For incidence from material 1 into 2, 1+R=T and Z1(1-R)=Z2 T give displacement
and velocity coefficients R=(Z1-Z2)/(Z1+Z2), T=2Z1/(Z1+Z2).
Traction signs differ. Flux balance is R²+(Z2/Z1)T²=1. Travel times sum distance/c
in each traversed material. Free/fixed displacement reflection is +1/-1.

## Energy and future plane strain

E=.5 integral rho q_t² dz+.5 integral A q_z² dz is constant for unforced fixed/free
boundaries. With impedance, dE/dt=-sum Z q_t² plus source work. Test with q_t;
internal interface terms cancel. The discrete invariant is derived separately.

Before reduction the vector energy is

$$E=\tfrac12\int_\Omega\rho|\dot{\mathbf u}|^2\,d\Omega+
\tfrac12\int_\Omega\boldsymbol\sigma(\mathbf u):
\boldsymbol\varepsilon(\mathbf u)\,d\Omega.$$

Time-independent symmetric elasticity gives
dE/dt=integral f dot u_t + integral_Gamma_N t dot u_t by taking v=u_t in the
weak form; zero prescribed displacement has zero boundary power. The scalar
energy above follows after selecting one polarization and dividing by transverse
area. Material heterogeneity does not add volume derivatives to this identity.

In 2D use (u_x,u_z), d/dy=0, u_y=0: plane strain, sigma_yy=lambda div u, retaining
3D lambda. Plane stress changes the constitutive law and is not this reduction.

References: Aki & Richards, *Quantitative Seismology*, 2nd ed., University Science
Books (2002); Hughes, *The Finite Element Method: Linear Static and Dynamic Finite
Element Analysis*, Dover (2000). The derivations above fix signs and conventions.

For the wider viscous absorbing-boundary context see Lysmer & Kuhlemeyer (1969),
[Finite Dynamic Model for Infinite Media](https://doi.org/10.1061/JMCEA3.0001144).
The implemented scalar impedance and its dissipation are derived above; this
citation is not a claim that multidimensional absorption has been implemented.
