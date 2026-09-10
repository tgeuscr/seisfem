# Central differences, stability and resolution

With D=diag(m_i)>0 and C=diag(c_i)>=0, approximate derivatives at t_n:

$$D(U^{n+1}-2U^n+U^{n-1})/k^2+C(U^{n+1}-U^{n-1})/(2k)+KU^n=F(t_n).$$

Therefore

$$U^{n+1}={2DU^n-(D-kC/2)U^{n-1}+k^2(F_n-KU^n)\over D+kC/2}.$$

Division is entrywise. Each step needs one sparse matrix-vector product and local
array operations, no solve. Initial a0=D^-1(F0-KU0-CV0),
U^-1=U0-k V0+k² a0/2. Fixed DOFs have zero initial data and are projected to zero
on every update. Stiffness is assembled without artificial boundary diagonals;
projection evolves its free principal subproblem.

Receiver velocity is (U^{n+1}-U^{n-1})/(2k), acceleration the second difference.
An auxiliary next state at final time permits these centered outputs; it is not
recorded as another completed step.

The unforced undamped invariant at t_(n+1/2) is

$$E^{n+1/2}=\tfrac12\|(U^{n+1}-U^n)/k\|_D^2+
\tfrac12(U^{n+1})^T K U^n.$$

Multiply the recurrence by (U^{n+1}-U^{n-1})/2 to get
E^(n+1/2)-E^(n-1/2)=k v_n^T F_n-k v_n^T C v_n. Cross potential is intentional.
Equivalently E=.5 v_half^T(D-k²K/4)v_half+.5 u_mid^T K u_mid, positive on
non-rigid modes if k² lambda_max(D^-1 K)<4. Integer-time physical energy need
not be exactly constant. Free boundaries admit a rigid mode.

## Bound for the supported elements

Interval P1 stiffness with positive A has nonpositive off diagonals and zero row
sum. Gershgorin bounds eigenvalues of D^-1 K by 2 max_i(K_ii/m_i). Constraining
DOFs cannot increase the largest generalized eigenvalue. Hence
k<sqrt(2/max_i K_ii/m_i) suffices. Build computes this from assembled operators
using an MPI maximum. On uniform h, K_ii/m_i<=2 c_max²/h² even at aligned jumps,
so h/c_max also suffices. Config validation enforces k<=safety*h/c_max,
0<safety<1, default 0.9. Build independently checks the assembled bound.
Centered positive diagonal damping retains this sufficient undamped bound via
the dissipative energy identity. These results do not transfer automatically to
higher-order/vector elements.

## Accuracy diagnostics

f_max is a bandwidth of interest, not a compact spectral cutoff. Default 3f0 has
ideal Ricker spectral amplitude 9 exp(-8), about 0.003 of peak. We warn when
c_min/(f_max h)<12 elements/wavelength, an empirical screen, not a guarantee.
Contrast, path length and singularity matter. Uniform homogeneous lumped P1 has
omega_h=(2c/h)sin(kappa h/2); central differences give
sin(omega_num dt/2)=(c dt/h)sin(kappa h/2). These expose separate spatial and
temporal dispersion. Numerical anisotropy requires multidimensional tests.

Convergence tests use standing waves and exact semidiscrete frequencies to
isolate errors. No smooth-solution convergence claim applies at a point source.
References: Hughes (2000), transient dynamics; Hairer, Lubich & Wanner (2006),
*Geometric Numerical Integration*, Störmer–Verlet. The spectral bounds and energy
identities used here are explicitly derived above.
