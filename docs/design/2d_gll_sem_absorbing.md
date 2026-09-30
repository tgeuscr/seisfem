# SEM absorbing-boundary extension design

The production triangular condition is sigma*n=-B*v, with
B=Zs*I+(Zp-Zs)*n⊗n, Zp=rho*Vp, Zs=rho*Vs. The weak damping is
integral_ds w·B*v. For layered materials, synchronized DG0 fields give
Zp=sqrt(rho*(lambda+2mu)), Zs=sqrt(rho*mu) from the unique exterior cell.
There is no interior-facet term. Owned exterior facets are tagged once; corners
receive both incident side integrals by ordinary shared-DOF assembly. The public
left/right/lower/upper fields independently select free or absorbing sides.

Triangular P1 assembles consistent C and its nonnegative row sums C_L. The
shared recurrence is (D+dt*C_L/2)u_next=2D*u-(D-dt*C_L/2)u_prev+dt²(f-Ku).
Centered damping does not add a damping CFL: the undamped generalized spectral
condition remains sufficient for nonnegative C_L. The production conservative
absolute-row bound on D^(-1/2)KD^(-1/2) and startup stay unchanged.

Multiplication by (u_next-u_prev)/(2dt) gives exactly
E_next_half-E_prev_half=dt*v_centered·f-dt*v_centered·C_L*v_centered,
where E_next_half=0.5*v_half·D*v_half+0.5*u_next·K*u.
The existing tests validate second-order damped time integration/startup,
normal P/S reflection refinement, free-surface preservation, oblique imperfect
absorption, local facet impedance and corners, reciprocity and discrete loss
balance, and MPI. They do not establish a PML or arbitrary-angle exactness.

SEM will call the same boundary helper and reuse its impedance expression and
facet selection. Only SEM supplies GLL edge quadrature of degree 2p-1 and
extracts the assembled diagonal directly. On an affine axis-aligned edge,
GLL trace collocation and diagonal B make C diagonal to roundoff, including
summed corner contributions. A diagonality guard must reject any violation;
there is no row-sum lumping on the SEM branch. Facet code generation uses the
ordinary FFCx path, separately from the unchanged sum-factorized volume forms.
The no-absorber helper returns None/zeros, preserving the free path. Fixed,
componentwise, VTI and other unsupported SEM extensions remain rejected.
