# Mass lumping: a restricted, verified method

On an affine interval with constant rho, M_e=rho h/6 [[2,1],[1,2]]. Consistent mass
couples accelerations; solving it every step is not our explicit method.
Row sums give D_e=rho h/2 I>0. P1 partition of unity implies
m_i=sum_j M_ij=integral rho phi_i. We assemble this **linear UFL form**, reverse-add
ghosts once, then use owned entries. No consistent matrix is built in production.

Endpoint quadrature gives the same diagonal for P1 with DG0 density. At a jump
each cell contributes its own density; a single nodal material value is wrong.
Interfaces must align with vertices before assigning DG0 fields.

Lumping approximates the kinetic inner product. Smooth solutions retain
second-order displacement accuracy on these intervals, tested by refinement;
consistent and lumped spectra differ. Assembly and diagonal application are
linear in cells/DOFs. PETSc applies the once-assembled stiffness; no Python DOF loop.

## Degree restrictions

Generic high-order row sums need not be positive. Standard P2 triangle vertex
functions integrate to zero; P2 tetrahedron vertex functions have negative
integrals. High-order simplicial lumping needs enriched spaces and suitable
positive quadrature with sufficient exactness. GLL interval nodes with matching
GLL quadrature and tensor products provide a different route for quads/hexes.
Their mass quadrature is generally inexact, especially with variable properties
and curved geometry; each extension needs stability and convergence tests.

Basix node placement alone does not set UFL quadrature. Arbitrary quadrature does
not create a matching collocation element. No legacy `dolfin` quadrature flags
are used. Degree is a config parameter but unsupported values are rejected.

Primary references:
* Cohen, Joly, Roberts & Tordjman (2001), [Higher Order Triangular Finite Elements
  with Mass Lumping for the Wave Equation](https://doi.org/10.1137/S0036142997329554).
* Geevers, Mulder & van der Vegt (2018), [New Higher-Order Mass-Lumped Tetrahedral
  Elements for Wave Propagation Modelling](https://doi.org/10.1137/18M1175549).

These justify specialized high-order methods; they are not implemented here.
