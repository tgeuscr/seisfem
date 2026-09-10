# ADR 0005: FE point evaluation, not arbitrary nodal forcing

Status: accepted.

Context: a Dirac source is a distribution, and mesh-dependent nodal amplitudes can
invalidate interface and refinement studies. Alternatives: regularized Gaussian
with fixed width and quadrature normalization; nearest node; FE dual evaluation.
Decision: F_i=B*r(t)*phi_i(zs), with P1 basis weights computed once on exactly one
owned cell. Elect the lowest containing rank at partitions; reverse-add the load.
Reasons: exact action on the FE test space, partition-of-unity and first-moment
checks, no regularization width, and transparent 1D Green-function comparison.
Consequences: B is Pa (force per transverse area) in the planar 1D reduction;
signed amplitude controls the selected component. Receivers share localization
and interpolation infrastructure, but store traces only on their elected rank.
Limitations: singular-source field convergence is weaker than smooth manufactured
solutions. Boundary sources are rejected. P1 interval basis logic cannot be reused
as general vector point insertion. Replicated point discovery is an initial
scaling limit; DOLFINx distributed ownership/routing should replace it before
large arrays in 2D/3D.
