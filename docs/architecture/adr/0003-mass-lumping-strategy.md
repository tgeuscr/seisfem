# ADR 0003: positive interval-P1 row-sum mass

Status: accepted.

Context: a consistent mass inversion would defeat explicit scalability. Generic
high-order row sums can be zero or negative. Alternatives: consistent mass solves,
P1 row sums, GLL quadrature/tensor elements, enriched simplicial elements.
Decision: allow only affine uniform interval CG1, assembling integral rho*phi_i
as a linear form, with one reverse-add of ghost contributions.
Reasons: equivalence to positive local row sums, direct total-mass verification,
second-order smooth-solution convergence, and a provable stiffness spectral bound.
Consequences: degree is in configuration but values above 1 fail explicitly.
Limitations: no promise that the formula transfers to higher degree/dimension.
Before enabling those, choose element and quadrature together and test positivity,
polynomial exactness, dispersion, stability and convergence. See the mass note.
