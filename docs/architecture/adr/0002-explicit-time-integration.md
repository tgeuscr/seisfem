# ADR 0002: central differences with diagonal endpoint damping

Status: accepted.

Context: seismic transients need many cheap steps with defensible stability and
energy behavior. Alternatives: consistent-mass explicit-looking updates requiring
solves; implicit Newmark; velocity Verlet; central displacement differences.
Decision: central differences on owned arrays, a sparse stiffness action, positive
diagonal mass and optional diagonal damping. Fixed-zero constraints are projected.
Reasons: a truly explicit local update; second order; exact discrete energy/work
identity; endpoint impedance retains a diagonal division. UFL stays spatial.
Consequences: initialization uses acceleration, and centered velocity/acceleration
require an auxiliary next state at the final sample. Energy has a half-step axis.
Limitations: fixed dt, no adaptivity, no nonzero moving constraints. The current
callable stiffness/array interface can accept another integrator, but future
implicit methods require an explicit linear-solver contract rather than pretending
that this local-array interface already suffices.
