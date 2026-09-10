# Verification gates

All solver tests use DOLFINx. Analytical tests precede regression fixtures.

1. Ricker symmetry, peak, roots; material conversion and stable-solid constraints;
   config round-trip and rejection of unsupported, misaligned and unstable cases.
2. Assembled mass/stiffness including density jumps, positive/total mass, rigid
   nullspace, source partition of unity and first moment.
3. Homogeneous P and S: delayed Ricker **velocity**, signed amplitude B/(2Z),
   travel time and refinement, with off-node source/receiver positions.
4. Two layers: direct/reflected/transmitted velocity amplitudes and arrivals
   against welded-interface coefficients, before boundary reflections interfere.
5. Standing waves: staggered energy invariant, spatial and isolated temporal
   convergence, including nonzero initial velocity.
6. Free/fixed reflection signs; impedance residual reflection and energy decay.
7. Serial/2-rank/4-rank traces with partition-boundary points and empty shards;
   output round-trip and CLI/Python agreement.

Tolerances must distinguish sampling/dispersion/interpolation errors from
floating-point identities. Record actual results in `results.md` after execution.

Next manufactured vector test: u=(sin(pi x/Lx)sin(pi z/Lz),
cos(pi x/Lx)sin(2pi z/Lz))cos(omega t), f=rho u_tt-div(sigma(u)), exact boundary
and initial data. Require L2 O(h²), H1 O(h), and O(dt²) separately. This is a future
2D gate, not a current claim.
