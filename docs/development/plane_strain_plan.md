# Narrow 2D plane-strain milestone plan

Baseline: main / v0.1.1, commit 43f3dd70c1875a7dafe37910fc98e7c490e9b5dd.
Branch: feature/2d-plane-strain-core. This plan was stated before coding.
Pre-existing .gitignore and untracked scripts changes belong to the user and
are excluded from milestone commits.

The 1D configuration, interval operators, Simulation, PointMap, CLI, output and
CentralDifference remain unchanged. Reuse only Isotropic material validation
and the dimension-independent owned-array time integrator.

1. Add separate immutable rectangular/P1/plane-strain configuration and vector
   operators. Homogeneous 3D Lamé parameters, consistent/lumped mass, tensor
   stiffness, natural boundaries and selected homogeneous component constraints.
2. Verify hand triangle matrices, component ordering, rigid translations/rotation,
   affine strain/stress/energy, symmetry and definiteness on several diagonals.
3. Establish a sufficient timestep from the absolute row sums of the symmetric
   mass-scaled free operator (no scalar off-diagonal-sign assumption). Add SLEPc
   largest-eigenvalue diagnostics; compare to independent dense spectra and
   observe central differences below/above the true spectral threshold.
4. Add an independently derived smooth forced vector manufactured solution;
   measure integrated L2 and H1 spatial convergence, then temporal convergence
   against its exact forced semidiscrete modal solution to remove spatial error.
5. Run the full 1D regression, dedicated 2D tests, MPI smoke, Ruff and pre-commit;
   document measured results and exact reproduction commands.

No 2D sources, receivers, absorbers, geology, output system, GUI or 3D. Commits
separate this plan, operator/algebraic verification, spectral dynamics, and MMS.
