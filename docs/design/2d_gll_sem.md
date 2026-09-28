# Quadrilateral GLL backend design

Base: main/v0.7.1, b4df73b58463415971093af84c01addd0cdb3f6d.

The existing PlaneStrainOperators owns assembly, PETSc action, spectral
diagnostics, and the CentralDifference adapter. Simulation2D adds point-force
and receiver functionals and result handling. Only assembly depends on the
triangular P1 discretization; PointMap2D additionally hardcodes three-node
barycentric interpolation.

Add a small discriminated configuration selector, defaulting to tri_p1.
quad_gll specifies degree 1 through 6 and initially accepts only homogeneous
isotropic material, rectangular structured meshes, and free boundaries.
The initial scope rejected constraints, absorbers, and other materials;
the extensions below add isotropic layers and local absorbers. Existing
configurations retain the existing triangular assembly and point-weight path.

PlaneStrainOperators dispatches only its assembly method to a separate sem2d
module. The SEM module constructs continuous blocked vector Qp using Basix GLL
tensor-product ordering and matching GLL quadrature, and takes the assembled
mass diagonal directly. No row-sum lumping is permitted. Retain assembled M
and K for independent inspection. Shared methods provide stiffness actions,
the conservative spectral bound, SLEPc critical-timestep diagnostics, resource
management, and the unchanged central-difference integrator.

PointMap2D retains localization and deterministic physical cell ownership.
For SEM it pulls points back to the reference quadrilateral and evaluates the
same Basix nodal basis. Loads remain exact discrete point functionals, with
neither quadrature nor mass scaling. High-order interpolation weights may be
negative. Simulation2D forwards the selector and otherwise remains shared.

Installed APIs were inspected: Basix/DOLFINx/FFCx 0.11.0, create_tp_element,
basix.ufl.wrap_element/blocked_element, make_quadrature(rule=gll), and FFCx's
sum_factorization option. Validate collocation, tensor ordering, and compiler
support experimentally before choosing compiler options. Assembled PETSc K
remains the production reference; matrix-free optimization is deferred.

Original probe: GLL mass off-diagonals were exactly zero for p=1,2,4,6.
FFCx 0.11 sum_factorization=True initially raised AssertionError in
codegeneration/access.py:table_access because the standard coordinate element
did not expose tensor factors. The follow-up resolves this by supplying an
explicit degree-1 Basix tensor-product coordinate element to create_mesh.
Structured corner coordinates and physical cell sets match create_rectangle
bitwise. SEM mass and stiffness now compile with sum_factorization=True and
agree with non-factorized assembly to roundoff. The global PETSc matrix action
is still assembled, not matrix-free; the triangular path is unchanged.

Independent NumPy tensor quadrature and polynomial differentiation validate
element matrices. Actual generalized eigenpairs determine experimental dtcrit;
production start retains its existing sufficient row-sum spectral bound.
Test stable and unstable highest-mode evolution and time-consistent energy.
Finite homogeneous P/SV packets use a continuum Fourier reference, separate
from FEM, to measure phase, polarization, field error, and motion before free
boundary returns. Compare equal approximate DOFs, accuracy targets, timestep
sweeps, DOF-step work, and actual assembly/stepping times without assuming an
SEM advantage. Preserve all existing acceptance thresholds.

## Element-aligned isotropic heterogeneity

Reuse the public horizontal `layered` material model and `CellMaterials2D`'s
cell-ID mapping, vertex containment check and coefficient ghost updates. SEM
supplies a tensor-product discontinuous degree-zero coefficient space to that
shared mapper; the triangular caller retains its original DG0 construction.
Only SEM's density and Lamé coefficients become cell functions. Its homogeneous
scalar-coefficient path, GLL quadrature, collocated mass extraction, assembled
stiffness and explicit dynamics remain shared/unchanged. No interface terms or
nodal material averaging are introduced. Reject cut elements, VTI and displacement
constraints; retain tensor-factorized form compilation. Free and absorbing
boundaries are supported through the boundary extension below.

## Local absorbing boundaries

The [absorber design](2d_gll_sem_absorbing.md) extends the shared boundary helper
with GLL edge quadrature and direct diagonal extraction. Volume sum factorization
and assembled stepping remain unchanged. Free/absorbing side combinations are
now supported for homogeneous and element-aligned isotropic layers.
