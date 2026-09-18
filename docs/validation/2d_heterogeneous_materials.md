# Mesh-aligned heterogeneous 2D elasticity — v0.5.0

This milestone adds horizontal, cellwise isotropic materials to the validated
plane-strain solver. It establishes one normal-incidence P interface against
analytical impedances and the audited 1D operators. No oblique scattering physics
or heterogeneous absorber generalization is included.

## Configuration and assembly

Coordinates remain `(x,z)` in metres, with **positive-up z**, not depth. Existing
homogeneous `material={"density": ..., "vp": ..., "vs": ...}` inputs are unchanged.
For a domain spanning z=-2000 to 2000 m, a layered material is:

```python
material = {
    "type": "layered",
    "layers": [
        {"lower": -2000, "upper": 0, "material": {"density": 2000, "vp": 2000, "vs": 1000}},
        {"lower": 0, "upper": 2000, "material": {"density": 2400, "vp": 3000, "vs": 1500}},
    ],
}
```

Layers reuse the existing `Layered`, `Layer`, and `Isotropic` models. They must
be ordered by increasing z, exactly cover the domain, and have no gaps or overlaps.
Every interface must coincide with a horizontal row of the generated uniform
triangular mesh, allowing only floating-point roundoff in the row index. Each
layer spans at least one row. A separate check of every cell's vertices rejects
cut cells before assembly; a centroid lookup cannot silently move an interface.
At an interface there are two cell material traces. Shared displacement nodes
are not assigned a single interpolated or averaged material.

`CellMaterials2D` in `materials2d.py` constructs scalar DG0 fields for
rho [kg/m³], lambda [Pa], and mu [Pa], fills owned/ghost cell DOFs, and refreshes
ghost values. Cellwise `mu=rho Vs²`, `lambda=rho(Vp²-2Vs²)` use the ordinary 3D
Lamé moduli under plane strain. Direct Lamé input remains supported. The reused
admissibility validator requires finite values, rho>0, mu>0 and
lambda+2mu/3>0; admissible negative lambda is allowed.

The existing vector P1 forms become

\[
m(u,w)=\int_\Omega\rho\,u\cdot w\,dx,\qquad
a(u,w)=\int_\Omega[\lambda\,\mathrm{tr}(\epsilon(u))I
 +2\mu\epsilon(u)]:\epsilon(w)\,dx.
\]

There are **no ad hoc internal-interface traction terms**. Conforming FE
displacement is continuous, and traction continuity follows from the weak form.
The consistent mass is row-sum lumped; stiffness, constraints, sources, receivers,
and `CentralDifference` retain their existing semantics. Homogeneous inputs keep
their original scalar-coefficient forms; identical-property DG0 layers provide
an independently tested homogeneous limit of the new path.

`SimulationConfig2D` and `Simulation2D(cfg).run()` need no new workflow. The
[public example](../../examples/2d/layered.py) uses the material mapping above,
an ordinary vector Ricker line force, and arbitrary-coordinate receivers. The
force amplitude remains **N/m**; displacement/centered-velocity arrays retain
`(time, receiver, 2)`, components `(x,z)`, and complete results on every MPI rank.
That force excites both P and SV and is not the pure-P validation initial field.

**Historical v0.5.0 absorber rule:** this material milestone rejected absorbing
boundaries in layered configurations, including one-layer/equal-property models.
The subsequent [heterogeneous-absorber milestone](2d_heterogeneous_absorbers.md)
adds validated facet-local impedance using the adjacent cell's DG0 material.
Current layered configurations support the existing absorbing-boundary syntax;
free/fixed semantics remain unchanged. The measurements in this v0.5.0 report
retain their original free/fixed-boundary scope.

## Operator, homogeneous-limit, and stability evidence

`test_materials.py` already covers cell assignment on all four supported triangle
diagonal patterns; independent triangle mass/stiffness formulas; total mass;
positive lumped mass; stiffness symmetry and nonnegative energy; affine-strain
energy; density/speed scaling; invalid materials, gaps, overlaps, coverage, and
alignment; and the separate cut-cell guard. The small two-layer mass probe gives
`sum(mass_x)=sum(mass_z)=6(2.3+4.1)=38.4 kg/m`.

One-layer and two-identical-layer DG0 matrices agree with homogeneous matrices
to tight floating-point tolerance (matrix comparisons use relative tolerance
3e-15 with small absolute allowances for near-zero entries). The maximum measured
relative receiver difference is **1.64e-15 for displacement** and **3.49e-15 for
velocity**, including component-fixed runs. Separately saved v0.4.0 free, fixed,
and absorbing baselines were compared during implementation: homogeneous M, K,
lumped mass, damping, safe timestep, displacement, and velocity were **bitwise
unchanged**.

Runtime timestep acceptance still uses the assembled spectral upper bound for
`D_f^(-1/2) K_ff D_f^(-1/2)` with the existing safety factor. It is not replaced by
`h/max(Vp)`. Doubling Vp and Vs only in the already faster upper layer changes
the small probe's bound from **0.0706646921 s to 0.0353323461 s**; the previously
admissible requested step is rejected. Uniform density scaling leaves the bound
unchanged; doubling both speeds everywhere halves it.

## Normal-incidence reference and measurement

The interface is z=0. Medium 1 below it has rho1=2000 kg/m³, Vp1=2000 m/s,
Vs1=1000 m/s; medium 2 above has rho2=2400 kg/m³, Vp2=3000 m/s, Vs2=1500 m/s.
Thus Z1=4.0e6 and Z2=7.2e6 kg/(m² s), an impedance ratio of 1.8.

For a fixed **positive-z displacement component**, write incident, reflected,
and transmitted fields as

\[
u_i=Aq(t-t_I-z/c_1),\quad u_r=ARq(t-t_I+z/c_1),\quad
u_t=ATq(t-t_I-z/c_2).
\]

Displacement continuity gives `1+R=T`. Their normal stresses are respectively
`-Z1 A q'`, `+Z1 A R q'`, and `-Z2 A T q'`, so traction continuity gives
`Z1(1-R)=Z2 T`. Therefore the signed displacement (also velocity) coefficients are

\[
R=\frac{Z_1-Z_2}{Z_1+Z_2}=-\frac27=-0.2857142857,\qquad
T=\frac{2Z_1}{Z_1+Z_2}=\frac57=0.7142857143.
\]

The reflected sign is negative in this fixed-component convention. Continuum
energy-flux fractions include the impedance weighting:
`R_E=R²=0.0816326531`, `T_E=(Z2/Z1)T²=0.9183673469`, and `R_E+T_E=1`.
Squared displacement alone is not the transmitted energy fraction.

The test-only initial field is `u=grad Phi`, with
`Phi=A_u(z+1600) exp(-((z+1600)/w)²)`, A_u=1 m, w=Vp1/(pi*6 Hz), and
`v=-Vp1 partial_z u`. It is analytically irrotational and travels upward.
`PlaneStrainOperators.start(u0, v0)` initializes it; no public pure-P source was
added. The rectangle is `[-400,400] x [-4000,4000]` m. Side constraints `u_x=0`
admit the plane P reference; upper/lower sides are traction-free. The equivalent
1D run uses the existing `IntervalOperators`, `PointMap`, and `CentralDifference`
with the same materials, z geometry, initial packet, grid spacing, and timestep.

Record at `(17.3,-800.3)` and `(17.3,800.3)` m with dt=0.0005 s to 1.4 s.
Predicted incident/reflected/transmitted centers are **0.39985, 1.20015, and
1.0667667 s**. Significant lower/upper boundary returns arrive only after about
2.8/3.2 s. The Gaussian tails are not strictly compact; their startup values at
the outer boundaries are negligible in these windows.

For each arrival use a window of half-width `0.8/f0` and project the signed
displacement onto the analytical Ricker-shaped displacement template:
`a_fit = dot(q,u)/dot(q,q)`. Divide reflected and transmitted fits by the incident
fit. These are **signed displacement-amplitude proxies**, sensitive to both
amplitude and phase dispersion, not direct integrated energy-flux measurements.

| h (m) | R, 2D | absolute R error | T, 2D | absolute T error | 2D/1D trace relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | -0.2510023047 | 3.47120e-2 | 0.6545890112 | 5.96967e-2 | 4.38995e-5 |
| 10 | -0.2843644113 | 1.34987e-3 | 0.7094598813 | 4.82583e-3 | 1.11352e-5 |
| 5 | -0.2859068871 | 1.92601e-4 | 0.7138422264 | 4.43488e-4 | 2.90408e-6 |

The finest 1D result is **R=-0.2859072565063761, T=0.7138428662938844**;
the corresponding 2D values are **R=-0.2859068871453366,
T=0.7138422264418576**. The full two-receiver trace norm difference is
**2.9040826e-6 relative**. Errors decrease under refinement with all physical
parameters fixed; no formal convergence order is claimed. The coarse mesh has
substantial propagation dispersion. At h=5 m there are 22.2 grid intervals per
P wavelength at 3f0 in medium 1. The maximum measured transverse displacement
also decreases: 1.80e-4, 5.84e-5, 1.47e-5 m.

In the **zero-contrast** h=20 m control, the layered and original homogeneous
traces agree exactly: relative difference **0**, maximum difference **0 m**,
and fitted reflected coefficient **-6.47e-15**. Its raw transmitted fit is
0.86413 because dispersion over the longer propagation path affects this
fixed-template estimator equally in both descriptions; it is not interface
reflection. [Saved measurements](2d_heterogeneous_measurements.json) retain the
full numerical values from the completed study, without a new standalone run.

## MPI and regression coverage

`test_interface.py` covers analytical signs/energy sum, the direct 1D comparison,
three-resolution refinement and zero contrast. `test_mpi.py` launches 1/2/4-rank
workers comparing the physical-coordinate cell map, mass and stiffness matrices,
stiffness action, safe timestep, receiver ordering and production source traces.
It also checks ghost material values, empty-owned-cell ranks and coherent failure
for rank-asymmetric invalid or differing configurations.

| Maximum absolute difference from one rank | 2 ranks | 4 ranks |
| --- | ---: | ---: |
| physical cell material map / lumped mass / safe timestep | 0 | 0 |
| consistent M | 1.11e-16 | 1.11e-16 |
| K | 2.84e-14 | 2.84e-14 |
| K action | 1.10e-13 | 1.60e-13 |
| displacement (m) | 6.48e-17 | 1.12e-16 |
| centered velocity (m/s) | 1.78e-15 | 2.13e-15 |

The MPI operator probe uses the small material/geometry fixture; these absolute
matrix errors are not bounds for arbitrary units or model sizes. Receiver peak
relative differences are below 1.20e-14. MPI agreement establishes partition
consistency; the separate analytical/1D tests establish the normal-incidence
physical result. The last focused run passed **25 tests in 84.89 s**. The full
release regression and tool outcomes are recorded in the
[release check transcript](2d_heterogeneous_release_checks.txt).

## Reproduction, version, and limits

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pip install --no-deps -e .
python -m pytest -q tests/heterogeneous2d
python -m pytest -ra
ruff check .
ruff format --check .
pre-commit run --all-files
git diff --check
python examples/2d/layered.py
```

Run pytest once, not under mpiexec: the MPI test launches its own processes.
For separate scheduled probes, use `mpiexec -n 2` or `-n 4` with
`python -m tests.heterogeneous2d.mpi_worker /tmp/heterogeneous-N.npz`.
The optional `python -m tests.heterogeneous2d.interface --output FILE.json`
reproduces the full interface study; it is unnecessary when merely inspecting
the saved evidence and duplicates the expensive work in `test_interface.py`.

The runtime release version is **0.5.0**, with `seisfem.__version__` the single
source. The existing Hatchling backend reads that value through its
[standard version configuration](https://hatch.pypa.io/latest/version/).
Reinstall editable metadata after a version change; restart already-imported
notebook kernels to see the new runtime value. The existing output readback test
checks agreement among runtime, installed distribution and written metadata.

This release is limited to horizontal mesh-aligned layers of isotropic,
lossless solids under plane strain. It does not validate oblique conversion,
Snell/Zoeppritz amplitudes, critical/evanescent waves, nonconforming or curved
interfaces, arbitrary geology, anisotropy, attenuation, heterogeneous absorbing
boundaries, PML, moment tensors, or 3D. No snapshots or acceleration receivers
were added. Below-critical oblique P/SV interface validation is the next milestone.
