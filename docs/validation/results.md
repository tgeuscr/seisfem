# Measured verification: 2026-09-10

**33 tests passed**, including serial/2-rank/4-rank execution, in a separately
recreated **headless** environment from the repository lock files
`environments/linux-64.explicit.txt` and `environments/pip-lock.txt`. The first
reproduction run used a fresh XDG cache to exercise JIT compilation; the final
33-test run reused that cache. Ruff check/format and the installed pre-commit
hooks passed. The original environment also passed the 32-test suite before the
additional mesh-output guard test was added. No remote CI execution is claimed.

Stack: Python 3.14.6, DOLFINx/Basix 0.11.0, UFL 2026.1.0, PETSc 3.25.5 real double,
MPICH 5.0.1. [Raw test output](test_run.txt) captures the printed measurements.

## Homogeneous propagation

Source z=1000.7 m, receiver z=1600.3 m, rho=2000 kg/m³, force amplitude 8e6 Pa,
f0=10 Hz, shift=0.15 s. Peak search uses a +/-0.07 s isolated pulse window.
Waveform errors use the analytical retarded Green function in that window,
without fitting or shifting the numerical trace to improve agreement.

| Measurement | P, c=2000 m/s, h=2 m | S, c=1000 m/s, h=0.5 m |
|---|---:|---:|
| dt | 0.0002 s | 0.0002 s |
| Predicted peak | 0.4498 s | 0.7496 s |
| Peak timing error | +0.0002 s | <1e-12 s (sample alignment) |
| Relative peak velocity error | -0.0377% | -0.0211% |
| Relative displacement waveform L2 error | 0.381% | 0.167% |
| Relative velocity waveform L2 error | 0.661% | 0.289% |
| Relative acceleration waveform L2 error | 0.982% | 0.430% |

An exactly aligned S peak is **not** evidence of exact propagation speed: its
whole waveform still has phase/discretization error. Peak tolerances are 2 ms;
waveform gates are 2% velocity and 2.5% displacement/acceleration.

## Welded two-layer interface

Impedances for P are 4e6 and 7.2e6 kg/(m² s), yielding R=-2/7, T=5/7. S impedances
are half these values, so the coefficients are unchanged. The automated tests
use off-node source/receiver positions, h=2 m for P and h=0.5 m for S, dt=0.0002 s.

| Phase | P peak-time error | P velocity waveform error | S peak-time error | S velocity waveform error |
|---|---:|---:|---:|---:|
| Direct | <1e-12 s | 0.441% | <1e-12 s | 0.193% |
| Reflected | +0.0003 s | 1.763% | +0.0002 s | 0.770% |
| Transmitted | +0.00025 s | 1.286% | +0.0001 s | 0.546% |

All peak amplitude errors in these automated cases are below 0.05%. Tests require
2.5 ms timing, 2.5% peak amplitude and 3.5% waveform error, with resolution fixed
explicitly. The looser waveform gate accounts for accumulation along the longer
reflected path, rather than matching a peak alone. It was not enlarged after a
failed test.

The runnable YAML example uses nodal positions and dt=0.0004 s. Its independently
measured peaks are 0.3500, 0.9504 and 0.8500 s versus 0.3500, 0.9500 and 0.8500 s;
velocities are 1.000603, -0.286043 and 0.714427 m/s versus 1, -2/7 and 5/7 m/s.
[Machine-readable measurements](layered_measurements.json) and
[the seismogram](layered_seismograms.svg) are included. Raw fields, traces and
normalized configuration are in `outputs/layered/` in the working checkout.

## Convergence and energy

Spatial standing-wave error is the **integrated FE L2 error**, including between-
node interpolation, with quadrature degree 8. N=20,40,80 cells, dt=0.00005 s and
final time 0.37 s give errors 2.26885e-3, 5.67668e-4, 1.41942e-4 m sqrt(m), and
rates **1.99884, 1.99975**. Temporal refinement keeps N=20 and compares against the
exact semidiscrete frequency with nonzero initial velocity: dt=0.01,0.005,0.0025 s
gives errors 1.75282e-4, 4.38132e-5, 1.09528e-5 in the lumped mass norm and rates
**2.00024, 2.00006**. Both gates require rates between 1.95 and 2.05.

The unforced undamped half-step energy has relative range **1.05e-13** over
2000 updates, below the roundoff allowance 2e-11. For impedance damping, the
maximum absolute discrete energy-balance defect is **1.74e-13 J/m²** and final
energy is **1.20e-6** of initial energy in the tested escaping-pulse experiment.
These concern the derived cross-time discrete energy, not an unjustified claim
of exact integer-time physical energy conservation.

Fixed/free reflected velocity signs are -/+; waveform errors are about 1.93%.
At h=2 m, dt=0.0004 s the impedance endpoint's residual reflected peak is
**0.0623%** of incident amplitude, below the resolution-specific 0.3% gate.

Point-source P waveform errors on h=8,4,2 m at fixed CFL decrease from 10.82% to
2.65% to 0.661%. This checks refinement and source strength without assuming
smoothness at the source singularity.

## MPI and I/O

Two- and four-rank tests compare displacement, velocity, acceleration and global
energy with serial results. The source is exactly at the density interface and
a receiver is colocated there. Two receivers with four ranks force empty shards;
each receiver is recorded exactly once. Trace tolerance is 2e-10 relative plus
a 2e-10 scale-based absolute term (acceleration amplifies subtraction roundoff).
Energy tolerance is 1e-11 relative. Snapshots/material files execute collectively;
mesh readback and exact snapshot times are checked. CLI/API receiver arrays match
bit-for-bit on the same serial configuration. Existing output directories and
pre-existing HDF5 companions to CLI mesh destinations are protected.

## Failures that informed the verification

The initial S-wave experiment at h=2 m was stable and passed the 12-elements-per-
wavelength screen, but had a 5.29% homogeneous velocity waveform error and failed
the 2% gate. Refinement to h=0.5 m reduced the error to 0.289% with the same
scientific comparison. This is evidence that a wavelength heuristic is only a
screen and that stable timestep selection cannot certify seismic phase accuracy.

See the [critical scientific review](scientific_review.md) for unverified regimes,
real technical debt, performance risks and the explicit 2D transition gates.
