# seisfem

A verified finite-element seismic laboratory built with **DOLFINx 0.11**.
It includes the audited 1D P/S reference solver and a homogeneous isotropic
**2D plane-strain experiment API** with vector line forces and arbitrary-coordinate
FE displacement/velocity receivers. Both use positive lumped mass and the same
explicit central-difference integrator. No 3D is implemented.

The [2D source/receiver validation](docs/validation/2d_sources_receivers.md) records
force units, interpolation and MPI ownership, measured P/S arrivals, reciprocity,
and mesh refinement. The [2D numerical core](docs/development/plane_strain_core.md)
remains the operator foundation. `Simulation` and the CLI retain their existing
1D behavior; 2D experiments use `Simulation2D` and `SimulationConfig2D`.

## Install

FEniCSx/PETSc/MPI are binary dependencies, not a universal pip installation.
A supported conda-forge environment is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate seisfem
python -m pip install -e '.[cli,dev]'
seisfem info
```

For the exact tested Linux environment, including headless plotting for the
validation figure:

```bash
conda create -n seisfem-locked --file environments/linux-64.explicit.txt
conda activate seisfem-locked
python -m pip install -r environments/pip-lock.txt
python -m pip install -e '.[cli,dev]'
```

The explicit lock includes binary build URLs/checksums and is architecture-specific;
it is not a portable cluster lock. See [installation notes](docs/user_guide/installation.md).
Qt is absent from the headless lock and backend dependencies. `gui`, `notebook`
and `viz` extras are optional dependency groups; no GUI is implemented.

## Run the same experiment from either frontend

```bash
seisfem validate examples/1d/layered.yaml
seisfem run examples/1d/layered.yaml
python examples/1d/analyze_layered.py
```

```python
from seisfem import Simulation, SimulationConfig

config = SimulationConfig.from_yaml("examples/1d/layered.yaml")
result = Simulation(config).run()
# Optional xarray: result.to_xarray()
```

Output directories must be new. Use a different `output.directory` for another
run, or `null` for in-memory results with `snapshot_stride: 0`. Relative paths
are relative to the launch working directory. There is no numerical logic in
the CLI. Other commands: `inspect MODEL`, `mesh MODEL DESTINATION.xdmf`, and `info`.

```bash
mpiexec -n 4 seisfem run model-with-new-output-directory.yaml
```

All ranks call the same API with identical configuration. Each result contains
**locally owned receivers**, ordered by `receiver_ids`; serial is MPI size 1.
Trace axes are `(time, receiver, component)` and units are m, m/s, m/s². NPZ shards
need no pickle; metadata names every shard. Distributed XDMF/HDF5 stores materials
and strided displacement snapshots. Receiver histories remain in memory until
completion; this is a documented scale limit.

## Run a homogeneous 2D experiment

```bash
python examples/2d/homogeneous.py --cells 240
mpiexec -n 4 python examples/2d/homogeneous.py --cells 240
# Finer case used for the strongest arrival/refinement evidence:
python examples/2d/homogeneous.py --cells 640
```

The executable example uses only `SimulationConfig2D.model_validate(...)` and
`Simulation2D(cfg).run()`, and prints reproducible envelope-peak arrival times.
Results contain complete displacement and centered integer-time velocity arrays
on every rank, shaped `(time, receiver, component)` with components `("x", "z")`.
`result.to_xarray()` adds named coordinates and SI units. The source amplitude is
**N/m**, a line force per unit out-of-plane length. Source directions are normalized.

The [full API and validation example](docs/validation/2d_sources_receivers.md)
explains pre-return measurement windows and remaining S-wave dispersion.
Homogeneous 2D runs support free/fixed boundaries and per-side first-order local
elastic impedance absorption. For a physical free surface with absorbing sides
and bottom, set `boundaries={"left": "absorbing", "right": "absorbing",
"lower": "absorbing", "upper": "free"}`. Run `python examples/2d/absorbing.py`
for a public-API comparison. The [absorbing-boundary validation](docs/validation/2d_absorbing_boundaries.md)
documents normal P/S reflection reduction, preserved free-surface signals,
discrete energy balance, MPI agreement, and limits at oblique incidence.
Snapshots and acceleration receivers are not exposed by the 2D experiment API.

## 1D scientific contract

* Positive-up z in metres, SI throughout; no implicit depth conversion.
* P uses lambda+2mu, S uses mu; neither is an acoustic pressure substitution.
* Materials accept density/vp/vs or density/lambda/mu, with positive solid bulk
  and shear moduli. Every layer interface must be a mesh vertex.
* Degree is explicit but only 1 is verified. Unsupported settings fail validation.
* A point-force amplitude is signed **Pa** in this planar 1D reduction, not N.
* Free means zero traction; fixed means zero displacement. Endpoint impedance
  is a tested 1D absorber, not a multidimensional PML.
* Stability is bounded for the actual discretization; wavelength warnings do
  not certify accuracy. S waves often require finer meshes than P waves.

[Theory](docs/theory/navier_cauchy.md) · [Architecture](docs/architecture/overview.md) ·
[Time integration](docs/numerics/time_integration.md) ·
[Mass lumping](docs/numerics/mass_lumping.md) ·
[API audit](docs/architecture/dolfinx_011_api.md) ·
[Measured verification](docs/validation/results.md) ·
[Critical review and 2D gates](docs/validation/scientific_review.md)

The [independent adversarial audit](docs/validation/adversarial_audit.md) records
the baseline, claim-to-evidence matrix, additional counterexamples, demonstrated
validation correction, and limits of the 1D trust assessment.

![Layered velocity seismograms](docs/validation/layered_seismograms.svg)

## Develop and verify

```bash
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
pre-commit install
```

The suite launches its own two-/four-rank MPI subprocesses. Run pytest once,
not under mpiexec. For batch environments disallowing nested MPI jobs, run
`pytest -m 'not mpi'` locally and schedule the MPI tests separately. Source and
receiver physics, interface amplitudes, convergence and discrete energy have
analytical tests; regression snapshots are not the basis of acceptance.

The license and citation authors remain explicit placeholders pending the
copyright holder's decision. No authorship, institution or license was invented.
