"""Thin Typer frontend; all validation and numerical work belongs to the backend."""

import json
import logging
from pathlib import Path

import typer

from .config import SimulationConfig

app = typer.Typer(no_args_is_help=True, help="Verified 1D P/S finite-element elastodynamics.")


def read_config(path: Path) -> SimulationConfig:
    """Present declarative validation errors consistently to CLI users."""
    try:
        return SimulationConfig.from_yaml(path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def rank_zero() -> bool:
    from mpi4py import MPI

    return MPI.COMM_WORLD.rank == 0


@app.command()
def validate(model: Path):
    """Validate configuration, supported physics, stability and resolution."""
    config = read_config(model)
    if rank_zero():
        typer.echo(json.dumps({"valid": True, **config.diagnostics()}, indent=2))


@app.command()
def inspect(model: Path):
    """Show the normalized experiment and numerical diagnostics."""
    config = read_config(model)
    if rank_zero():
        typer.echo(
            json.dumps(
                {
                    "config": config.model_dump(mode="json", by_alias=True),
                    "diagnostics": config.diagnostics(),
                },
                indent=2,
            )
        )


@app.command()
def run(model: Path):
    """Run through the same public API used in Python/Jupyter."""
    from .simulation import Simulation

    config = read_config(model)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = Simulation(config).run()
    if rank_zero():
        typer.echo(
            f"Completed {config.time.steps} steps; output: {config.output.directory or 'memory'}"
        )
        typer.echo(f"MPI ranks: {result.metadata['mpi_size']}")


@app.command()
def mesh(model: Path, destination: Path):
    """Write the configured interval mesh to XDMF; builds via the shared backend."""
    from dolfinx.io import XDMFFile
    from mpi4py import MPI

    from .simulation import Simulation

    config = read_config(model)
    exists = destination.exists() or destination.with_suffix(".h5").exists()
    if MPI.COMM_WORLD.allreduce(exists, op=MPI.LOR):
        raise typer.BadParameter("Destination XDMF or companion HDF5 file already exists")
    with Simulation(config) as sim:
        with XDMFFile(sim.comm, destination, "w") as writer:
            writer.write_mesh(sim.operators.mesh)


@app.command()
def info():
    """Report numerical dependency versions and MPI size."""
    import dolfinx
    from mpi4py import MPI
    from petsc4py import PETSc

    from . import __version__

    if rank_zero():
        typer.echo(
            json.dumps(
                {
                    "seisfem": __version__,
                    "dolfinx": dolfinx.__version__,
                    "petsc": PETSc.Sys.getVersion(),
                    "mpi_size": MPI.COMM_WORLD.size,
                }
            )
        )
