"""Collective field I/O and partitioned receiver output with completion metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import basix
import dolfinx
import numpy as np
import ufl
from dolfinx import io
from mpi4py import MPI
from petsc4py import PETSc

from . import __version__


def runtime_metadata(config, comm) -> dict:
    """Capture configuration, binary versions and source identity without gathering fields."""
    source_root = Path(__file__).resolve().parents[2]
    identity = None
    if comm.rank == 0:

        def git(*args):
            if not (source_root / ".git").exists():
                return None
            try:
                process = subprocess.run(
                    ["git", "-C", str(source_root), *args],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                return None
            return process.stdout.strip() if process.returncode == 0 else None

        digest = hashlib.sha256()
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        identity = {
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": digest.hexdigest(),
        }
    identity = comm.bcast(identity, root=0)
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "dolfinx": dolfinx.__version__,
        "basix": basix.__version__,
        "ufl": ufl.__version__,
        "petsc": PETSc.Sys.getVersion(),
        "mpi_size": comm.size,
        "mpi_library": MPI.Get_library_version(),
        "scalar_dtype": str(np.dtype(PETSc.ScalarType)),
        "config": config.model_dump(mode="json", by_alias=True),
        "diagnostics": config.diagnostics(),
        "source_identity": identity,
        "initial_conditions": "zero displacement and velocity",
        "receiver_axes": ["time", "receiver", "component"],
        "coordinate": "z_positive_up",
        "energy_units": "J/m^2",
    }


def collective_file_action(comm, action, *, root_only=False):
    """Propagate ordinary Python filesystem failures before the next MPI collective."""
    error = None
    if not root_only or comm.rank == 0:
        try:
            action()
        except Exception as exc:
            error = f"rank {comm.rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(error)
    if any(errors):
        raise OSError("; ".join(e for e in errors if e))


class RunOutput:
    """Never overwrite an existing run directory. Completion is written last."""

    def __init__(self, config, operators, metadata):
        self.comm = operators.comm
        self.path = Path(config.output.directory) if config.output.directory else None
        self.snapshots = None
        self.field = operators.field
        self.metadata = metadata
        if self.path is None:
            return
        collective_file_action(
            self.comm, lambda: self.path.mkdir(parents=True, exist_ok=False), root_only=True
        )
        collective_file_action(
            self.comm, lambda: config.to_yaml(self.path / "config.yaml"), root_only=True
        )
        collective_file_action(
            self.comm,
            lambda: (self.path / "metadata.json").write_text(
                json.dumps({**metadata, "status": "running"}, indent=2)
            ),
            root_only=True,
        )
        with io.XDMFFile(self.comm, self.path / "materials.xdmf", "w") as writer:
            writer.write_mesh(operators.mesh)
            for field in (operators.rho, operators.lam, operators.mu, operators.modulus):
                writer.write_function(field)
        if config.output.snapshot_stride:
            self.snapshots = io.XDMFFile(self.comm, self.path / "wavefield.xdmf", "w")
            self.snapshots.write_mesh(operators.mesh)

    def snapshot(self, time: float, owned: np.ndarray) -> None:
        """Write displacement collectively without gathering it through Python."""
        if self.snapshots is not None:
            self.field.x.array[: len(owned)] = owned
            self.field.x.scatter_forward()
            self.snapshots.write_function(self.field, time)

    def close(self) -> None:
        """Close snapshots collectively, including after an interrupted run."""
        if self.snapshots is not None:
            self.snapshots.close()
            self.snapshots = None

    def complete(self, result) -> None:
        """Write one trace shard per rank, then mark the manifest complete."""
        if self.path is None:
            return
        filename = f"receivers.rank{self.comm.rank:05d}.npz"
        collective_file_action(
            self.comm,
            lambda: np.savez(
                self.path / filename,
                time=result.time,
                receiver_ids=result.receiver_ids,
                receiver_names=np.array(result.receiver_names, dtype=str),
                positions=result.positions,
                component=np.array(result.component),
                displacement=result.displacement,
                velocity=result.velocity,
                acceleration=result.acceleration,
                requested=result.requested,
                energy_time=result.energy_time,
                energy=result.energy,
            ),
        )
        collective_file_action(
            self.comm,
            lambda: (self.path / "metadata.json").write_text(
                json.dumps(
                    {
                        **self.metadata,
                        "status": "complete",
                        "receiver_shards": [
                            f"receivers.rank{i:05d}.npz" for i in range(self.comm.size)
                        ],
                    },
                    indent=2,
                )
            ),
            root_only=True,
        )
