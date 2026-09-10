"""Public lifecycle: one backend for Python, CLI and future graphical frontends."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from .config import SimulationConfig
from .fem1d import IntervalOperators
from .output import RunOutput, runtime_metadata
from .points import PointMap
from .results import SimulationResult
from .sources import ricker
from .timestepping import CentralDifference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Progress:
    """Frontend-neutral event, emitted on all ranks at about one percent intervals."""

    step: int
    total_steps: int
    time: float
    rank: int


class Simulation:
    """Build/run a validated experiment collectively on the supplied communicator.

    Use a context manager to inspect operators or explicitly build()/close().
    A simple run() builds and closes automatically. Configuration is immutable.
    """

    def __init__(self, config: SimulationConfig, comm=None):
        self.config = config
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.operators: IntervalOperators | None = None

    def build(self) -> Simulation:
        """Assemble runtime operators once; calling again is harmless."""
        if self.operators is None:
            signatures = self.comm.allgather(self.config.model_dump_json(by_alias=True))
            if len(set(signatures)) != 1:
                raise ValueError("All ranks must use identical simulation configuration")
            self.operators = IntervalOperators(self.config, self.comm)
        return self

    def close(self) -> None:
        """Collectively release runtime operators."""
        if self.operators is not None:
            self.operators.close()
            self.operators = None

    def __enter__(self) -> Simulation:
        return self.build()

    def __exit__(self, *_):
        self.close()

    def run(self, progress: Callable[[Progress], None] | None = None) -> SimulationResult:
        """Run from zero initial data, returning rank-local named receiver traces."""
        owns_build = self.operators is None
        self.build()
        try:
            return self._run(progress)
        finally:
            if owns_build:
                self.close()

    def _run(self, progress) -> SimulationResult:
        cfg, op, comm = self.config, self.operators, self.comm
        for warning in cfg.diagnostics()["warnings"]:
            if comm.rank == 0:
                logger.warning(warning)
        receivers = PointMap(op.V, [r.position for r in cfg.receivers], cfg.h)
        source = cfg.source
        load = (
            PointMap(op.V, [source.position], cfg.h).unit_load()
            if source is not None
            else np.zeros(op.n)
        )

        def force(time):
            if source is None:
                return load
            return load * source.amplitude * ricker(time, source.frequency, source.time_shift)

        stepper = CentralDifference(
            op.mass,
            op.damping,
            op.fixed,
            cfg.time.dt,
            op.apply,
            np.zeros(op.n),
            np.zeros(op.n),
            force(0),
        )
        sample_steps = np.unique(
            np.append(np.arange(0, cfg.time.steps + 1, cfg.output.receiver_stride), cfg.time.steps)
        )
        shape = (len(sample_steps), len(receivers.ids), 1)
        traces = {
            key: np.full(shape, np.nan) for key in ("displacement", "velocity", "acceleration")
        }
        requested = np.array(
            [[key in cfg.receivers[i].quantities for key in traces] for i in receivers.ids],
            dtype=bool,
        ).reshape(-1, 3)
        metadata = runtime_metadata(cfg, comm)
        metadata["mesh"] = {
            "cells": cfg.mesh.cells,
            "dofs": op.V.dofmap.index_map.size_global,
            "cell_type": "interval",
            "geometry_degree": 1,
        }
        metadata["assembled_dt_bound_s"] = op.spectral_dt_bound
        output = RunOutput(cfg, op, metadata)
        energy_times, energies = [], []
        sample_index = 0
        try:
            for n in range(cfg.time.steps + 1):
                time = n * cfg.time.dt
                record_energy = (
                    cfg.output.energy_stride > 0
                    and n < cfg.time.steps
                    and n % cfg.output.energy_stride == 0
                )
                nxt, velocity, acceleration, local_energy = stepper.evaluate(
                    force(time), energy=record_energy
                )
                if sample_index < len(sample_steps) and n == sample_steps[sample_index]:
                    for j, (key, values) in enumerate(
                        zip(traces, (stepper.current, velocity, acceleration), strict=True)
                    ):
                        sampled = receivers.evaluate(values)
                        sampled[~requested[:, j]] = np.nan
                        traces[key][sample_index, :, 0] = sampled
                    sample_index += 1
                if record_energy:
                    energies.append(comm.allreduce(local_energy, op=MPI.SUM))
                    energy_times.append(time + cfg.time.dt / 2)
                if cfg.output.snapshot_stride and (
                    n % cfg.output.snapshot_stride == 0 or n == cfg.time.steps
                ):
                    output.snapshot(time, stepper.current)
                if progress is not None and (
                    n % max(1, cfg.time.steps // 100) == 0 or n == cfg.time.steps
                ):
                    progress(Progress(n, cfg.time.steps, time, comm.rank))
                if n < cfg.time.steps:
                    stepper.advance(nxt)
            result = SimulationResult(
                time=sample_steps * cfg.time.dt,
                receiver_ids=receivers.ids,
                receiver_names=tuple(cfg.receivers[i].name for i in receivers.ids),
                positions=np.array([cfg.receivers[i].position for i in receivers.ids]),
                component="z" if cfg.mode == "P" else "x",
                requested=requested,
                energy_time=np.array(energy_times),
                energy=np.array(energies),
                metadata=metadata,
                **traces,
            )
        finally:
            output.close()
        output.complete(result)
        return result
