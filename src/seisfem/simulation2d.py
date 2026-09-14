"""Public homogeneous plane-strain source-to-receiver workflow."""

import numpy as np
from mpi4py import MPI

from ._collective import collective_input, identical_input
from .config2d import PlaneStrainConfig, SimulationConfig2D
from .fem2d import PlaneStrainOperators
from .points2d import PointMap2D
from .results2d import SimulationResult2D
from .sources2d import PointForce2D


class Simulation2D:
    """Run collectively with identical config, returning complete traces on every rank.

    Accept a validated SimulationConfig2D or a mapping. Mapping validation occurs
    collectively, so rank-specific malformed input fails before mesh assembly.
    As with any MPI API, all ranks must enter the call. Model validation performed
    by the caller before entering this class cannot be guarded by this class.
    """

    def __init__(self, config, comm=None):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.config = collective_input(
            self.comm, lambda: SimulationConfig2D.model_validate(config), "2D simulation config"
        )
        identical_input(
            self.comm, self.config.model_dump_json(by_alias=True), "2D simulation config"
        )
        self.operators = None
        self.source = self.receivers = None

    def build(self):
        """Assemble operators and cache all point supports; idempotent and collective."""
        if self.operators is None:
            cfg = self.config
            kernel = PlaneStrainConfig(
                domain=cfg.domain,
                material=cfg.material,
                constraints=cfg.constraints,
                boundaries=cfg.boundaries,
            )
            self.operators = PlaneStrainOperators(kernel, self.comm)
            try:
                self.source = PointForce2D(self.operators.V, cfg.source) if cfg.source else None
                self.receivers = PointMap2D(self.operators.V, [r.position for r in cfg.receivers])
            except Exception:
                self.close()
                raise
        return self

    def close(self):
        """Release collective PETSc resources; safe to call repeatedly."""
        if self.operators is not None:
            self.operators.close()
            self.operators = None
        self.source = self.receivers = None

    def __enter__(self):
        return self.build()

    def __exit__(self, *_):
        self.close()

    def run(self):
        """Propagate from rest; a standalone call builds and closes automatically."""
        owns_build = self.operators is None
        self.build()
        try:
            return self._run()
        finally:
            if owns_build:
                self.close()

    def _run(self):
        cfg, op, receivers = self.config, self.operators, self.receivers
        zero = np.zeros(op.n)
        force = self.source if self.source is not None else lambda time: zero
        stepper = op.start(cfg.time.dt, force0=force(0), safety=cfg.time.safety)
        time = np.arange(cfg.time.steps + 1) * cfg.time.dt
        shape = (len(time), len(receivers.ids), 2)
        displacement, velocity = np.empty(shape), np.empty(shape)
        for n, instant in enumerate(time):
            # Force at t_n produces u[n+1] and v[n]. At the final sample the
            # lookahead is evaluated solely to give v at the requested final time.
            nxt, centered_velocity, _, _ = stepper.evaluate(force(instant))
            displacement[n] = receivers.evaluate(stepper.current)
            velocity[n] = receivers.evaluate(centered_velocity)
            if n < cfg.time.steps:
                stepper.advance(nxt)
        # Gather receiver histories only, once after propagation; never wavefields.
        pieces = self.comm.allgather((receivers.ids, displacement, velocity))
        shape = (len(time), len(cfg.receivers), 2)
        full_u, full_v = np.empty(shape), np.empty(shape)
        for ids, local_u, local_v in pieces:
            full_u[:, ids, :] = local_u
            full_v[:, ids, :] = local_v
        return SimulationResult2D(
            time=time,
            receiver_coordinates=np.array([r.position for r in cfg.receivers]).reshape(-1, 2),
            receiver_names=tuple(r.name for r in cfg.receivers),
            displacement=full_u,
            velocity=full_v,
            metadata={
                "config": cfg.model_dump(mode="json", by_alias=True),
                "mpi_size": self.comm.size,
                "receiver_owners": receivers.owners.tolist(),
                "source_owner": int(self.source.points.owners[0]) if self.source else None,
                "source_units": "N m-1",
                "velocity_time_convention": "integer-step centered difference",
                "result_distribution": "complete on every rank",
            },
        )
