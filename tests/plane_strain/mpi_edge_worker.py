"""Collective argument rejection and empty ownership probes (run with mpiexec)."""

import traceback

import numpy as np
import ufl
from mpi4py import MPI

from seisfem.fem2d import PlaneStrainOperators
from tests.plane_strain.helpers import configuration

comm = MPI.COMM_WORLD

try:
    with PlaneStrainOperators(configuration(n=2, fixed=True), comm) as op:
        for name in ["u0", "v0", "force0"]:
            for kind in ["conversion", "shape", "finite"]:
                value = np.zeros(op.n)
                if comm.rank == 0:
                    value = {
                        "conversion": ["not numeric"],
                        "shape": np.zeros(op.n + 1),
                        "finite": np.full(op.n, np.nan),
                    }[kind]
                message = None
                try:
                    op.start(0.001, **{name: value})
                except ValueError as error:
                    message = str(error)
                messages = comm.allgather(message)
                assert all(messages), messages
                assert len(set(messages)) == 1, messages
                assert name in message and "rank 0" in message, message
                assert {"conversion": "ValueError", "shape": "shape", "finite": "finite"}[
                    kind
                ] in message
        # Only the interior vertex is free: at least one rank has no free DOFs.
        assert any(comm.allgather(not np.any(~op.fixed)))
        assert op.spectral_diagnostic().lambda_max > 0
        assert np.isfinite(op.stable_dt)
        zero = op.assemble_load(ufl.as_vector([0.0, 0.0]))
        assert zero.shape == (op.n,)
        np.testing.assert_array_equal(zero, 0)
        step = op.start(0.001, force0=zero)
        nxt, velocity, acceleration, _ = step.evaluate(zero)
        for values in [nxt, velocity, acceleration]:
            np.testing.assert_array_equal(values, 0)
    with PlaneStrainOperators(configuration(n=1, fixed=True), comm) as op:
        assert comm.allreduce(int(np.count_nonzero(~op.fixed))) == 0
        if comm.size == 4:
            assert any(comm.allgather(op.n == 0))
        assert op.spectral_diagnostic().lambda_max == 0
        assert op.stable_dt == float("inf")
        zero = op.assemble_load(ufl.as_vector([0.0, 0.0]))
        assert zero.shape == (op.n,)
        np.testing.assert_array_equal(zero, 0)
        step = op.start(0.001, u0=np.ones(op.n), force0=np.ones(op.n))
        nxt, velocity, acceleration, _ = step.evaluate(np.ones(op.n))
        for values in [nxt, velocity, acceleration]:
            np.testing.assert_array_equal(values, 0)
    if comm.rank == 0:
        print(f"MPI edge cases passed on {comm.size} ranks", flush=True)
except Exception:
    traceback.print_exc()
    comm.Abort(1)
