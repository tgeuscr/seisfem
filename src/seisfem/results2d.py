"""Plain NumPy receiver results for homogeneous plane-strain experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult2D:
    """Complete ordered receiver traces on every rank, axes (time, receiver, x/z).

    Both fields refer to integer times. Velocity is the integrator's centered
    (u[n+1]-u[n-1])/(2*dt), including its consistent startup and final lookahead.
    No continuum acceleration output is provided.
    """

    time: np.ndarray
    receiver_coordinates: np.ndarray
    receiver_names: tuple[str, ...]
    displacement: np.ndarray
    velocity: np.ndarray
    metadata: dict
    components: tuple[str, str] = ("x", "z")

    def to_xarray(self):
        """Convert without changing ordering; xarray is an optional dependency."""
        import xarray as xr

        dims = ("time", "receiver", "component")
        result = xr.Dataset(
            {
                "displacement": (dims, self.displacement, {"units": "m"}),
                "velocity": (dims, self.velocity, {"units": "m s-1"}),
            },
            coords={
                "time": ("time", self.time, {"units": "s"}),
                "receiver": list(self.receiver_names),
                "receiver_id": ("receiver", np.arange(len(self.receiver_names))),
                "receiver_x": ("receiver", self.receiver_coordinates[:, 0], {"units": "m"}),
                "receiver_z": (
                    "receiver",
                    self.receiver_coordinates[:, 1],
                    {"units": "m", "positive": "up"},
                ),
                "component": list(self.components),
            },
            attrs={
                "model": "homogeneous isotropic plane strain",
                "mpi_size": self.metadata["mpi_size"],
            },
        )
        result.velocity.attrs["time_convention"] = "integer-step centered difference"
        return result
