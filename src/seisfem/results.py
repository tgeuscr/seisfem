"""Named receiver arrays; numerical results do not depend on plotting libraries."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    """Rank-local receiver data with axes (time, receiver, component).

    receiver_ids index the global configuration. Missing requested quantities
    contain NaN and are identified by the requested mask (receiver, quantity).
    Energy is a global scalar diagnostic on its own half-step time axis.
    """

    time: np.ndarray
    receiver_ids: np.ndarray
    receiver_names: tuple[str, ...]
    positions: np.ndarray
    component: str
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    requested: np.ndarray
    energy_time: np.ndarray
    energy: np.ndarray
    metadata: dict

    def to_xarray(self):
        """Return an optional Xarray Dataset for this rank's receivers, with SI units."""
        import xarray as xr

        dims = ("time", "receiver", "component")
        ds = xr.Dataset(
            {
                "displacement": (dims, self.displacement, {"units": "m"}),
                "velocity": (dims, self.velocity, {"units": "m s-1"}),
                "acceleration": (dims, self.acceleration, {"units": "m s-2"}),
                "energy": (("energy_time",), self.energy, {"units": "J m-2"}),
                "requested": (("receiver", "quantity"), self.requested),
            },
            coords={
                "time": self.time,
                "receiver": list(self.receiver_names),
                "receiver_id": ("receiver", self.receiver_ids),
                "z": ("receiver", self.positions, {"units": "m", "positive": "up"}),
                "component": [self.component],
                "energy_time": self.energy_time,
                "quantity": ["displacement", "velocity", "acceleration"],
            },
            attrs={"mpi_size": self.metadata["mpi_size"], "mode": self.metadata["config"]["mode"]},
        )
        ds.time.attrs["units"] = "s"
        ds.energy_time.attrs["units"] = "s"
        return ds
