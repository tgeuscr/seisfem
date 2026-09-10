"""Finite-element seismic elastodynamics. Binary runtime imports are lazy."""

from .config import SimulationConfig

__version__ = "0.1.0"
__all__ = ["Simulation", "SimulationConfig", "__version__"]


def __getattr__(name):
    if name == "Simulation":
        from .simulation import Simulation

        return Simulation
    raise AttributeError(name)
