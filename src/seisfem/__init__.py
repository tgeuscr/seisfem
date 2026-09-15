"""Finite-element seismic elastodynamics. Binary runtime imports are lazy."""

from .config import SimulationConfig
from .config2d import SimulationConfig2D

__version__ = "0.5.0"
__all__ = ["Simulation", "SimulationConfig", "Simulation2D", "SimulationConfig2D", "__version__"]


def __getattr__(name):
    if name == "Simulation":
        from .simulation import Simulation

        return Simulation
    if name == "Simulation2D":
        from .simulation2d import Simulation2D

        return Simulation2D
    raise AttributeError(name)
