"""Spatial assignment is separate from the elastic polarization."""

import numpy as np

from .config import Homogeneous, SimulationConfig


def properties_at(config: SimulationConfig, z: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return rho, lambda, mu at z; interfaces take the material on increasing-z side."""
    if isinstance(config.materials, Homogeneous):
        m = config.materials.material
        return tuple(np.full_like(z, value, dtype=float) for value in (m.density, *m.lame))
    layers = config.materials.layers
    indices = np.searchsorted([layer.upper for layer in layers[:-1]], z, side="right")
    values = np.array([(layer.material.density, *layer.material.lame) for layer in layers])
    return tuple(values[indices, i] for i in range(3))
