"""Homogeneous plane-strain kernel and experiment configuration, separate from 1D."""

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .config import Config, Isotropic, TimeConfig

Count = Annotated[int, Field(ge=1, strict=True)]


class Rectangle(Config):
    """Physical (x,z) corners [m]; the second geometric slot is positive-up z."""

    lower: tuple[float, float] = (0, 0)
    upper: tuple[float, float] = (1, 1)
    cells: tuple[Count, Count] = (8, 8)
    diagonal: Literal["left", "right", "left_right", "right_left"] = "right"

    @model_validator(mode="after")
    def extent(self):
        lengths = [b - a for a, b in zip(self.lower, self.upper, strict=True)]
        if any(not math.isfinite(length) or length <= 0 for length in lengths):
            raise ValueError("Rectangle must have finite positive x and z extents")
        return self


class ZeroDisplacement(Config):
    """Constrain selected components to zero on all facets of a rectangle side."""

    side: Literal["left", "right", "lower", "upper"]
    components: tuple[Literal["x", "z"], ...] = ("x", "z")

    @model_validator(mode="after")
    def nonempty_unique(self):
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Constrained components must be nonempty and unique")
        return self


class PlaneStrainConfig(Config):
    """Vector P1 triangles, 3D isotropic solid moduli, natural or fixed-zero sides.

    There is deliberately no source, receiver, absorber, output or geology schema.
    Time integration is a Python kernel operation, not a new Simulation frontend.
    """

    domain: Rectangle = Rectangle()
    material: Isotropic
    constraints: tuple[ZeroDisplacement, ...] = ()


def unit_direction(direction):
    """Normalize finite (x,z) orientation without overflow or underflow."""
    if len(direction) != 2 or not all(math.isfinite(x) for x in direction):
        raise ValueError("Direction must contain two finite components")
    scale = max(abs(x) for x in direction)
    if scale == 0:
        raise ValueError("Source direction must be nonzero")
    scaled = tuple(x / scale for x in direction)
    norm = math.hypot(*scaled)
    return tuple(x / norm for x in scaled)


class RickerWavelet(Config):
    """Line-force wavelet: f0 [Hz], amplitude [N/m], time_shift [s].

    A*(1-2*a²)*exp(-a²), a=pi*f0*(t-time_shift). No normalization or
    causal truncation is applied; the simulation starts at t=0.
    """

    type: Literal["ricker"] = "ricker"
    f0: float = Field(gt=0)
    amplitude: float = 1.0
    time_shift: float = 0.0

    def __call__(self, time):
        import numpy as np

        time = np.asarray(time, dtype=float)
        if not np.all(np.isfinite(time)):
            raise ValueError("Wavelet evaluation time must be finite")
        # Beyond |a|=30 the exponential is below double precision range.
        # Clipping also avoids inf*0 for extreme but finite input parameters.
        with np.errstate(over="ignore"):
            a = np.clip((time - self.time_shift) * self.f0 * np.pi, -30, 30)
        return self.amplitude * ((1 - 2 * a**2) * np.exp(-(a**2)))


class ForceSource2D(Config):
    """Vector delta_2 line force, with orientation normalized internally."""

    type: Literal["force"] = "force"
    position: tuple[float, float]
    direction: tuple[float, float]
    wavelet: RickerWavelet

    @model_validator(mode="after")
    def orientation(self):
        unit_direction(self.direction)
        return self

    @property
    def unit_direction(self):
        return unit_direction(self.direction)


class Receiver2D(Config):
    """Named FE point receiver for both displacement [m] and velocity [m/s]."""

    name: str = Field(min_length=1)
    position: tuple[float, float]


class SimulationConfig2D(PlaneStrainConfig):
    """Homogeneous plane-strain experiment, zero initial data and vector P1 FE."""

    # Reuse time validation, but stability is checked on the assembled 2D operator.
    time: TimeConfig
    source: ForceSource2D | None = None
    receivers: tuple[Receiver2D, ...] = ()

    @model_validator(mode="after")
    def points_in_domain(self):
        points = [r.position for r in self.receivers]
        if self.source is not None:
            points.append(self.source.position)
        for point in points:
            if not all(
                lo <= value <= hi
                for lo, value, hi in zip(self.domain.lower, point, self.domain.upper, strict=True)
            ):
                raise ValueError("Source and receiver points must lie inside the closed rectangle")
        if len({r.name for r in self.receivers}) != len(self.receivers):
            raise ValueError("Receiver names must be unique")
        return self
