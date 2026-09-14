"""Narrow homogeneous plane-strain kernel configuration, separate from 1D runs."""

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .config import Config, Isotropic

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
