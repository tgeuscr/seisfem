"""Canonical in-plane VTI stiffnesses; no 1D or SH constitutive extension."""

import math
from typing import Literal

from pydantic import Field, model_validator

from .config import Config, Isotropic, Layer, Layered, Positive


class VTI(Config):
    """Vertical-axis plane-strain solid: density [kg/m³], stiffnesses [Pa].

    Positive in-plane strain energy is required with a double-precision margin.
    This is not a complete 3D TI material: C66 and gamma are intentionally absent.
    """

    type: Literal["vti"]
    density: Positive
    c11: Positive
    c33: Positive
    c13: float
    c55: Positive

    @model_validator(mode="after")
    def admissible(self):
        scale = max(self.c11, self.c33, abs(self.c13), self.c55)
        a, b, c, s = (x / scale for x in self.stiffnesses)
        largest = (a + b + math.hypot(a - b, 2 * c)) / 2
        if largest == 0:
            raise ValueError("VTI normal stiffnesses underflow relative to the material scale")
        smallest = (a * b - c * c) / largest
        if min(smallest, s) <= 64 * math.ulp(1.0):
            raise ValueError("VTI in-plane strain energy must be positive with a numerical margin")
        if not all(
            math.isfinite(x / self.density) and x / self.density > 0
            for x in (scale, scale * smallest, self.c55)
        ) or not math.isfinite(self.c11 + self.c33 + 2 * self.c55):
            raise ValueError("VTI stiffness and squared-speed scales must be finite and positive")
        return self

    @property
    def stiffnesses(self):
        return self.c11, self.c33, self.c13, self.c55


class LayerVTI(Layer):
    material: Isotropic | VTI


class LayeredVTI(Layered):
    """Mixed layers; all-isotropic inputs continue to use the original Layered."""

    layers: tuple[LayerVTI, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def includes_vti(self):
        if not any(isinstance(layer.material, VTI) for layer in self.layers):
            raise ValueError("Use the existing isotropic layered representation without VTI")
        return self


def stiffnesses(material):
    """Canonicalize an isotropic layer only inside an anisotropic assembly."""
    if isinstance(material, VTI):
        return material.stiffnesses
    lam, mu = material.lame
    return lam + 2 * mu, lam + 2 * mu, lam, mu
