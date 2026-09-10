"""Immutable, serializable experiment descriptions. SI and positive-up z throughout."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0)]


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous YAML mappings instead of silently taking the last value."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Configuration YAML keys must be strings")
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate YAML mapping key")
        return super().construct_mapping(node, deep=deep)


class Config(BaseModel):
    """Reject unknown keys, nonfinite values and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Isotropic(Config):
    """Stable 3D solid: density [kg/m³], speeds [m/s] or Lamé moduli [Pa]."""

    density: Positive
    vp: Positive | None = None
    vs: Positive | None = None
    lambda_: float | None = Field(default=None, alias="lambda")
    mu: Positive | None = None

    @model_validator(mode="after")
    def physical(self) -> Isotropic:
        speeds = self.vp is not None and self.vs is not None
        lame = self.lambda_ is not None and self.mu is not None
        if not (
            (speeds and self.lambda_ is None and self.mu is None)
            or (lame and self.vp is None and self.vs is None)
        ):
            raise ValueError("Specify exactly density,vp,vs OR density,lambda,mu")
        try:
            lam, mu = self.lame
        except OverflowError as exc:
            raise ValueError("Derived elastic moduli overflow") from exc
        if not all(math.isfinite(x) for x in (lam, mu, lam + 2 * mu)):
            raise ValueError("Derived elastic moduli must be finite")
        if mu <= 0 or lam + 2 * mu / 3 <= 0:
            raise ValueError("A stable 3D solid requires mu>0 and lambda+2mu/3>0")
        return self

    @property
    def lame(self) -> tuple[float, float]:
        """Return lambda and mu in Pa without changing the input parameterization."""
        if self.vp is not None and self.vs is not None:
            return (self.density * (self.vp**2 - 2 * self.vs**2), self.density * self.vs**2)
        return float(self.lambda_), float(self.mu)

    def speed(self, mode: Literal["P", "S"]) -> float:
        """Characteristic speed of the specified elastic polarization [m/s]."""
        lam, mu = self.lame
        return math.sqrt((lam + 2 * mu if mode == "P" else mu) / self.density)


class Layer(Config):
    """One interval of solid material, ordered by increasing z [m]."""

    lower: float
    upper: float
    material: Isotropic

    @model_validator(mode="after")
    def ordered(self) -> Layer:
        if self.upper <= self.lower:
            raise ValueError("Layer upper must exceed lower")
        return self


class Homogeneous(Config):
    type: Literal["homogeneous"] = "homogeneous"
    material: Isotropic


class Layered(Config):
    type: Literal["layered"] = "layered"
    layers: tuple[Layer, ...] = Field(min_length=1)


class DomainConfig(Config):
    lower: float = 0
    upper: float
    coordinate: Literal["z_positive_up"] = "z_positive_up"

    @model_validator(mode="after")
    def ordered(self) -> DomainConfig:
        if self.upper <= self.lower or not math.isfinite(self.upper - self.lower):
            raise ValueError("Domain must have finite positive length")
        return self


class MeshConfig(Config):
    cells: int = Field(ge=2, strict=True)


class FEMConfig(Config):
    family: Literal["Lagrange"] = "Lagrange"
    degree: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def supported(self) -> FEMConfig:
        if self.degree != 1:
            raise ValueError("Only interval P1 row-sum lumping is verified; degree must be 1")
        return self


class TimeConfig(Config):
    dt: Positive
    duration: Positive
    safety: float = Field(default=0.9, gt=0, lt=1)
    integrator: Literal["central_difference"] = "central_difference"

    @model_validator(mode="after")
    def whole_steps(self) -> TimeConfig:
        ratio = self.duration / self.dt
        if (
            not math.isfinite(ratio)
            or ratio < 1
            or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("duration must be an integer multiple of dt (at least one step)")
        return self

    @property
    def steps(self) -> int:
        return round(self.duration / self.dt)


class SourceConfig(Config):
    """Planar point force: signed amplitude [Pa], frequency [Hz], shift [s]."""

    type: Literal["point_force"] = "point_force"
    position: float
    amplitude: float = 1.0
    frequency: Positive
    time_shift: float = Field(ge=0)
    f_max: Positive | None = None

    @model_validator(mode="after")
    def bandwidth(self) -> SourceConfig:
        if self.f_max is not None and self.f_max < self.frequency:
            raise ValueError("f_max must be at least the dominant frequency")
        return self


Quantity = Literal["displacement", "velocity", "acceleration"]


class ReceiverConfig(Config):
    name: str = Field(min_length=1)
    position: float
    quantities: tuple[Quantity, ...] = ("displacement", "velocity", "acceleration")

    @model_validator(mode="after")
    def unique(self) -> ReceiverConfig:
        if not self.quantities or len(set(self.quantities)) != len(self.quantities):
            raise ValueError("Receiver quantities must be nonempty and unique")
        return self


class BoundaryConfig(Config):
    lower: Literal["fixed", "free", "absorbing"] = "free"
    upper: Literal["fixed", "free", "absorbing"] = "free"


class OutputConfig(Config):
    directory: str | None = None
    receiver_stride: int = Field(default=1, ge=1, strict=True)
    snapshot_stride: int = Field(default=0, ge=0, strict=True)
    energy_stride: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def writable(self) -> OutputConfig:
        if self.directory == "":
            raise ValueError("Output directory cannot be empty")
        if self.snapshot_stride and self.directory is None:
            raise ValueError("Snapshots require an output directory")
        return self


class SimulationConfig(Config):
    """Complete reproducible 1D experiment; no runtime resources are stored here."""

    schema_version: Literal[1] = 1
    mode: Literal["P", "S"] = "P"
    domain: DomainConfig
    mesh: MeshConfig
    materials: Annotated[Homogeneous | Layered, Field(discriminator="type")]
    fem: FEMConfig = FEMConfig()
    time: TimeConfig
    source: SourceConfig | None = None
    receivers: tuple[ReceiverConfig, ...] = ()
    boundaries: BoundaryConfig = BoundaryConfig()
    output: OutputConfig = OutputConfig()

    @property
    def h(self) -> float:
        return (self.domain.upper - self.domain.lower) / self.mesh.cells

    @property
    def material_values(self) -> tuple[Isotropic, ...]:
        if isinstance(self.materials, Homogeneous):
            return (self.materials.material,)
        return tuple(layer.material for layer in self.materials.layers)

    @property
    def stable_dt(self) -> float:
        """Sufficient P1 uniform-interval step bound with configured safety [s]."""
        return self.time.safety * self.h / max(m.speed(self.mode) for m in self.material_values)

    @model_validator(mode="after")
    def experiment(self) -> SimulationConfig:
        lo, hi = self.domain.lower, self.domain.upper
        if isinstance(self.materials, Layered):
            layers = self.materials.layers
            edges = [layers[0].lower, *(layer.upper for layer in layers)]
            if edges[0] != lo or edges[-1] != hi:
                raise ValueError("Layers must exactly cover the domain")
            if any(a.upper != b.lower for a, b in zip(layers, layers[1:], strict=False)):
                raise ValueError("Layers must be ordered, contiguous and nonoverlapping")
            if any(abs((z - lo) / self.h - round((z - lo) / self.h)) > 1e-8 for z in edges):
                raise ValueError("Every layer interface must coincide with a mesh vertex")
        if self.source is not None and not lo < self.source.position < hi:
            raise ValueError("Point source must lie strictly inside the domain")
        if any(not lo <= r.position <= hi for r in self.receivers):
            raise ValueError("Every receiver must lie within the domain")
        if len({r.name for r in self.receivers}) != len(self.receivers):
            raise ValueError("Receiver names must be unique")
        if self.time.dt > self.stable_dt * (1 + 1e-12):
            raise ValueError(f"dt exceeds sufficient P1 stability limit {self.stable_dt:.8g} s")
        return self

    def diagnostics(self) -> dict:
        """Return separate stability and empirical wavelength-resolution diagnostics."""
        warnings = []
        epw = None
        if self.source is not None:
            bandwidth = self.source.f_max or 3 * self.source.frequency
            epw = min(m.speed(self.mode) for m in self.material_values) / (bandwidth * self.h)
            if epw < 12:
                warnings.append("Fewer than 12 elements/minimum wavelength; test dispersion")
            if self.source.time_shift * self.source.frequency < 1.5:
                warnings.append("Ricker startup tail is appreciable; consider time_shift >= 1.5/f0")
        return {
            "h_m": self.h,
            "safe_dt_s": self.stable_dt,
            "elements_per_min_wavelength": epw,
            "warnings": warnings,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        """Read YAML. Relative output paths are relative to the launch working directory."""
        try:
            data = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write normalized configuration including defaults, using external field names."""
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json", by_alias=True), sort_keys=False)
        )
