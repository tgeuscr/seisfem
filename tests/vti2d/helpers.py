import json
import os
from pathlib import Path

from seisfem.config2d import PlaneStrainConfig

RHO = 2500.0
C = (40e9, 25e9, 9e9, 7e9)
VTI = dict(type="vti", density=RHO, **dict(zip(["c11", "c33", "c13", "c55"], C, strict=True)))
ISO = dict(density=2200, vp=3000, vs=1700)


def config(layered=False, **updates):
    material = (
        dict(
            type="layered",
            layers=[
                dict(lower=-1, upper=0, material=ISO),
                dict(lower=0, upper=1, material=VTI),
            ],
        )
        if layered
        else VTI
    )
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(lower=(-1, -1), upper=(1, 1), cells=(4, 4)),
            material=material,
        )
        | updates
    )


def record(name, values):
    directory = os.environ.get("SEISFEM_VTI_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(values, indent=2) + "\n")
