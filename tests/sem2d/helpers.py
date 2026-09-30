import json
import os
from pathlib import Path

from seisfem.config2d import PlaneStrainConfig


def config(p=4, cells=(2, 3), extent=(1.0, 1.5)):
    return PlaneStrainConfig.model_validate(
        dict(
            domain=dict(cells=cells, upper=extent),
            material=dict(density=2.0, vp=3.0, vs=1.5),
            discretization=dict(type="quad_gll", degree=p),
        )
    )


def record(name, data):
    directory = os.environ.get("SEISFEM_SEM_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
