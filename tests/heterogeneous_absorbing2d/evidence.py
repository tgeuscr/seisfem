"""Optional output of freshly computed diagnostics; tests never read this evidence."""

import json
import os
from pathlib import Path


def record(name, result):
    directory = os.environ.get("SEISFEM_HETERO_ABSORBER_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
