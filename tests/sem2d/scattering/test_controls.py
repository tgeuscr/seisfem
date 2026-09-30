"""No artificial interface in the identical-material limit."""

import numpy as np

from tests.sem2d.helpers import record

from .experiment import run
from .packets import Packet


def test_zero_contrast_matches_homogeneous_oblique_packet():
    packet = Packet("P", identical=True)
    layered, a = run(packet, h=600)
    homogeneous, b = run(packet, h=600, homogeneous=True)
    differences = {
        "field_absolute": float(np.max(abs(a - b))),
        "history_absolute": float(np.max(abs(np.array(layered["trace"]) - homogeneous["trace"]))),
        "safe_dt_absolute": abs(layered["safe_dt"] - homogeneous["safe_dt"]),
    }
    # Scattered field means the field minus the identically initialized
    # homogeneous control, not the remaining incident Gaussian tail.
    axis = np.linspace(-packet.analysis, packet.analysis, len(a))
    differences["reflected_scattered_field"] = float(np.max(abs((a - b)[axis < 0, :, :2])))
    assert max(differences.values()) == 0
    record(
        "scattering-zero-contrast",
        dict(
            errors=differences,
            h=600,
            degree=4,
            angle=packet.angle,
            energy_drift=layered["energy_drift"],
            definition="layered minus homogeneous field with identical oblique initial data",
        ),
    )
