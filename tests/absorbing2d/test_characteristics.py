import numpy as np
import pytest

from tests.absorbing2d.characterize import continuum_reflection


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 75])
def test_independent_plane_wave_boundary_flux(angle):
    free = continuum_reflection(angle, absorbing=False)
    absorbing = continuum_reflection(angle)
    assert free["energy_flux_ratio"] == pytest.approx(1, abs=7e-16)
    assert 0 <= absorbing["energy_flux_ratio"] < 1
    if angle == 0:
        np.testing.assert_array_equal([absorbing["P"], absorbing["SV"]], 0)
    if angle >= 60:
        assert absorbing["energy_flux_ratio"] > 0.03
