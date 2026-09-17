import numpy as np
import pytest

from .packets import CENTER, initial, parameters


@pytest.mark.parametrize("mode", ["P", "S"])
def test_potential_purity_and_translating_velocity(mode):
    rng = np.random.default_rng(731)
    x = CENTER + rng.normal(size=(200, 2)) * 300
    eps = 0.01
    dx = (initial(x + [eps, 0], mode)[0] - initial(x - [eps, 0], mode)[0]) / (2 * eps)
    dz = (initial(x + [0, eps], mode)[0] - initial(x - [0, eps], mode)[0]) / (2 * eps)
    div, curl = dx[:, 0] + dz[:, 1], dz[:, 0] - dx[:, 1]
    ratio = np.linalg.norm(curl) / np.linalg.norm(div)
    assert (ratio if mode == "P" else 1 / ratio) < 1e-8
    n, _, c, _, _ = parameters(mode)
    derivative = (initial(x + eps * n, mode)[0] - initial(x - eps * n, mode)[0]) / (2 * eps)
    np.testing.assert_allclose(initial(x, mode)[1], -c * derivative, atol=2e-7, rtol=1e-7)
