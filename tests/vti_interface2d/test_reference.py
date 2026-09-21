import numpy as np
import pytest

from tests.oblique2d import reference as isotropic

from .reference import LOWER, NAMES, UPPER, Material, candidates, residual, solve, squared_roots


@pytest.mark.parametrize("angle", [0, 5, 15, 25, 35, -15, -25, -35])
def test_roots_continuity_flux_and_symmetry(angle):
    p = np.sin(np.deg2rad(angle)) / 3000
    inc, branches = solve(p)
    for m in [LOWER, UPPER]:
        roots = candidates(p, m)
        assert max(abs(roots["determinant"])) < 3e-14
        assert max(roots["residual"]) < 3e-14
        np.testing.assert_allclose(roots["flux"], 0.5 * m.rho * roots["g"][:, 1], rtol=3e-15)
        # Independent finite-difference group derivative of the Christoffel eigenvalue.
        for j, q in enumerate(roots["q"]):
            k = np.array([p, q])
            d = roots["d"][j]

            def frequency(k, m=m, kind=roots["kind"][j]):
                x, z = k
                D = np.array(
                    [
                        [m.c11 * x * x + m.c55 * z * z, (m.c13 + m.c55) * x * z],
                        [(m.c13 + m.c55) * x * z, m.c55 * x * x + m.c33 * z * z],
                    ]
                )
                ev = np.linalg.eigvalsh(D)
                return np.sqrt(ev[kind] / m.rho)

            step = 1e-9
            group = np.array(
                [
                    (frequency(k + step * e) - frequency(k - step * e)) / (2 * step)
                    for e in np.eye(2)
                ]
            )
            np.testing.assert_allclose(group, roots["g"][j], rtol=2e-9, atol=1e-6)
            assert abs(np.linalg.norm(d) - 1) < 5e-16
    assert inc["g"][1] > 0
    assert max(abs(residual(p, branches))) < 1e-14
    assert abs(sum(w["fraction"] for w in branches.values()) - 1) < 3e-14
    _, negative = solve(-p)
    for name, w in branches.items():
        assert w["g"][1] * (1 if name[0] == "T" else -1) > 0
        np.testing.assert_allclose(
            w["amplitude"], negative[name]["amplitude"] * (1 if name[1] == "P" else -1), atol=2e-15
        )
        np.testing.assert_allclose(w["fraction"], negative[name]["fraction"], atol=2e-15)


@pytest.mark.parametrize("mode,angles", [("P", [0, 15, 25, 35]), ("S", [0, 5, 15, 20])])
def test_independent_isotropic_zoeppritz_limit(mode, angles):
    lower = Material.isotropic(isotropic.LOWER.rho, isotropic.LOWER.vp, isotropic.LOWER.vs)
    upper = Material.isotropic(isotropic.UPPER.rho, isotropic.UPPER.vp, isotropic.UPPER.vs)
    for angle in angles:
        p = np.sin(np.deg2rad(angle)) / isotropic.LOWER.speed(mode)
        _, new = solve(p, incident=mode, lower=lower, upper=upper)
        old = isotropic.solve(mode, p=p)
        for name in NAMES:
            np.testing.assert_allclose(new[name]["amplitude"], old[name]["amplitude"], atol=3e-15)
            np.testing.assert_allclose(new[name]["fraction"], old[name]["flux"], atol=3e-15)
            np.testing.assert_allclose(new[name]["n"], old[name]["direction"], atol=3e-15)
            np.testing.assert_allclose(new[name]["d"], old[name]["polarization"], atol=3e-15)
            speed = (isotropic.LOWER if name[0] == "R" else isotropic.UPPER).speed(name[1])
            assert abs(new[name]["q"] - old[name]["direction"][1] / speed) < 3e-18
            np.testing.assert_allclose(
                new[name]["g"] / speed, old[name]["direction"], rtol=0, atol=3e-15
            )


def test_normal_impedance_reduction_and_critical_rejection():
    _, b = solve(0.0)
    z1 = np.sqrt(LOWER.rho * LOWER.c33)
    z2 = np.sqrt(UPPER.rho * UPPER.c33)
    np.testing.assert_allclose(b["RP"]["amplitude"], (z2 - z1) / (z1 + z2), atol=2e-16)
    np.testing.assert_allclose(b["TP"]["amplitude"], 2 * z1 / (z1 + z2), atol=2e-16)
    assert b["RS"]["amplitude"] == b["TS"]["amplitude"] == 0
    with pytest.raises(ValueError, match="propagating"):
        solve(np.sin(np.deg2rad(55)) / 3000)
    scan = np.linspace(-35, 35, 701)
    assert np.all(squared_roots(np.sin(np.deg2rad(scan)) / 3000, UPPER)[1])
